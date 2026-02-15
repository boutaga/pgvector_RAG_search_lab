# Lab 5 -- Embedding Versioning & Event-Driven Refresh

Companion lab for the blog post **"Embedding Versioning with pgvector"**.
This lab walks you through a production-grade **triggers + queue + worker** pipeline that keeps embeddings fresh as source documents change. You will simulate content edits, watch PostgreSQL triggers populate a queue, run workers that call OpenAI, and observe how the system decides which changes actually need re-embedding.

**What you will learn:**

- How PostgreSQL triggers detect content changes via MD5 hashing
- How a SKIP LOCKED queue replaces external message brokers
- How to filter trivial edits (typo fixes) from meaningful ones before spending API credits
- How versioned embeddings enable blue-green model upgrades with instant rollback
- How a quality feedback loop detects stale embeddings and triggers re-processing

## Architecture

```
  Article UPDATE                    Queue Worker
  ----------------                  ------------
  content changes                   polls / listens
       |                                 |
       v                                 v
  trg_content_hash          +---- embedding_queue <----- change_detector
  (md5 + updated_at)        |     (SKIP LOCKED)         (EMBED / SKIP)
       |                    |            |
       v                    |            v
  trg_queue_embedding       |     OpenAI Embeddings API
  (INSERT into queue)       |            |
       |                    |            v
       v                    |   article_embeddings_versioned
  pg_notify                 |     (is_current = true/false)
  ('embedding_updates')     |            |
                            |            v
                            |     DiskANN index
                            |     (similarity search)
                            |
                            +---- freshness_monitor
                                  quality_feedback_loop
```

Read the diagram left-to-right: an article update fires two triggers (content hash, then queue insert), a worker picks the item up, calls OpenAI, and stores the new embedding. Old embeddings are kept for rollback.

## Prerequisites

- PostgreSQL with `pgvector` and `pgvectorscale` extensions
- Wikipedia database loaded (25K articles in the `articles` table)
- Python 3.11+
- OpenAI API key (for embedding generation)
- Linux/macOS/WSL2 (the LISTEN/NOTIFY worker uses `select.select()` which is not supported on native Windows)

---

## Phase 1: Setup

### Step 1 -- Apply the schema

```bash
psql -d wikipedia -f schema.sql
```

This adds `content_hash` and `updated_at` columns to `articles`, creates the `article_embeddings_versioned` table with a DiskANN index, creates the `embedding_queue` with SKIP LOCKED support, installs triggers for automatic queue population, and backfills `content_hash` for all existing articles.

### Step 2 -- Verify the schema

```sql
-- Content hash should be populated for all articles
SELECT count(*) AS hashed FROM articles WHERE content_hash IS NOT NULL;

-- All four new tables should exist
\dt article_embeddings_versioned
\dt embedding_queue
\dt embedding_change_log
\dt retrieval_quality_log
```

Expected output for the first query:

```
 hashed
--------
  25000
(1 row)
```

If the count is 0, the backfill in `schema.sql` did not run. Re-run it manually:

```sql
UPDATE articles SET content_hash = md5(content) WHERE content_hash IS NULL;
```

### Step 3 -- Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 4 -- Set environment variables

```bash
export DATABASE_URL="postgresql://user:pass@host:5432/wikipedia"
export OPENAI_API_KEY="sk-..."
```

Replace user, pass, and host with your actual connection details.

---

## Phase 2: The Core Pipeline

This phase walks through the full cycle: content change, trigger, queue, change detection, embedding, verification.

### Step 1 -- Trigger the pipeline

```bash
python examples/simulate_document_changes.py --count 50
```

This applies random mutations to 50 articles. The mutations follow a realistic distribution:

| Type            | Weight | Expected decision |
|-----------------|--------|-------------------|
| typo_fix        | 40%    | SKIP              |
| metadata_only   | 20%    | SKIP              |
| paragraph_add   | 25%    | EMBED             |
| section_rewrite | 10%    | EMBED             |
| major_rewrite   | 5%     | EMBED             |

Each mutation fires the `trg_queue_embedding_update` trigger, which inserts a row into `embedding_queue` and sends a NOTIFY on the `embedding_updates` channel.

Expected output:

```
Article 4821: typo_fix applied
Article 12003: paragraph_add applied
Article 7519: typo_fix applied
...

Summary:
  typo_fix:        22
  metadata_only:    9
  paragraph_add:   12
  section_rewrite:  5
  major_rewrite:    2
  Total:           50
```

Use `--dry-run` to preview without applying changes, or `--type paragraph_add` to force a specific mutation type.

> **What you just learned:** PostgreSQL triggers detect content changes automatically. You never have to remember to queue an embedding update -- it happens at the database level.

### Step 2 -- Inspect the queue

Run these queries to see what the triggers created:

```sql
-- Status distribution (all should be 'pending')
SELECT status, count(*) FROM embedding_queue GROUP BY status;

-- Change types match the mutations we applied
SELECT change_type, count(*) FROM embedding_queue GROUP BY change_type;
```

Expected output:

```
  status  | count
----------+-------
 pending  |    50

 change_type    | count
----------------+-------
 content_update |    50
```

All 50 items are `pending` and typed as `content_update` because the trigger saw article content change. New articles would show as `new`.

### Step 3 -- Filter trivial changes

```bash
python change_detector.py --analyze-queue
```

The change detector examines each pending queue item and decides whether the content change is significant enough to justify an API call. It compares old content (reconstructed from stored embedding chunks) against new content using text similarity and structural analysis.

Decision logic:

- Similarity >= 0.95 (typo-level change): **SKIP**
- New headings added or removed: **EMBED**
- Length changed by more than 20%: **EMBED**
- Paragraph-level similarity < 0.85: **EMBED**
- Otherwise: **EMBED** if below threshold, **SKIP** if above

Expected output:

```
Analyzing 50 pending queue items (threshold=0.95)
Article 4821: SKIP (similarity=0.9987)
Article 12003: EMBED (similarity=0.8234)
Article 7519: SKIP (similarity=0.9991)
...
Results: 19 EMBED, 31 SKIP
```

Roughly 60% of items should be SKIP (typo fixes + metadata-only changes). This saves API credits by avoiding re-embedding when the meaning hasn't changed.

Verify the decisions in the database:

```sql
SELECT decision, count(*), round(avg(similarity)::numeric, 4) AS avg_sim
FROM embedding_change_log
GROUP BY decision;
```

Expected output:

```
 decision | count | avg_sim
----------+-------+---------
 EMBED    |    19 |  0.7842
 SKIP     |    31 |  0.9973
```

Skipped items have high similarity (near 1.0). Embedded items have lower similarity, meaning the content genuinely changed.

Use `--threshold 0.90` to lower the bar (more items get embedded) or `--threshold 0.99` to raise it (only major changes get embedded).

> **What you just learned:** Not every content edit needs re-embedding. By analyzing text similarity and structure before calling OpenAI, you can cut API costs by 50-60% on typical workloads without sacrificing search quality.

### Step 4 -- Process embeddings

The worker picks up items still marked `pending` (the ones the change detector decided should be embedded) and calls OpenAI to generate new embeddings.

**Single-batch mode** (recommended for this walkthrough):

```bash
python worker.py --once
```

This processes one batch and exits. Expected output:

```
Worker main started (model=text-embedding-3-small, batch=10)
Worker main claimed 10 items
Article 12003: embedded 3 chunks
Article 8291: embedded 2 chunks
...
Worker main: processed 10 items, stopping (--once mode)
```

Each article is chunked (2000 chars with 200 char overlap), and each chunk gets its own embedding. Old embeddings for the same article are marked `is_current = false` with a `replaced_at` timestamp.

**Other modes** (for reference):

```bash
# Continuous polling (runs until Ctrl+C, polls every 5 seconds)
python worker.py

# Multi-worker (4 parallel processes, each uses SKIP LOCKED)
python worker.py --workers 4

# LISTEN/NOTIFY mode (instant reaction, no polling delay)
python worker_notify.py
```

The multi-worker and LISTEN/NOTIFY modes are covered in the Deep Dives section.

> **What you just learned:** The worker uses `SELECT FOR UPDATE SKIP LOCKED` to claim queue items safely. Multiple workers can run in parallel without any external coordinator -- PostgreSQL handles the locking.

### Step 5 -- Verify results

Check that embeddings were created:

```sql
-- How many articles have current embeddings
SELECT count(DISTINCT article_id)
FROM article_embeddings_versioned
WHERE is_current = true;

-- Recent embeddings with chunk counts
SELECT article_id, count(*) AS chunks, max(created_at) AS last_embed
FROM article_embeddings_versioned
WHERE is_current = true
GROUP BY article_id
ORDER BY last_embed DESC
LIMIT 10;
```

Run the freshness monitor for a comprehensive report:

```bash
python freshness_monitor.py --report
```

Expected output (sections):

```
=== Freshness Summary ===
  Total articles:     25000
  Embedded:              19
  Not embedded:       24981
  Stale:                  0
  Coverage:           0.08%
  Staleness:          0.00%

=== Queue Health ===
  Status     | Count | Oldest           | Newest
  completed  |    19 | 2026-02-15 10:01 | 2026-02-15 10:03
  skipped    |    31 | 2026-02-15 10:01 | 2026-02-15 10:01

=== Change Decisions ===
  Decision | Count | Avg Similarity | First             | Last
  EMBED    |    19 |         0.7842 | 2026-02-15 10:01  | 2026-02-15 10:01
  SKIP     |    31 |         0.9973 | 2026-02-15 10:01  | 2026-02-15 10:01
```

Use `--stale` to see only stale articles, or `--queue` for queue health only.

> **What you just learned:** The freshness monitor gives you a single dashboard view of your embedding pipeline health: coverage, staleness, queue state, and change decision history.

---

## Phase 3: Deep Dives

Each deep dive demonstrates a specific capability of the system. They can be run independently in any order, but they assume Phase 1 (setup) is complete.

### Deep Dive A -- End-to-end trigger flow

See the full pipeline in a single script: pick one article, modify it, watch the trigger fire, and optionally embed it.

```bash
# Without OpenAI (shows trigger + queue only)
python examples/demo_trigger_flow.py --skip-embed

# With OpenAI (full pipeline including embedding)
python examples/demo_trigger_flow.py
```

Expected output (`--skip-embed`):

```
=== Article Selected ===
  ID: 7412
  Title: History of computing
  Content length: 14,832 chars

=== Before Update ===
  Pending queue items: 0

=== Applying Content Change ===
  Appended demo text to article

=== Verifying Trigger ===
  Queue entry created: id=51, status=pending, change_type=content_update
  Content hash updated: 8a3f...
  updated_at set: 2026-02-15 10:15:03

=== Skipping embedding (--skip-embed) ===
```

Use `--article-id 1234` to target a specific article instead of a random one.

> **What you just learned:** The trigger fires synchronously within the UPDATE transaction. By the time your UPDATE commits, the queue entry already exists.

### Deep Dive B -- Change significance analysis

See how the change detector evaluates different types of edits against the same article.

```bash
python examples/demo_change_significance.py
```

Expected output:

```
=== Article: History of computing (id=7412, 14832 chars) ===

Change Type          | Decision | Similarity | Para Sim | Len Ratio
---------------------|----------|------------|----------|-----------
Typo Fix (1 char)    | SKIP     |     0.9999 |   1.0000 |    1.0000
Whitespace Cleanup   | SKIP     |     0.9985 |   1.0000 |    0.9998
Append Sentence      | SKIP     |     0.9962 |   0.9945 |    0.9978
Add New Paragraph    | EMBED    |     0.9201 |   0.9100 |    0.9524
Rewrite Middle 20%   | EMBED    |     0.7856 |   0.7200 |    0.9800
Major Rewrite        | EMBED    |     0.3241 |   0.2800 |    0.8100

=== Structural Analysis (Major Rewrite) ===
  paragraph_similarity: 0.2800
  headings_added: 3
  headings_removed: 2
  length_ratio: 0.8100
  char_diff: 2812
```

Use `--article-id 500` to pick a specific article.

> **What you just learned:** The SKIP/EMBED threshold at 0.95 catches typos and whitespace edits but lets through anything that changes meaning. Structural analysis (headings, paragraphs) catches reorganizations even when raw similarity stays high.

### Deep Dive C -- Concurrency with SKIP LOCKED

Prove that multiple workers never process the same queue item by running 4 workers against 50 items.

```bash
python examples/demo_skip_locked.py --workers 4 --items 50
```

Expected output:

```
Added 50 demo queue entries (total pending: 50)

demo-worker-0: claimed 13 items (articles: [1201, 3045, ...])
demo-worker-1: claimed 12 items (articles: [892, 4501, ...])
demo-worker-2: claimed 13 items (articles: [2341, 6789, ...])
demo-worker-3: claimed 12 items (articles: [110, 5432, ...])

Total claimed: 50, Unique: 50, Elapsed: 0.34s
ZERO OVERLAP -- SKIP LOCKED working correctly!
```

No OpenAI calls are made -- this demo simulates the work. The key result is "ZERO OVERLAP", proving that PostgreSQL's `SKIP LOCKED` prevents double-processing without any external coordination.

> **What you just learned:** `SELECT FOR UPDATE SKIP LOCKED` is a production-grade concurrency primitive built into PostgreSQL. You can scale embedding workers horizontally by simply starting more processes.

### Deep Dive D -- Blue-green model upgrade

Walk through a zero-downtime model upgrade from `text-embedding-3-small` to `text-embedding-3-large`.

```bash
# Without OpenAI (shows queue + cutover/rollback mechanics)
python examples/demo_model_upgrade.py --limit 10

# With OpenAI (actually generates new embeddings)
python examples/demo_model_upgrade.py --limit 10 --embed
```

Expected output (without `--embed`):

```
=== Current Model Status ===
  Model                    | Articles | Chunks | Current
  text-embedding-3-small   |       19 |     47 | Yes

=== Queuing Upgrade ===
  Queued 10 articles for text-embedding-3-large

=== Skipping embedding (no --embed flag) ===

=== Demonstrating Cutover ===
  Cutover: retired 0 old embedding rows
  (No new embeddings to cut over to -- use --embed to generate them)

=== Demonstrating Rollback ===
  Rollback: restored 0 rows
```

With `--embed`, the script actually calls OpenAI to generate embeddings with the new model, then demonstrates a live cutover (old model marked non-current) and rollback (old model restored).

For a manual upgrade workflow using the core tools:

```bash
# 1. Queue articles for the new model
python model_upgrade.py --queue-upgrade --new-model text-embedding-3-large --limit 100

# 2. Process the queue (generates new embeddings)
python worker.py --model text-embedding-3-large

# 3. Compare search quality between models
python model_upgrade.py --compare

# 4. If satisfied, cut over
python model_upgrade.py --cutover --new-model text-embedding-3-large

# 5. If something goes wrong, rollback
python model_upgrade.py --rollback --old-model text-embedding-3-small
```

> **What you just learned:** Blue-green deployment for embeddings means you can generate new model embeddings alongside old ones, compare search quality, and switch atomically. Rollback is instant because old embeddings are never deleted.

### Deep Dive E -- Quality drift detection

Simulate user feedback indicating poor search results, then watch the quality feedback loop identify and re-queue affected articles.

```bash
python examples/demo_quality_drift.py --simulate-queries 20
```

Expected output:

```
=== Current Staleness ===
  Articles with stale embeddings: 3

=== Simulating Quality Feedback ===
  Generated 20 quality log entries
  8 poor results (nDCG < 0.5 or negative feedback)
  12 good results

=== Quality Summary ===
  Model                  | Queries | Avg nDCG | Negative
  text-embedding-3-small |      20 |   0.6234 |        5

=== Poor Queries Found ===
  "history of computing" (nDCG=0.32, feedback=negative)
  "quantum physics basics" (nDCG=0.41, feedback=negative)
  ...

=== Affected Articles Identified ===
  Article 7412 (History of computing) -- stale hash
  Article 2103 (Quantum mechanics) -- low nDCG correlation

=== Re-embedding Queued ===
  Queued 2 articles for re-embedding (priority=2, change_type=quality_reembed)

=== Resulting Queue ===
  status  | count | change_type
  pending |     2 | quality_reembed
```

Quality-triggered re-embeddings get priority 2 (highest), so they are processed before regular content updates (priority 5).

> **What you just learned:** A feedback loop closes the gap between "embeddings exist" and "embeddings work well". When search quality drops, the system can automatically identify which articles to re-embed.

---

## Key Concepts

### Content Hash Trigger
Every content change on `articles` auto-computes `md5(content)` and updates `updated_at`. This is the foundation for staleness detection.

### Embedding Queue
A PostgreSQL-native job queue using `SELECT FOR UPDATE SKIP LOCKED` for safe multi-worker concurrency without external message brokers.

### Change Significance
Not all changes need re-embedding. Typo fixes (similarity > 0.95) are skipped. Structural changes (new sections, major rewrites) always trigger re-embedding.

### Versioned Embeddings
Old embeddings are kept (`is_current = false`) for rollback and auditing. Only current embeddings are indexed by DiskANN.

### Blue-Green Model Upgrade
New model embeddings are created alongside old ones. Compare quality, then cutover atomically. Rollback restores the old version.

### Embedding Dimensions
The schema defaults to `vector(1536)` for `text-embedding-3-small`. If upgrading to `text-embedding-3-large` (3072 dims), you must alter the column first:
```sql
ALTER TABLE article_embeddings_versioned
    ALTER COLUMN embedding TYPE vector(3072);
```

### Queue Priorities
Items are processed in priority order (lower number = higher priority):

| Priority | Change type      | Meaning                        |
|----------|------------------|--------------------------------|
| 2        | quality_reembed  | Quality feedback flagged issue |
| 3        | new              | New article inserted           |
| 4        | model_upgrade    | Blue-green model migration     |
| 5        | content_update   | Regular content edit           |

---

## CLI Reference

| Script | Key flags | Purpose |
|--------|-----------|---------|
| `examples/simulate_document_changes.py` | `--count N`, `--type TYPE`, `--dry-run` | Generate test mutations |
| `change_detector.py` | `--analyze-queue`, `--threshold F`, `--limit N` | Decide EMBED vs SKIP |
| `worker.py` | `--once`, `--workers N`, `--batch-size N`, `--model M` | Process queue (polling) |
| `worker_notify.py` | `--batch-size N`, `--model M` | Process queue (LISTEN/NOTIFY) |
| `freshness_monitor.py` | `--report`, `--stale`, `--queue` | Staleness diagnostics |
| `model_upgrade.py` | `--queue-upgrade`, `--compare`, `--cutover`, `--rollback`, `--status` | Blue-green model management |
| `quality_feedback_loop.py` | `--check`, `--ndcg-threshold F`, `--lookback-hours N` | Quality-based re-embedding |
| `examples/demo_trigger_flow.py` | `--skip-embed`, `--article-id N` | End-to-end pipeline demo |
| `examples/demo_skip_locked.py` | `--workers N`, `--items N` | Concurrency proof |
| `examples/demo_change_significance.py` | `--article-id N` | SKIP vs EMBED decision demo |
| `examples/demo_model_upgrade.py` | `--embed`, `--limit N` | Blue-green lifecycle demo |
| `examples/demo_quality_drift.py` | `--simulate-queries N` | Quality feedback demo |

All scripts accept `--db-url` to override `DATABASE_URL`.

---

## Troubleshooting

**DiskANN index creation fails**
pgvectorscale is not installed. Install it or comment out the `CREATE INDEX ... USING diskann` line in `schema.sql` and use HNSW instead:
```sql
CREATE INDEX ix_embed_hnsw ON article_embeddings_versioned
    USING hnsw (embedding vector_cosine_ops)
    WHERE is_current = true;
```

**No OPENAI_API_KEY set**
Everything except actual embedding generation works without an API key. You can run the trigger flow (`--skip-embed`), change detection, concurrency demo, model upgrade mechanics (`--embed` omitted), and freshness monitoring. Only `worker.py`, `worker_notify.py`, and the `--embed` flag on demos require the key.

**Queue items stuck in "processing"**
A worker crashed mid-processing. The worker automatically recovers stuck items on startup (anything in "processing" for more than 10 minutes gets reset to "pending"). You can also reset manually:
```sql
UPDATE embedding_queue
SET status = 'pending', started_at = NULL, worker_id = NULL
WHERE status = 'processing'
  AND started_at < now() - interval '10 minutes';
```

**No articles in database**
The Wikipedia dataset must be loaded before running this lab. The `articles` table should contain approximately 25,000 rows. Check with:
```sql
SELECT count(*) FROM articles;
```

---

## Connection to Blog Post

This lab implements **Level 1** of the embedding versioning hierarchy described in the blog post "Embedding Versioning with pgvector":

- **Level 1** (this lab): Triggers + Queue + Worker
- **Level 2**: CDC-based (Debezium/pglogical)
- **Level 3**: Full platform (Airflow, Kubernetes, observability)
