# Lab 5 — Embedding Versioning & Event-Driven Refresh

Level 1 implementation of embedding versioning using **triggers + queue + worker** pattern with PostgreSQL, pgvector, and pgvectorscale (DiskANN).

## Architecture

```
  Article UPDATE                    Queue Worker
  ────────────                      ────────────
  content changes                   polls / listens
       │                                 │
       ▼                                 ▼
  trg_content_hash          ┌──── embedding_queue ◄───── change_detector
  (md5 + updated_at)        │     (SKIP LOCKED)         (EMBED / SKIP)
       │                    │            │
       ▼                    │            ▼
  trg_queue_embedding       │     OpenAI Embeddings API
  (INSERT into queue)       │            │
       │                    │            ▼
       ▼                    │   article_embeddings_versioned
  pg_notify                 │     (is_current = true/false)
  ('embedding_updates')     │            │
                            │            ▼
                            │     DiskANN index
                            │     (similarity search)
                            │
                            └──── freshness_monitor
                                  quality_feedback_loop
```

## Prerequisites

- PostgreSQL with `pgvector` and `pgvectorscale` extensions
- Wikipedia database loaded (25K articles in `articles` table)
- Python 3.11+
- OpenAI API key (for embedding generation)

## Setup

### 1. Apply the schema

```bash
psql -d wikipedia -f schema.sql
```

This will:
- Add `content_hash` and `updated_at` columns to `articles`
- Create the `article_embeddings_versioned` table with DiskANN index
- Create the `embedding_queue` with SKIP LOCKED support
- Set up triggers for automatic queue population
- Backfill `content_hash` for all existing articles

### 2. Verify

```sql
-- Check new columns
\d articles

-- Check content_hash was backfilled
SELECT count(*) FROM articles WHERE content_hash IS NOT NULL;

-- Check tables were created
\dt article_embeddings_versioned
\dt embedding_queue
\dt embedding_change_log
\dt retrieval_quality_log
```

### 3. Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Set environment variables

```bash
export DATABASE_URL="postgresql://user:pass@host:5432/wikipedia"
export OPENAI_API_KEY="sk-..."
```

## Step-by-Step Demo Flow

### Step 1: Simulate document changes

```bash
python examples/simulate_document_changes.py --count 50
```

Applies random mutations (typo fixes, paragraph additions, major rewrites) to articles. The trigger fires automatically, creating queue entries.

### Step 2: Check the queue

```sql
SELECT status, count(*) FROM embedding_queue GROUP BY status;
SELECT change_type, count(*) FROM embedding_queue GROUP BY change_type;
```

### Step 3: Run change significance detector

```bash
python change_detector.py --analyze-queue
```

Analyzes pending items and marks trivial changes as SKIP. Only significant changes remain for embedding.

### Step 4: Check skip rate

```sql
SELECT decision, count(*), round(avg(similarity), 4) AS avg_sim
FROM embedding_change_log
GROUP BY decision;
```

### Step 5: Run the embedding worker

```bash
# Polling mode (single worker)
python worker.py --once

# Or continuous mode
python worker.py

# Or multi-worker
python worker.py --workers 4

# Or LISTEN/NOTIFY mode (instant reaction)
python worker_notify.py
```

### Step 6: Verify embeddings

```sql
SELECT count(*) FROM article_embeddings_versioned WHERE is_current = true;

SELECT article_id, count(*) AS chunks, max(created_at) AS last_embed
FROM article_embeddings_versioned
WHERE is_current = true
GROUP BY article_id
ORDER BY last_embed DESC
LIMIT 10;
```

### Step 7: Run freshness monitor

```bash
python freshness_monitor.py --report
```

### Step 8: Demo concurrency (SKIP LOCKED)

```bash
python examples/demo_skip_locked.py --workers 4 --items 50
```

## Additional Demos

### End-to-end trigger flow
```bash
python examples/demo_trigger_flow.py
python examples/demo_trigger_flow.py --skip-embed  # without OpenAI
```

### Change significance analysis
```bash
python examples/demo_change_significance.py
```

### Blue-green model upgrade
```bash
python examples/demo_model_upgrade.py
python examples/demo_model_upgrade.py --embed  # with OpenAI
```

### Quality drift detection
```bash
python examples/demo_quality_drift.py --simulate-queries 20
```

## File Reference

| File | Purpose |
|------|---------|
| `schema.sql` | Database tables, triggers, indexes, queue |
| `worker.py` | Queue-based embedding worker (SKIP LOCKED polling) |
| `worker_notify.py` | LISTEN/NOTIFY variant of the worker |
| `change_detector.py` | Change significance analysis (EMBED vs SKIP) |
| `freshness_monitor.py` | Staleness monitoring with 7 diagnostic queries |
| `quality_feedback_loop.py` | Quality-based re-embedding trigger |
| `model_upgrade.py` | Blue-green model version management |
| `examples/simulate_document_changes.py` | Generate test mutations |
| `examples/demo_trigger_flow.py` | End-to-end: modify -> queue -> embed |
| `examples/demo_skip_locked.py` | Multi-worker concurrency demo |
| `examples/demo_change_significance.py` | Show SKIP vs EMBED decisions |
| `examples/demo_model_upgrade.py` | Blue-green switch demo |
| `examples/demo_quality_drift.py` | Staleness detection via nDCG |

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

## Connection to Blog Post

This lab implements **Level 1** of the embedding versioning hierarchy described in the blog post "Embedding Versioning with pgvector":

- **Level 1** (this lab): Triggers + Queue + Worker
- **Level 2**: CDC-based (Debezium/pglogical)
- **Level 3**: Full platform (Airflow, Kubernetes, observability)
