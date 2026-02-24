# Lab 04 — Full Skeleton & Blueprint

> **Purpose:** This document is the spec for building Lab 04 with Claude Code.
> Draft code already exists in this directory — review it, fix it, test it end-to-end.
> This skeleton captures every architectural decision, gotcha, and edge case.

---

## 1. THE STORY (for the presentation)

**Slide 29 (Before):**
Data lake holds everything in Parquet on S3. BI analyst wants a KPI. They file a ticket → data engineer writes SQL → tests → deploys → 2 weeks later the analyst gets their dashboard. Or they try to query the lake directly → 10 min query, BI connector timeout, no indexes, no JOINs.

**Slide 30 (After — this lab):**
BI analyst types: "Show me portfolio exposure by sector and client segment."

1. **Agent 1 (RAG Search)** embeds the question → searches pgvector metadata catalog → returns recommended tables, columns, joins, governance flags. **Never touches raw data.**
2. **Agent 2 (Pipeline)** reads recommended Parquet files from S3 → stages them temporarily in PG → generates governed `CREATE TABLE AS SELECT` → applies PII masking + classification grants → drops staging → logs audit trail.
3. **Result:** Governed data mart in PostgreSQL. Fast JOINs, proper indexes, seconds instead of minutes. Full audit trail for FINMA compliance.

**The punchline:** PostgreSQL at the center — one database handles metadata search (pgvector), governance (audit, masking, RLS), quality monitoring (precision/recall/nDCG), AND the final data mart (compute). Right tool for the right job: S3 for storage, PG for compute.

---

## 2. ARCHITECTURE

```
┌─────────────────┐
│  MinIO (S3)     │  ← Parquet data lake (storage layer)
│  Port 9000/9001 │     14 tables, ~4K total rows (demo scale)
└────────┬────────┘
         │ boto3 + pyarrow
         │
┌────────▼───────────────────────────────────────────────────────────┐
│  PostgreSQL 18 + pgvector + pgvectorscale                          │
│  Port 5433 — database: metadata_catalog                            │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ catalog.*    │  │ governance.* │  │ rag_monitor.*│             │
│  │ metadata +   │  │ audit trail  │  │ search logs  │             │
│  │ embeddings   │  │ mart registry│  │ evaluations  │             │
│  │ vector(1024) │  │ masking fns  │  │ nDCG, MRR... │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐                                │
│  │ lake.*       │  │ data_mart.*  │                                │
│  │ (ephemeral)  │  │ governed     │                                │
│  │ Parquet→PG   │  │ CTAS output  │                                │
│  │ staging      │  │ fast BI      │                                │
│  └──────────────┘  └──────────────┘                                │
└────────────────────────────────────────────────────────────────────┘
         ▲                    ▲
         │                    │
    Agent 1 (RAG)        Agent 2 (Pipeline)
    Voyage finance-2     gpt-4o DDL gen
    pgvector search      Parquet→PG→mart
    NEVER reads data     governance enforcement
```

---

## 3. INFRASTRUCTURE

### 3.1 Docker: PostgreSQL 18 + pgvector + pgvectorscale

**File:** `docker/Dockerfile.pg18` (already written, needs testing)

**Critical build details:**
- Base: `postgres:18-bookworm`
- pgvector v0.8.0 (latest stable): `git clone --branch v0.8.0`
- pgvectorscale: requires Rust toolchain + pgrx framework
  - pgrx version must match pgvectorscale expectations — check their `Cargo.toml`
  - Current pin: `cargo-pgrx --version 0.12.9` — **verify this against latest pgvectorscale `Cargo.toml`**
  - Build: `cargo pgrx install --release --pg-config /usr/bin/pg_config`
- Multi-stage build: builder stage has Rust/build tools, runtime stage is clean PG image
- Copy both `.so` and `extension/*` files from builder to runtime

**Known gotchas to handle:**
- pgvectorscale's pgrx version must match exactly — if build fails, check `pgvectorscale/Cargo.toml` for `pgrx = "=0.12.x"` and pin accordingly
- PG 18 may not be officially supported by pgvectorscale yet — check their CI. Fallback: PG 17.
- The `vectorscale*.so` filename may vary — use glob in COPY
- Runtime needs `libssl3` for pgvectorscale

**Testing the build:**
```bash
docker compose build
docker compose up -d
docker exec lab04_pg18 psql -U dba_admin -d metadata_catalog -c "CREATE EXTENSION vector; CREATE EXTENSION vectorscale CASCADE; SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector','vectorscale');"
```

### 3.2 Docker: MinIO (S3-compatible)

**File:** `docker/docker-compose.yml` (already written)

- Image: `minio/minio:latest`
- Ports: 9000 (S3 API), 9001 (web console)
- Credentials: `minio_admin` / `minio_2026!`
- Bucket: `trading-lake` (created by Python script)
- Web console: http://localhost:9001 — useful for demo to show Parquet files

### 3.3 Compose health flow

```yaml
postgres → healthcheck: pg_isready → then Python scripts can connect
minio → healthcheck: mc ready local → then Python can upload Parquet
```

Python scripts should retry connection if containers aren't ready yet.

---

## 4. SQL SCHEMAS

### 4.1 `00_extensions.sql` (written)

Extensions: `vector`, `vectorscale`, `pgcrypto`, `uuid-ossp`
Schemas: `catalog`, `governance`, `rag_monitor`, `data_mart`, `lake`

### 4.2 `01_metadata_catalog.sql` (written)

**Key design decision:** Two enrichment fields on every catalog entry.

| Field | Source | Purpose | Example |
|-------|--------|---------|---------|
| `detail_bi` | Manual (BI team) | Business context, refresh cycles, regulatory notes | "FIX-protocol aligned. Settlement T+2. MiFID II best execution reporting." |
| `detail_agent` | Auto (from data inspection) | Cardinality, distributions, null rates | "515 rows, 12 columns. side: 3 distinct (BUY, SELL, SHRT). 2% null rate on limit_price." |

Both are concatenated into `metadata_text` → embedded → stored as `vector(1024)`.
This produces dramatically richer vectors than pure schema names alone.

**Tables:**
- `catalog.table_metadata` — lake_path (S3 URI), row_count, file_size, partition_keys, detail_bi, detail_agent, classification, contains_pii, embedding
- `catalog.column_metadata` — arrow_type, n_distinct, null_fraction, sample_values, min/max_value, detail_bi, detail_agent, is_pii, masking_rule, embedding
- `catalog.relationship_metadata` — logical joins (inferred from domain, not FK constraints since Parquet has none), join_condition text, embedding
- `catalog.kpi_patterns` — pre-documented analytical patterns with sql_template, domain, required_tables/columns, embedding

**Vector indexes:**
- HNSW: `(m=16, ef_construction=128)` on all embedding columns — sufficient for demo scale
- StreamingDiskANN (pgvectorscale): commented out, ready to enable for production scale

### 4.3 `02_governance.sql` (written)

**Audit trail:** `governance.provisioning_audit` — immutable log: request_id, requester, role, natural language question, RAG results, generated SQL, classification, masking/RLS applied, grants, status, timing.

**Registry:** `governance.data_mart_registry` — lineage tracking for every provisioned mart.

**Roles:**
| Role | Password | Purpose |
|------|----------|---------|
| `bi_analyst` | `bi_2026!` | General BI queries |
| `risk_manager` | `risk_2026!` | Risk monitoring |
| `compliance_officer` | `compl_2026!` | AML/SAR investigation (full PII) |
| `pipeline_agent` | `pipe_2026!` | Agent 2 service account |

**Classification → Access matrix:**
```
public       → bi_analyst, risk_manager, compliance_officer
internal     → bi_analyst, risk_manager, compliance_officer
confidential → risk_manager, compliance_officer
restricted   → compliance_officer ONLY
```

**PII masking functions:**
- `governance.mask_name('Dr. Elena Brunner')` → `'D** E**** B******'`
- `governance.mask_account('CH9300010001')` → `'CH93********'`
- `governance.mask_isin('CH0012032048')` → `'CH00*****048'`

PII masking is applied by Agent 2 in generated SQL **unless** requester_role = 'compliance_officer'.

### 4.4 `03_rag_monitoring.sql` (written)

**Tables:**
- `rag_monitor.search_log` — every search: query_text, query_embedding (stored for offline analysis), results as JSONB, timing breakdown (embed_ms, search_ms, reasoning_ms), requester, model used
- `rag_monitor.relevance_judgments` — ground truth: query → catalog_item → grade (0-3, TREC convention)
- `rag_monitor.evaluation_runs` — batch metrics: P@5, P@10, R@5, R@10, nDCG@5, nDCG@10, MRR, MAP, latency percentiles, per_query_metrics JSONB
- `rag_monitor.feedback_log` — user thumbs up/down on individual search results

**Views:**
- `v_search_quality_trend` — daily aggregation of eval metrics by embedding model
- `v_feedback_summary` — relevance rate per query pattern from user feedback

---

## 5. PYTHON PIPELINE

### 5.1 `config.py` (written)

```python
# Key settings:
DATABASE_URL    # postgresql://dba_admin:dbi2026!@localhost:5433/metadata_catalog
S3_ENDPOINT     # http://localhost:9000
S3_BUCKET       # trading-lake
VOYAGE_API_KEY  # Voyage AI (embeddings)
OPENAI_API_KEY  # OpenAI (LLM reasoning)
EMBEDDING_MODEL # voyage-finance-2 (1024d, finance-optimized)
CHAT_MODEL      # gpt-4o (Agent 2 DDL)
CHAT_MODEL_FAST # gpt-4o-mini (Agent 1 reasoning)
SIMILARITY_THRESHOLD  # 0.40
TOP_K                 # 15
```

### 5.2 `00_generate_parquet.py` — Data Lake Generator (written, needs testing)

**What it does:** Generates 14 DataFrames of realistic Swiss private bank trading data, converts to Parquet via pyarrow, uploads to MinIO bucket `trading-lake`.

**Data generated (seed=42):**
| Table | Rows | Key fields |
|-------|------|-----------|
| exchanges | 8 | SIX, Xetra, Euronext, LSE, NYSE, NASDAQ, TSE, HKEX |
| instruments | 34 | Swiss blue chips, EU/US equities, govt/corp bonds, ETFs, structured products. ISINs, SEDOLs, Bloomberg tickers. |
| counterparties | 6 | Goldman Sachs, JPM, UBS (brokers), SIX SIS, Clearstream (custodians), Eurex (clearing) |
| clients | 25 | Swiss-centric mix. Segments: retail→UHNWI→institutional. 2 PEPs (Whitfield UK, Volkov RU). Risk ratings: low→critical. |
| accounts | ~63 | UHNWI/institutional get 3 accounts, HNWI/corporate get 2, retail gets 1 |
| orders | ~515 | FIX lifecycle statuses. Volkov (client 19) gets sanctions flags. |
| executions | ~668 | Multiple fills for large orders. Commissions in bps. |
| positions | ~358 | EOD snapshot with P&L. 3-12 positions per non-cash account. |
| market_prices | ~1,000 | 30 trading days OHLCV for all instruments. |
| cash_balances | ~63 | Balance, available, reserved per account. |
| risk_limits | ~75 | 2-4 limit types per client. Some above 80% utilization. |
| risk_metrics | 125 | 5 days × 25 clients. VaR, CVaR, Sharpe, volatility, drawdown. |
| compliance_checks | ~1,040 | 1-3 checks per order. Sanctions flags for Volkov. |
| aml_alerts | ~15 | High-risk clients: Volkov, Al Rashid, Whitfield, Oliveira, Lefèvre. |

**S3 upload pattern:**
```python
s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT, ...)
df.to_parquet(buf, engine="pyarrow", index=False)
s3.put_object(Bucket="trading-lake", Key="orders.parquet", Body=buf.getvalue())
```

### 5.3 `10_scan_and_embed.py` — Metadata Scanner + Embedder (written, needs testing)

**What it does:**
1. Lists Parquet files in MinIO bucket
2. For each file: reads Parquet schema (pyarrow) + sample data (pandas) WITHOUT loading full file
3. Computes column statistics: n_distinct, null_fraction, sample_values, min/max
4. Attaches `detail_bi` from hardcoded BI annotations (14 table-level, ~15 column-level)
5. Generates `detail_agent` from computed stats
6. Builds `metadata_text` = concatenation of schema info + detail_bi + detail_agent
7. Embeds all texts with Voyage finance-2 (`input_type="document"`)
8. Stores everything in `catalog.*` tables

**BI team annotations (hardcoded for demo, would be a UI in production):**
```python
TABLE_DETAIL_BI = {
    "orders": "FIX-protocol aligned order book. cl_ord_id is UUID. Status follows FIX lifecycle. compliance_check is pre-trade result. Settlement T+2 equities.",
    "aml_alerts": "Generated by Actimize/NICE. severity 'critical' = immediate MLRO escalation. RESTRICTED. FINMA AMLA compliance.",
    ...
}
COLUMN_DETAIL_BI = {
    ("clients", "is_pep"): "PEP flag from WorldCheck. Enhanced Due Diligence required.",
    ("risk_metrics", "cvar_1d_95"): "Conditional VaR (Expected Shortfall) at 95%. More conservative than VaR for tail risk.",
    ...
}
```

**Classification rules:**
```python
RESTRICTED_TABLES   = {"aml_alerts", "compliance_checks"}
CONFIDENTIAL_TABLES = {"risk_metrics", "risk_limits", "orders", "executions"}
INTERNAL_TABLES     = {"positions", "cash_balances", "market_prices", "clients", "accounts"}
PUBLIC_TABLES       = {"exchanges", "instruments", "counterparties"}
```

**PII columns:**
```python
PII_COLUMNS = {
    ("clients", "legal_name"):       "partial_mask",
    ("clients", "short_name"):       "partial_mask",
    ("accounts", "account_number"):  "partial_mask",
}
```

**Relationships (logical, not FK — Parquet has no constraints):**
15 relationships: orders→accounts, orders→instruments, executions→orders, accounts→clients, positions→accounts, positions→instruments, etc.

**Embedding batch size:** 50 texts per API call to Voyage.

### 5.4 `20_agent_rag_search.py` — Agent 1: RAG Search (written, needs testing)

**Input:** Natural language question + requester identity
**Output:** `SearchRecommendation` dataclass

**Flow:**
1. `embed_query(question)` — Voyage finance-2 with `input_type="query"` (asymmetric search)
2. `vector_search(conn, embedding)` — 4 parallel queries against catalog tables using `<=>` cosine distance
3. Extract governance: max_classification from results, PII fields
4. `llm_reason(question, results)` — gpt-4o-mini explains which tables/columns are needed
5. Log to `rag_monitor.search_log` with full timing breakdown

**Critical: Agent 1 NEVER sees raw data values.** It only sees table names, column names, types, statistics, and BI annotations. This is the trust boundary that prevents sensitive data leaks.

**Search queries (all use `ORDER BY embedding <=> %s::vector`):**
- Tables: `LIMIT TOP_K`
- Columns: `LIMIT TOP_K * 2` (more granular)
- Relationships: `LIMIT 10` (lower threshold: `SIMILARITY_THRESHOLD * 0.8`)
- KPI patterns: `LIMIT 5`

### 5.5 `30_agent_pipeline.py` — Agent 2: Pipeline (written, needs testing)

**Input:** SearchRecommendation from Agent 1 + requester identity
**Output:** `ProvisioningResult` dataclass

**Flow:**
1. `generate_mart_sql()` — LLM (gpt-4o) generates `CREATE TABLE data_mart.dm_xxx AS SELECT ...` referencing `lake.*` tables. Prompt includes PII masking instructions if needed.
2. `load_parquet_to_staging()` — For each required table: download Parquet from S3 → `pandas.read_parquet()` → create `lake.<table>` in PG → `COPY` data in via `copy_expert()`. **These are ephemeral staging tables.**
3. Execute the generated DDL → `data_mart.dm_xxx` is created with data.
4. `apply_governance()` — GRANT SELECT to allowed roles, REVOKE from others.
5. `cleanup_staging()` — DROP all `lake.*` tables. No raw data persists in PG.
6. Log to `governance.provisioning_audit` + register in `governance.data_mart_registry`.

**Staging pattern (Parquet → PG → DROP):**
```python
# Read Parquet from S3
obj = s3.get_object(Bucket=bucket, Key="orders.parquet")
df = pd.read_parquet(io.BytesIO(obj["Body"].read()))

# Create staging table (type-mapped from pandas dtypes)
cur.execute("CREATE TABLE lake.orders (order_id BIGINT, ...)")

# COPY data via StringIO for performance
buf = StringIO()
df.to_csv(buf, index=False, header=False, sep='\t', na_rep='\\N')
buf.seek(0)
cur.copy_expert("COPY lake.orders (...) FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '\\N')", buf)

# After CTAS into data_mart, drop staging:
cur.execute("DROP TABLE IF EXISTS lake.orders CASCADE")
```

**LLM prompt for DDL generation includes:**
- Available tables and columns from recommendation
- Join paths
- PII masking instructions (with specific function names)
- KPI template hint from matching patterns
- Strict instruction: output ONLY SQL, no markdown

### 5.6 `40_demo.py` — Demo Orchestrator (written, needs testing)

3 scenarios:

| # | Title | Requester | Role | Expected Classification | PII Masked? |
|---|-------|-----------|------|------------------------|-------------|
| 1 | Portfolio Exposure | maria.chen | bi_analyst | 🔵 internal | Yes |
| 2 | Risk Limit Monitoring | thomas.weber | risk_manager | 🟠 confidential | Yes |
| 3 | AML Investigation | sophie.martin | compliance_officer | 🔴 restricted | **No** (compliance privilege) |

For each scenario: Agent 1 → print recommendation → Agent 2 → print result → sample data → then show governance summary (audit trail + registry).

### 5.7 `50_evaluate_rag.py` — RAG Quality Evaluation (written, needs testing)

**Golden set:** `benchmarks/golden_queries.json` — 12 queries with expected tables, columns, KPI pattern name.

**Metrics computed:**
| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| Precision@K | relevant_retrieved / K | Are the top results relevant? |
| Recall@K | relevant_retrieved / total_relevant | Did we find everything? |
| nDCG@K | DCG / ideal_DCG | Are relevant results ranked higher? |
| MRR | 1/rank_of_first_relevant | How fast do we find something useful? |
| MAP | mean of per-query average precision | Overall retrieval quality |
| KPI match rate | correct_kpi / total_kpi_queries | Does it find the right pattern? |
| Latency | p50, p95, p99 | Performance envelope |

**Results stored in:**
- `rag_monitor.relevance_judgments` — per-query ground truth
- `rag_monitor.evaluation_runs` — batch metrics with per_query_metrics JSONB

---

## 6. THINGS TO VERIFY / FIX IN CLAUDE CODE

### 6.1 Docker build
- [ ] `Dockerfile.pg18`: verify pgrx version compatibility with pgvectorscale
- [ ] Check if `postgres:18-bookworm` exists, fall back to `postgres:17-bookworm` if needed
- [ ] Test that both extensions create successfully
- [ ] Verify SQL init scripts run in order (00→01→02→03)

### 6.2 Parquet generator
- [ ] Test `00_generate_parquet.py` against live MinIO
- [ ] Verify all 14 Parquet files upload correctly
- [ ] Check Parquet metadata is readable by pyarrow without downloading full file (currently downloads full — optimize with `pq.ParquetFile` range reads if needed)
- [ ] Ensure seed=42 produces consistent output

### 6.3 Metadata scanner
- [ ] Test `10_scan_and_embed.py` end-to-end
- [ ] Verify column statistics extraction from sample data
- [ ] Verify embedding dimensions match `vector(1024)` in catalog tables
- [ ] Check that metadata_text includes both detail_bi AND detail_agent
- [ ] Verify all 15 relationships are inserted
- [ ] Verify all 10 KPI patterns are inserted

### 6.4 Agent 1 (RAG Search)
- [ ] Test with each of the 3 demo scenarios
- [ ] Verify search logging to `rag_monitor.search_log`
- [ ] Verify `input_type="query"` is used for search (vs `"document"` for catalog)
- [ ] Test with queries that should return NO results (edge case)
- [ ] Check similarity threshold tuning — 0.40 may be too low or too high

### 6.5 Agent 2 (Pipeline)
- [ ] Test full staging flow: Parquet → lake.* → data_mart.dm_* → DROP lake.*
- [ ] Verify PII masking in generated SQL for bi_analyst/risk_manager
- [ ] Verify NO masking for compliance_officer
- [ ] Test GRANT/REVOKE application
- [ ] Verify audit trail completeness
- [ ] Handle LLM returning markdown-wrapped SQL (strip ``` fences)
- [ ] Handle LLM generating invalid SQL (catch exception, log error, rollback)
- [ ] Test dry-run mode

### 6.6 Evaluation
- [ ] Run `50_evaluate_rag.py` against all 12 golden queries
- [ ] Verify metrics are stored in `rag_monitor.evaluation_runs`
- [ ] Check relevance judgments are inserted/updated correctly

### 6.7 End-to-end integration test
```bash
docker compose up -d
# wait for healthy
python python/00_generate_parquet.py
python python/10_scan_and_embed.py
python python/40_demo.py --scenario 1
python python/50_evaluate_rag.py --verbose
# Verify governance:
psql -h localhost -p 5433 -U dba_admin -d metadata_catalog \
  -c "SELECT target_object, classification, masking_applied, grants_applied FROM governance.provisioning_audit"
```

---

## 7. DATA MODEL SUMMARY

### 7.1 What's in S3 (Parquet)
All raw data. 14 tables. No governance, no indexes. Cheap storage.

### 7.2 What's in PostgreSQL

**NO raw data content.** Only:

| Schema | Contains | Purpose |
|--------|----------|---------|
| `catalog` | Table/column metadata, relationships, KPI patterns + vector(1024) embeddings | Agent 1 RAG search |
| `governance` | Audit trail, mart registry, masking functions, roles | Regulatory compliance |
| `rag_monitor` | Search logs, relevance judgments, evaluation runs, feedback | Quality improvement |
| `lake` | **Ephemeral** staging tables (created during provisioning, dropped after) | Bridge: S3→PG |
| `data_mart` | Provisioned analytical tables (`dm_xxx`) | BI queries (the output) |

### 7.3 Trust boundary

```
Agent 1 reads:  catalog.*     (metadata only — no data values)
Agent 2 reads:  S3 Parquet    (raw data — for staging)
Agent 2 writes: lake.*        (ephemeral staging)
Agent 2 writes: data_mart.*   (governed output)
Agent 2 writes: governance.*  (audit trail)
BI users read:  data_mart.*   (governed, masked, granted)
```

---

## 8. EMBEDDING STRATEGY

### 8.1 Model: Voyage finance-2

| Property | Value |
|----------|-------|
| Dimensions | 1024 |
| Context | 32K tokens |
| Domain | Finance-optimized (+7% vs OpenAI on financial benchmarks) |
| Asymmetric | `input_type="document"` for catalog, `input_type="query"` for questions |
| Free tier | 50M tokens |
| Provider | Voyage AI (Anthropic recommended) |

### 8.2 What gets embedded

| Catalog table | metadata_text formula | Example |
|---------------|----------------------|---------|
| table_metadata | `"Table: {name} \| Columns: {n} \| Rows: ~{count} \| Business context: {detail_bi}"` | "Table: orders \| Columns: 18 \| Rows: ~515 \| Business context: FIX-protocol aligned order book..." |
| column_metadata | `"Column: {table}.{col} \| Type: {type} \| Business context: {detail_bi} \| Distinct: {n} \| Samples: {vals}"` | "Column: risk_metrics.cvar_1d_95 \| Type: double \| Business context: Conditional VaR (Expected Shortfall)..." |
| relationship_metadata | `"Relationship: {src}.{col} → {tgt}.{col} ({type})"` | "Relationship: orders.account_id → accounts.account_id (one_to_many)" |
| kpi_patterns | `"{name}: {description}"` | "Execution Quality (TCA): Transaction Cost Analysis — fill price vs arrival price..." |

### 8.3 Vector indexes

**HNSW (pgvector):** `(m=16, ef_construction=128)` on all 4 catalog tables. Good for demo scale (~200 vectors total).

**StreamingDiskANN (pgvectorscale):** Commented out. Enable for 100K+ vectors:
```sql
CREATE INDEX idx_column_meta_diskann ON catalog.column_metadata
    USING diskann (embedding vector_cosine_ops);
```

---

## 9. GOVERNANCE ENFORCEMENT MATRIX

| Scenario | Requester | Classification | PII Masking | Grants | RLS |
|----------|-----------|---------------|-------------|--------|-----|
| Exposure dashboard | bi_analyst | internal | `mask_name`, `mask_account` | all 3 roles | No |
| Risk limit monitoring | risk_manager | confidential | `mask_name`, `mask_account` | risk+compliance | Yes (pattern) |
| AML investigation | compliance_officer | restricted | **NONE** (full PII) | compliance only | Yes (pattern) |

**How masking works in generated SQL:**
Agent 2's LLM prompt includes masking instructions. The generated SQL looks like:
```sql
CREATE TABLE data_mart.dm_exposure AS
SELECT
    governance.mask_name(c.legal_name) AS client_name,  -- masked
    c.segment,
    i.sector,
    SUM(p.market_value) AS total_exposure
FROM lake.positions p
JOIN lake.accounts a ON p.account_id = a.account_id
JOIN lake.clients c ON a.client_id = c.client_id
JOIN lake.instruments i ON p.instrument_id = i.instrument_id
GROUP BY 1, 2, 3;
```

For compliance_officer, the same query has `c.legal_name AS client_name` (no masking).

---

## 10. MONITORING DASHBOARD QUERIES

After running demos and evaluations, these queries show the monitoring data:

```sql
-- RAG search quality over time
SELECT * FROM rag_monitor.v_search_quality_trend;

-- Latest evaluation run
SELECT run_name, ndcg_at_5, mrr, map, avg_latency_ms
FROM rag_monitor.evaluation_runs ORDER BY run_date DESC LIMIT 1;

-- Worst-performing queries (lowest recall)
SELECT query, r5 AS recall, ndcg5, latency
FROM rag_monitor.evaluation_runs e,
     jsonb_to_recordset(e.per_query_metrics) AS x(query text, r5 float, ndcg5 float, latency int)
WHERE run_id = (SELECT MAX(run_id) FROM rag_monitor.evaluation_runs)
ORDER BY r5 ASC LIMIT 5;

-- User feedback relevance rate
SELECT * FROM rag_monitor.v_feedback_summary;

-- Provisioning audit (who created what, with what governance)
SELECT target_object, requested_by, classification,
       masking_applied, grants_applied, row_count, execution_time_ms
FROM governance.provisioning_audit
WHERE status = 'executed'
ORDER BY requested_at DESC;

-- Data mart registry
SELECT mart_name, classification, allowed_roles, row_count
FROM governance.data_mart_registry;
```

---

## 11. FILE INVENTORY

| File | Status | Notes |
|------|--------|-------|
| `docker/Dockerfile.pg18` | Written | Test build, verify pgrx version |
| `docker/docker-compose.yml` | Written | PG + MinIO |
| `sql/00_extensions.sql` | Written | vector, vectorscale, pgcrypto |
| `sql/01_metadata_catalog.sql` | Written | 4 tables + HNSW indexes |
| `sql/02_governance.sql` | Written | Audit, registry, roles, masking |
| `sql/03_rag_monitoring.sql` | Written | Search log, judgments, evals |
| `python/config.py` | Written | Config vars |
| `python/00_generate_parquet.py` | Written | Parquet → MinIO |
| `python/10_scan_and_embed.py` | Written | Schema scan → Voyage → pgvector |
| `python/20_agent_rag_search.py` | Written | Agent 1: RAG search |
| `python/30_agent_pipeline.py` | Written | Agent 2: Parquet→PG→mart |
| `python/40_demo.py` | Written | 3-scenario demo |
| `python/50_evaluate_rag.py` | Written | Golden set evaluation |
| `benchmarks/golden_queries.json` | Written | 12 benchmark queries |
| `requirements.txt` | Written | psycopg2, openai, voyageai, pandas, pyarrow, boto3 |
| `README.md` | Written | Full docs |

**Everything is written but UNTESTED.** Claude Code task: review, fix bugs, test end-to-end, handle edge cases.

---

## 12. PRESENTATION INTEGRATION

This lab directly maps to the conference talk architecture:

```
Slide 29: "Before" — manual, slow, no governance
Slide 30: "After"  — this lab

Key demo moments:
1. Show MinIO console with Parquet files (storage layer)
2. Agent 1 searches metadata — show similarity scores, detail_bi context
3. Agent 2 provisions mart — show staging, DDL generation, masking
4. Query the mart — instant JOINs vs 10min lake query
5. Show audit trail — who requested what, what governance was applied
6. Show monitoring — precision, recall, nDCG of the RAG search
```

Blog series connections:
- "RAG, MCP, Skills — Governance" → RAG as gatekeeper trust boundary
- "Embedding Versioning" → Voyage finance-2 vs text-embedding-3-small comparison
- "Hybrid Search" → HNSW cosine on finance-optimized vectors
- "Event-driven architecture" → in production, catalog refreshes via CDC
