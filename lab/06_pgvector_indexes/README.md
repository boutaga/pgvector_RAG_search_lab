# Lab 06 — pgvector Index Tuning Lab

Hands-on lab covering every pgvector index type, tuning parameter, and DBA operation on 25,000 Wikipedia articles with 3072-dimension embeddings. Companion material for the blog post *"pgvector, a guide for DBA - Part 2: Indexes"*.

## Prerequisites

- Docker and Docker Compose
- Python 3.11+
- OpenAI API key (for embedding generation)
- ~2 GB disk space for embeddings

## Quick Start

### 1. Build and start PostgreSQL 18

```bash
cd DEV/docker
docker compose build    # ~15 min first time (compiles pgvector + pgvectorscale)
docker compose up -d
```

Verify:
```bash
docker exec lab06_pg18 psql -U dba_admin -d wikipedia -c \
  "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector','vectorscale');"
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key

Edit `DEV/.env` and set your `OPENAI_API_KEY`.

### 4. Load data and generate embeddings

```bash
cd python
python 01_load_wikipedia.py           # Load 25,000 articles
python 02_create_embeddings_dense.py  # OpenAI text-embedding-3-large (3072d)
python 03_create_embeddings_sparse.py # SPLADE sparse embeddings
```

### 5. Run the SQL lab scripts

Connect to the database and run scripts in order:

```bash
psql -h localhost -p 5435 -U dba_admin -d wikipedia
```

```sql
\i sql/02_hnsw_indexes.sql
\i sql/03_ivfflat_indexes.sql
\i sql/04_diskann_indexes.sql
\i sql/05_iterative_scans.sql
\i sql/06_halfvec_quantization.sql
\i sql/07_operators_and_filtering.sql
\i sql/08_monitoring_maintenance.sql
```

## What's Inside

### SQL Scripts

| Script | Topic |
|--------|-------|
| `02_hnsw_indexes.sql` | HNSW: 2000-dim limit, halfvec workaround, ef_search tuning, parallel build |
| `03_ivfflat_indexes.sql` | IVFFlat: no dim limit, probes sweep, optimizer behavior, SET LOCAL |
| `04_diskann_indexes.sql` | DiskANN: native 3072d, CONCURRENTLY, query tuning |
| `05_iterative_scans.sql` | Iterative scans: filtered vector search done right |
| `06_halfvec_quantization.sql` | halfvec, binary_quantize, re-ranking, subvector |
| `07_operators_and_filtering.sql` | Operators, sargability, partial indexes |
| `08_monitoring_maintenance.sql` | DBA monitoring, GUC audit, maintenance |

### Key Takeaways

- **HNSW** has a 2000-dimension limit for `vector` but 4000 for `halfvec`
- **IVFFlat** has no dimension limit — works with `vector(3072)` natively
- **DiskANN** supports up to 16,384 dimensions and `CREATE INDEX CONCURRENTLY`
- **Iterative scans** solve the vector + WHERE truncation problem
- **Binary quantize + re-ranking** gives binary speed with full-precision quality
- Wrong operator = index not used. Match your operator to your operator class.

## Environment

- PostgreSQL 18
- pgvector (main branch, includes 0.8.2 EXPLAIN fix)
- pgvectorscale 0.9.0
- text-embedding-3-large (3072 dimensions)
- 25,000 Wikipedia articles

## Connection Details

| Setting | Value |
|---------|-------|
| Host | localhost |
| Port | 5435 |
| Database | wikipedia |
| User | dba_admin |
| Password | dbi2026! |
