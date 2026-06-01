# Lab 07 — Security Observability Demo (Swiss PGDay 2026)

Companion lab for the talk **"The cost of security debt in PostgreSQL when implementing
AI workflows."** It makes the talk's thesis visible: apply a security control, measure the
delta in retrieval quality (Recall / Precision / nDCG), make an informed decision.

The arc is **leak → fix → measure**:

1. **Leak** — a Bank A analyst's RAG query returns Bank B content, because RLS protects the
   `documents` table but not the `embeddings` table (which holds the chunk text). pgaudit
   logs it as authorized access.
2. **Fix** — one RLS policy on `embeddings` closes the leak; anonymizing client names then
   degrades answer quality.
3. **Measure** — a 3-state dashboard turns the cost of each control into numbers.

## Stack

PostgreSQL 18 + `pgvector` 0.8 + `pgaudit` + `postgresql_anonymizer` (anon 3.1), in one
container (`lab07_pg18`, port **5436**, database `secobs`). Embeddings use OpenAI
`text-embedding-3-small` (1536-dim).

## Prerequisites

- Docker + Docker Compose
- Python 3.11+
- An OpenAI API key in `.env` (`OPENAI_API_KEY=...`)

## Setup

```bash
cd lab/07_security_observability

# 1. Build + start the database (extensions, schema, pgaudit, RLS-on-documents)
docker compose -f docker/docker-compose.yml up -d --build --wait

# 2. Python env
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# 3. Put your OpenAI key in .env  (OPENAI_API_KEY=sk-...)

# 4. Seed the corpus (3 banks, 60 docs) + build the labelled query set
./.venv/bin/python data/seed_documents.py

# 5. Generate embeddings (baseline, then the masked re-embedding)
./.venv/bin/python python/embed.py --mode baseline
./.venv/bin/python python/embed.py --mode masked
```

## The demo

```bash
# ACT 1 — the leak (embeddings has no RLS yet)
./.venv/bin/python python/ask.py --tenant bank_a --query "Q3 trading positions across the portfolio"
#   -> answer cites other banks; retrieved rows flagged LEAK

# ACT 2 — the fix
docker exec -i lab07_pg18 psql -U dba_admin -d secobs < sql/demo/10_rls_embeddings.sql
./.venv/bin/python python/ask.py --tenant bank_a --query "Q3 trading positions across the portfolio"
#   -> no leak

# anonymize client names (stage beat), then the answer quality drops
docker exec -i lab07_pg18 psql -U dba_admin -d secobs < sql/demo/20_anonymize.sql
./.venv/bin/python python/ask.py --tenant bank_a --query "What are Helvetia Industrials AG's positions and exposures?" --masked

# ACT 3 — measure all three states and refresh the dashboard data
./.venv/bin/python python/measure_security_cost.py
./.venv/bin/streamlit run python/dashboard.py     # http://localhost:8501
```

`measure_security_cost.py` toggles the demo SQL itself (rollback → baseline, apply RLS,
search masked column) and writes `mock/frozen_run.json`, which the dashboard reads. The
dashboard works offline from that file, so it is also the on-stage fallback.

```bash
# Reset to the baseline (leaking, unmasked) state for a fresh rehearsal
docker exec -i lab07_pg18 psql -U dba_admin -d secobs < sql/demo/99_rollback.sql
```

## pgaudit

Session and object auditing are on (`shared_preload_libraries=pgaudit`, `pgaudit.log` set
in `sql/init/02_pgaudit.sql`). Watch the leak get logged as an authorized read:

```bash
docker logs -f lab07_pg18 2>&1 | grep AUDIT
```

## Files

```
docker/   Dockerfile.pg18, docker-compose.yml   (PG18 + vector + pgaudit + anon)
sql/init/ 00_extensions, 01_schema, 02_pgaudit  (run at container start = baseline)
sql/demo/ 10_rls_embeddings, 20_anonymize, 99_rollback  (applied live on stage)
data/     seed_documents.py, test_cases.json    (corpus + labelled queries)
python/   embed.py, ask.py, measure_security_cost.py, dashboard.py
mock/     frozen_run.json                        (offline dashboard source)
```

## Notes

- This is a stage demo, not production. Local lab only; no TDE step (core PostgreSQL has no
  native TDE — it is a slide-level concept in the talk).
- The masked re-embedding uses `anon.pseudo_company()` (deterministic via `anon.salt`) so
  runs are reproducible.
