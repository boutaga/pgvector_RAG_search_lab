# Lab 04: Metadata-Driven RAG with Governed Data Mart Provisioning

**Conference demo for: "pgvector: Why is PostgreSQL at the center of AI workflows?"**

## The Problem

Data lakes are great for storing massive amounts of data cheaply (Parquet on S3). But they're awful for exploratory queries — no indexes, no JOINs, no governance. BI analysts wait 10+ minutes for queries, BI connectors time out, and every new KPI requires a data engineering ticket.

## The Solution: Right Tool for the Right Job

```
┌─────────────────────────────────────────────────────────────────────┐
│  AGENT 3: Broker Intelligence (Senior Portfolio Manager)            │
│  ────────────────────────────────────────────────                   │
│  • Scans: financial news, filings, analyst recs, earnings reports  │
│  • Cross-references signals with client position metadata          │
│  • Detects: overexposure, sector risk, analyst downgrades          │
│  • Presents alerts to human (human-in-the-loop)                   │
│  • On approval → triggers Agent 1 + Agent 2 pipeline              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ approved alert + question
                               ▼
┌───────────────────────────────────────────────────────────────────┐
│  AGENT 1: RAG Search                                              │
│  ────────────────────                                             │
│  • Embeds question (Voyage finance-2, input_type="query")         │
│  • Searches pgvector metadata catalog (1024d DiskANN)             │
│  • Returns: tables, columns, joins, KPI patterns                  │
│  • NEVER touches raw data — only metadata + embeddings            │
│  • Logs search to rag_monitor for quality tracking                │
└──────────────────────────────┬────────────────────────────────────┘
                               │ recommendation
                               ▼
┌───────────────────────────────────────────────────────────────────┐
│  AGENT 2: Pipeline                                                │
│  ────────────────────                                             │
│  1. Reads Parquet from S3 (MinIO) → stages into lake.* in PG     │
│  2. Generates CREATE TABLE AS SELECT (LLM)                        │
│  3. Applies PII masking (governance.mask_name/mask_account)       │
│  4. Creates data_mart.dm_xxx with governed access                 │
│  5. Drops staging tables (ephemeral)                              │
│  6. Logs audit trail + registers mart                             │
└──────────────────────────────┬────────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────────┐
│  PostgreSQL 18 — Governed Data Mart                               │
│  ──────────────────────────────────                               │
│  • Fast JOINs, indexes, aggregations                              │
│  • Role-based access (classification → grants)                    │
│  • PII masking enforced at SQL level                              │
│  • Full audit trail for regulatory compliance                     │
│  • BI analyst queries in seconds, not minutes                     │
└───────────────────────────────────────────────────────────────────┘
```

**The key insight:** PostgreSQL at the center — it handles metadata catalog (pgvector), governance framework, RAG quality monitoring, AND the final data mart. One database, multiple workloads.

## Infrastructure

| Component | Role | Image |
|-----------|------|-------|
| **PostgreSQL 18** | Metadata catalog + governance + data marts | Custom (Dockerfile.pg18) |
| **pgvector 0.8.1** | Vector data type and operators | Built from source |
| **pgvectorscale 0.9.0** | StreamingDiskANN indexes for production scale | Built from source (Timescale) |
| **MinIO** | S3-compatible object storage (data lake) | `minio/minio:latest` |

## Data Lake (Parquet on S3)

The data lake simulates a Swiss private bank trading system:

| File | Rows | Description |
|------|------|-------------|
| `exchanges.parquet` | 8 | ISO 10383 MIC codes |
| `instruments.parquet` | 34 | Swiss blue chips, EU/US equities, bonds, ETFs |
| `counterparties.parquet` | 6 | Brokers, custodians, clearing houses |
| `clients.parquet` | 25 | Swiss/EU/intl mix incl. PEPs |
| `accounts.parquet` | 63 | Trading, custody, cash |
| `orders.parquet` | ~515 | FIX-protocol order book |
| `executions.parquet` | ~668 | Fills with TCA data |
| `positions.parquet` | ~358 | EOD snapshot with P&L |
| `market_prices.parquet` | ~1,000 | 30-day OHLCV |
| `risk_limits.parquet` | ~75 | Exposure/concentration limits |
| `risk_metrics.parquet` | 125 | VaR, CVaR, Sharpe per client |
| `compliance_checks.parquet` | ~1,040 | Pre-trade checks |
| `aml_alerts.parquet` | 15 | AML investigation pipeline |
| `financial_news.parquet` | ~200 | Bloomberg/Reuters/AWP financial news |
| `public_filings.parquet` | ~80 | SIX/FINMA regulatory filings |
| `analyst_recommendations.parquet` | ~150 | Sell-side analyst research |
| `financial_reports.parquet` | ~60 | Quarterly/annual earnings reports |

**No raw data in PostgreSQL.** PG stores only metadata about structure and meaning.

## Metadata Catalog (pgvector)

Each catalog entry has two enrichment fields that dramatically improve retrieval quality:

- **`detail_bi`** — Manual annotation from BI team: business context, refresh cycles, regulatory requirements, caveats
- **`detail_agent`** — Auto-generated from data inspection: cardinality, distributions, common values, null rates

These fields are concatenated into `metadata_text` before embedding, producing vectors that capture both schema structure AND business semantics.

## Governance Framework

### Classification → Access Matrix

| Classification | bi_analyst | risk_manager | compliance_officer | portfolio_manager |
|---------------|:---:|:---:|:---:|:---:|
| 🟢 public | ✓ | ✓ | ✓ | ✓ |
| 🔵 internal | ✓ | ✓ | ✓ | ✓ |
| 🟠 confidential | ✗ | ✓ | ✓ | ✓ |
| 🔴 restricted | ✗ | ✗ | ✓ | ✗ |

### PII Masking

Applied automatically unless requester is `compliance_officer`:

| Column | Function | Example |
|--------|----------|---------|
| `clients.legal_name` | `governance.mask_name()` | "Dr. Elena Brunner" → "D** E**** B******" |
| `accounts.account_number` | `governance.mask_account()` | "CH9300010001" → "CH93********" |

## RAG Quality Monitoring

Built-in evaluation framework with:

- **Search logging** — every query with results, latencies, classifications
- **Golden benchmarks** — 16 annotated queries with expected results
- **IR metrics** — Precision@K, Recall@K, nDCG@K, MRR, MAP
- **Latency tracking** — embedding, search, reasoning breakdown
- **Feedback loop** — user relevance feedback for continuous improvement

Run evaluation:
```bash
python python/50_evaluate_rag.py --verbose
```

Query monitoring:
```sql
SELECT * FROM rag_monitor.v_search_quality_trend;
SELECT * FROM rag_monitor.v_feedback_summary;
```

## Quick Start

```bash
# 1. Build and start infrastructure
cd lab/04_metadata_rag_mart/docker
docker compose build          # builds PG 18 + pgvector + pgvectorscale
docker compose up -d
# Wait for healthy:
docker exec lab04_pg18 pg_isready -U dba_admin -d metadata_catalog

# 2. Install Python deps
pip install -r requirements.txt

# 3. Set API keys
export VOYAGE_API_KEY="pa-..."    # Voyage AI (embeddings)
export OPENAI_API_KEY="sk-..."    # OpenAI (LLM reasoning)

# 4. Generate data lake (Parquet → MinIO)
python python/00_generate_parquet.py

# 5. Scan schemas + embed metadata
python python/10_scan_and_embed.py

# 6. Run the demo (3 scenarios)
python python/40_demo.py

# 7. Run broker intelligence demo
python python/45_demo_broker.py --approve-all

# 8. Evaluate RAG quality
python python/50_evaluate_rag.py --verbose

# Optional: single scenario or dry-run
python python/40_demo.py --scenario 1
python python/40_demo.py --dry-run
python python/45_demo_broker.py --approve 1 --dry-run
```

## Demo Scenarios

### Agent 1 + Agent 2 Pipeline (40_demo.py)

**S1: Portfolio Exposure Dashboard (BI Analyst)**
→ 🔵 internal | PII masked | All roles access

**S2: Risk Limit Monitoring (Risk Manager)**
→ 🟠 confidential | PII masked | risk_manager + compliance only

**S3: AML Investigation View (Compliance Officer)**
→ 🔴 restricted | Full PII (compliance privilege) | compliance only

### Agent 3 Broker Intelligence (45_demo_broker.py)

**S-Broker-1: Nestlé Revenue Miss + Consumer Staples Overexposure**
→ Signal: Nestlé Q4 revenue misses by 2.3% → 🔴 critical alert
→ Action: Provision dm_nestle_exposure_impact with sector + client breakdown

**S-Broker-2: FINMA Regulatory Action on UBS + Financials Concentration**
→ Signal: FINMA orders UBS to increase capital buffer → 🔴 critical alert
→ Action: Provision dm_financials_sector_risk with concentration analysis

**S-Broker-3: Multiple Analyst Downgrades in Tech Sector**
→ Signal: Goldman downgrades NVIDIA, JP Morgan cuts ASML → 🟠 warning alerts
→ Action: Provision dm_tech_downgrade_impact with P&L impact

## Phase 10: Multi-Model Comparison + Claude LLM + Streamlit UI

Lab 04 now supports multiple embedding providers and LLM models behind unified abstractions, making it an interactive model comparison tool for conference demos.

### Supported Models

| Embedding Model | Provider | Notes |
|-----------------|----------|-------|
| `voyage-finance-2` | Voyage AI | Finance-optimized, +7% nDCG@10 (default) |
| `text-embedding-3-small` | OpenAI | Cost-effective, dimension reduction to 1024 |
| `text-embedding-3-large` | OpenAI | Highest quality, dimension reduction to 1024 |
| `mxbai-embed-large` | Ollama | Self-hosted, no API key needed |

| LLM Model | Provider | Tier |
|-----------|----------|------|
| `gpt-5.2` | OpenAI | Flagship (default) |
| `gpt-5-mini` | OpenAI | Fast |
| `claude-opus-4-6` | Anthropic | Flagship |
| `claude-sonnet-4-6` | Anthropic | Balanced |
| `claude-haiku-4-5` | Anthropic | Fast |

### Streamlit Dashboard

```bash
streamlit run python/60_streamlit_comparison.py
```

5 tabs: Single Query, Benchmark Comparison, Embedding Explorer, Broker Intelligence, Evaluation History.

### Multi-Model Evaluation

```bash
# Single model
python python/50_evaluate_rag.py --embedding-model text-embedding-3-small --llm-model claude-sonnet-4-6

# Compare all embedding models side-by-side
python python/50_evaluate_rag.py --compare-all --verbose
```

### Ollama Setup (self-hosted)

Ollama is included in Docker Compose for API-free embedding comparison:

```bash
docker compose up -d                            # starts PG + MinIO + Ollama
docker exec lab04_ollama ollama pull mxbai-embed-large   # download model (~670MB)
```

### Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  embedding_provider.py          llm_provider.py                      │
│  ┌─────────┐ ┌────────┐       ┌────────┐ ┌───────────┐             │
│  │ Voyage  │ │ OpenAI │       │ OpenAI │ │ Anthropic │             │
│  │ finance │ │ embed  │       │ GPT-5  │ │ Claude    │             │
│  └────┬────┘ └───┬────┘       └───┬────┘ └─────┬─────┘             │
│       │          │                │             │                    │
│  ┌────┴──┐  ┌────┴──┐       ┌────┴──┐    ┌────┴──┐                │
│  │Ollama │  │ Fake  │       │ Fake  │    │       │                │
│  │mxbai  │  │(test) │       │(test) │    │       │                │
│  └───────┘  └───────┘       └───────┘    └───────┘                │
│                                                                     │
│  ───────── unified interface ──────────────────────────────────     │
│       embed_texts() / embed_query()    chat(system, user)          │
└──────────────────────────────────────────────────────────────────────┘
```

## Technical Notes

- **Embedding:** Configurable via `DEFAULT_EMBEDDING_MODEL` env var (default: `voyage-finance-2`)
  - All models output 1024d vectors (OpenAI uses `dimensions=1024` parameter)
  - Asymmetric search: `input_type="document"` for catalog, `input_type="query"` for questions
- **Chat/DDL:** Configurable via `CHAT_MODEL` env var (default: `gpt-5.2`)
  - Claude models use `system=` top-level param, response at `message.content[0].text`
- **Vector index:** StreamingDiskANN (pgvectorscale) — production-grade ANN
- **Data seed:** 42 (fully reproducible)

### Environment Variables

```bash
# Embedding providers (set the ones you need)
export VOYAGE_API_KEY="pa-..."            # Voyage AI
export OPENAI_API_KEY="sk-..."            # OpenAI (embeddings + LLM)
export ANTHROPIC_API_KEY="sk-ant-..."     # Anthropic (Claude LLM)

# Ollama (self-hosted, no API key needed)
export OLLAMA_ENDPOINT="http://localhost:11434"    # default
export OLLAMA_EMBED_MODEL="mxbai-embed-large"      # default

# Default model selection
export DEFAULT_EMBEDDING_MODEL="voyage-finance-2"  # or text-embedding-3-small, etc.
export CHAT_MODEL="gpt-5.2"                        # or claude-sonnet-4-6, etc.
export CHAT_MODEL_FAST="gpt-5-mini"                # or claude-haiku-4-5, etc.

# Infrastructure
export S3_ENDPOINT="http://localhost:9000"    # MinIO (default)
export DB_HOST="localhost"                     # PG (default)
export DB_PORT="5433"                          # PG (default)
```

## File Structure

```
lab/04_metadata_rag_mart/
├── README.md
├── SKELETON-lab04.md              # reference doc (original design)
├── requirements.txt
├── docker/
│   ├── Dockerfile.pg18              # PG 18 + pgvector + pgvectorscale
│   └── docker-compose.yml           # PG + MinIO + Ollama
├── sql/
│   ├── 00_extensions.sql            # vector, vectorscale, pgcrypto
│   ├── 01_metadata_catalog.sql      # pgvector catalog (detail_bi/detail_agent)
│   ├── 02_governance.sql            # roles, audit, masking, registry
│   └── 03_rag_monitoring.sql        # search logs, judgments, evaluations
├── python/
│   ├── config.py                    # shared configuration + model registries
│   ├── embedding_provider.py        # unified embedding interface (4 providers)
│   ├── llm_provider.py              # unified LLM interface (3 providers)
│   ├── 00_generate_parquet.py       # → Parquet → MinIO (18 tables)
│   ├── 10_scan_and_embed.py         # Parquet schemas → pgvector catalog
│   ├── 20_agent_rag_search.py       # Agent 1: RAG metadata search
│   ├── 30_agent_pipeline.py         # Agent 2: Parquet → PG data mart
│   ├── 35_agent_broker.py           # Agent 3: Broker intelligence
│   ├── 40_demo.py                   # demo orchestrator (3 scenarios)
│   ├── 45_demo_broker.py            # broker intelligence demo
│   ├── 50_evaluate_rag.py           # RAG quality evaluation (multi-model)
│   ├── 60_streamlit_comparison.py   # Streamlit comparison dashboard
│   └── test_no_api.py               # full test without API keys
└── benchmarks/
    └── golden_queries.json          # 16 annotated benchmark queries
```

## Connection to Presentation

**Slide 29 (Before):** Manual SQL queries → 10 min, BI connector timeouts
**Slide 30 (After):** This lab — user asks in natural language, RAG searches metadata, Pipeline Agent provisions governed data mart in seconds

**Why PostgreSQL at the center:**
1. **pgvector** — metadata search (no sensitive data exposure)
2. **Governance** — audit trail, masking, RLS in SQL
3. **Data mart** — fast JOINs, indexes for BI queries
4. **Monitoring** — RAG quality metrics in the same DB
5. **One database** — metadata, governance, monitoring, compute
