# pgvector RAG Search Lab

A comprehensive educational repository for learning PostgreSQL pgvector, semantic search, and Retrieval-Augmented Generation (RAG) techniques. This lab provides hands-on experience with vector databases, from basic similarity search to advanced agentic AI systems.

**Version**: 1.2
**Target Audience**: Database administrators, data engineers, and developers learning vector search and RAG

---

## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Lab 1: Movie Recommendations](#lab-1-movie-recommendations)
- [Lab 2: Wikipedia RAG Search](#lab-2-wikipedia-rag-search)
- [Lab 3: Advanced RAG Techniques](#lab-3-advanced-rag-techniques)
- [Lab 4: Evaluation & Optimization](#lab-4-evaluation--optimization)
- [Database Details](#database-details)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

---

## Introduction

This repository demonstrates practical applications of pgvector for semantic search and RAG systems. It includes two main scenarios with progressively advanced techniques:

### Scenarios

1. **Movie Recommendations** - Recommend Netflix shows to DVD rental customers using similarity search
2. **Wikipedia RAG Search** - Query 25,000 Wikipedia articles using retrieval-augmented generation

### RAG Implementations (Progressive Complexity)

1. **Naive RAG** - Basic vector similarity search
2. **Hybrid RAG** - Combines dense (semantic) and sparse (keyword) embeddings
3. **Adaptive RAG** - Query classification with dynamic weight adjustment
4. **Agentic RAG** - LLM autonomously decides when to retrieve information

---

## Repository Structure

```
Movies_pgvector_lab/
│
├── original/                          # Standalone Python scripts (original implementation)
│   ├── create_emb.py                  # Generate dense embeddings for movies/Netflix
│   ├── create_emb_sparse.py           # Generate sparse (SPLADE) embeddings for movies
│   ├── create_emb_wiki.py             # Generate dense embeddings for Wikipedia
│   ├── create_emb_sparse_wiki.py      # Generate sparse embeddings for Wikipedia
│   ├── RAG_search.py                  # Simple RAG search for movies
│   ├── RAG_search_hybrid.py           # Hybrid RAG for movies with query classification
│   ├── RAG_search_wiki.py             # Simple RAG search for Wikipedia
│   ├── RAG_search_wiki_hybrid.py      # Hybrid RAG for Wikipedia
│   └── recommend_netflix.py           # Netflix recommendation engine
│
├── lab/                               # Modular architecture for advanced usage
│   ├── core/                          # Core services and utilities
│   │   ├── config.py                  # Configuration management (models, dimensions)
│   │   ├── database.py                # Database connection and pooling
│   │   ├── embeddings.py              # Embedding generation services (OpenAI, SPLADE)
│   │   ├── generation.py              # LLM text generation (GPT-4o, GPT-5-mini)
│   │   ├── ranking.py                 # Result ranking and re-ranking
│   │   └── search.py                  # Search operations (vector, sparse, hybrid)
│   │
│   ├── search/                        # RAG implementation examples
│   │   ├── simple_search.py           # Naive RAG - basic vector search
│   │   ├── hybrid_search.py           # Hybrid RAG - dense + sparse combination
│   │   ├── adaptive_search.py         # Adaptive RAG - query-based optimization
│   │   ├── agentic_search.py          # Agentic RAG - autonomous LLM agent
│   │   └── examples/
│   │       └── agentic_demo.py        # Interactive agentic RAG demonstration
│   │
│   ├── evaluation/                    # Metrics and benchmarking tools
│   │   ├── metrics.py                 # Precision, Recall, F1, nDCG, MRR calculations
│   │   ├── evaluator.py               # Evaluation framework
│   │   ├── relevance_manager.py       # Test query and relevance grade management
│   │   └── examples/
│   │       ├── k_balance_experiment.py         # Optimize k_retrieve and k_context
│   │       ├── compare_search_configs.py       # Compare search strategies
│   │       ├── demo_ranking_improvement.py     # Optimization demonstration
│   │       ├── README_K_BALANCE.md             # K-parameter optimization guide
│   │       ├── UNDERSTANDING_NDCG.md           # nDCG metric explained
│   │       └── RECALL_VS_PRECISION.md          # Basic metrics guide
│   │
│   ├── api/                           # API interfaces
│   │   ├── fastapi_server.py          # REST API with all search methods
│   │   └── streamlit_app.py           # Interactive web UI
│   │
│   ├── setup/                         # Setup and installation scripts
│   │   ├── setup.sh                   # Database setup automation
│   │   ├── setup.sql                  # SQL schema definitions
│   │   ├── evaluation_schema.sql      # Evaluation tables and functions
│   │   └── pgvectorscale_install.md   # pgvectorscale installation guide
│   │
│   └── workflows/                     # n8n workflow templates
│       ├── naive_rag_workflow.json
│       ├── hybrid_rag_workflow.json
│       ├── adaptive_rag_workflow.json
│       └── agentic_rag_workflow.json
│
├── docs/                              # Additional documentation
│
├── SQL files (root level)
│   ├── indexes.sql                    # General index creation examples
│   ├── create_3072_indexes.sql        # HNSW indexes for 3072-dim vectors
│   ├── create_3072_indexes_diskann.sql    # DiskANN indexes (production scale)
│   ├── create_3072_indexes_ivfflat.sql    # IVFFlat indexes (development)
│   └── lab_queries.sql                # Sample SQL queries
│
└── Data files (managed with Git LFS)
    ├── dvdrental.zip                  # DVD rental database
    ├── netflix_titles.csv.zip         # Netflix catalog data
    └── vector_database_wikipedia_articles_embedded.zip  # 25k Wikipedia articles (667MB)
```

---

## Prerequisites

### Required Software

- **PostgreSQL 17.x** with **pgvector 0.8.0+**
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```

- **Python 3.8+** with virtual environment
- **Git** (with Git LFS for large database files)

### Optional (for production-scale indexes)

- **pgvectorscale** - Provides DiskANN indexes for billions of vectors
  - See `lab/setup/pgvectorscale_install.md` for installation

### Required API Keys

- **OpenAI API Key** - For embedding generation and LLM responses
  - Get yours at: https://platform.openai.com/api-keys

### Python Dependencies

```bash
pip install psycopg2-binary openai pgvector transformers torch sentencepiece
```

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/Movies_pgvector_lab.git
cd Movies_pgvector_lab
```

### 2. Set Up Python Environment
```bash
python3 -m venv pgvector_lab
source pgvector_lab/bin/activate  # On Windows: pgvector_lab\Scripts\activate
pip install psycopg2-binary openai pgvector transformers torch sentencepiece
```

### 3. Configure Environment Variables
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/dbname"
export OPENAI_API_KEY="your_openai_api_key_here"
```

### 4. Choose Your Lab and Follow Instructions Below

---

## Lab 1: Movie Recommendations

**Goal**: Recommend Netflix shows to DVD rental customers based on their rental history using vector similarity.

### Database Setup

#### 1.1. Create Database
```bash
createdb dvdrental
```

#### 1.2. Import Data
```bash
# Import DVD rental database
unzip dvdrental.zip
pg_restore -d dvdrental dvdrental.tar

# Import Netflix data
unzip netflix_titles.csv.zip
psql -d dvdrental -c "CREATE TABLE netflix_shows (
    show_id VARCHAR(10) PRIMARY KEY,
    type VARCHAR(10),
    title VARCHAR(200),
    director TEXT,
    cast TEXT,
    country TEXT,
    date_added DATE,
    release_year INT,
    rating VARCHAR(10),
    duration VARCHAR(20),
    listed_in TEXT,
    description TEXT
);"
psql -d dvdrental -c "\COPY netflix_shows FROM 'netflix_titles.csv' CSV HEADER;"
```

#### 1.3. Add Vector Columns
```sql
-- Add embedding column to film table
ALTER TABLE film ADD COLUMN embedding vector(1536);

-- Add embedding columns to Netflix shows
ALTER TABLE netflix_shows ADD COLUMN embedding vector(1536);
ALTER TABLE netflix_shows ADD COLUMN sparse_embedding sparsevec;
```

### Generate Embeddings

#### Dense Embeddings (OpenAI)
```bash
export DATABASE_URL="postgresql://postgres@localhost/dvdrental"
python3 original/create_emb.py
```
**Note**: This will generate embeddings for ~1,000 films. Expect 5-10 minutes runtime.

#### Sparse Embeddings (SPLADE) - Optional
```bash
python3 original/create_emb_sparse.py
```
**Note**: Requires CUDA-capable GPU for optimal performance. CPU mode supported but slower.

### Create Indexes for Performance

```sql
-- HNSW index for dense vectors (recommended)
CREATE INDEX film_embedding_hnsw_idx
ON film
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX netflix_embedding_hnsw_idx
ON netflix_shows
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

See `indexes.sql` for additional index configurations.

### Run Searches

#### Simple RAG Search
```bash
python3 original/RAG_search.py
```
Enter your query when prompted. The system will:
1. Generate query embedding
2. Find similar films via vector search
3. Generate answer using GPT with retrieved context

#### Hybrid RAG Search (Advanced)
```bash
python3 original/RAG_search_hybrid.py
```
Supports:
- Dense + sparse vector fusion
- Query classification (semantic vs structured)
- SQL query generation for structured questions
- Weighted re-ranking

#### Netflix Recommendations
```bash
python3 original/recommend_netflix.py
```
Provides Netflix recommendations for DVD rental customers based on their rental history.

---

## Lab 2: Wikipedia RAG Search

**Goal**: Perform semantic search and RAG over 25,000 Wikipedia articles.

### Database Setup

#### 2.1. Import Wikipedia Database
```bash
# Extract database (667MB compressed, ~2GB uncompressed)
unzip vector_database_wikipedia_articles_embedded.zip

# Create database
createdb wikipedia

# Import (choose appropriate method based on file format)
# If .sql file:
psql -d wikipedia -f vector_database_wikipedia_articles_embedded.sql

# If .dump or .backup file:
pg_restore -d wikipedia -v vector_database_wikipedia_articles_embedded.dump
```

#### 2.2. Verify Installation
```sql
\c wikipedia

-- Check structure
\d articles

-- Verify data
SELECT COUNT(*) FROM articles;
-- Expected: 25,000

-- Check embeddings
SELECT
  COUNT(*) as total,
  COUNT(title_vector) as title_embeddings,
  COUNT(content_vector) as content_embeddings
FROM articles;
-- Expected: 25,000 for all counts
```

### Database Schema

The `articles` table includes:
- `id` - Unique article identifier
- `title` - Article title
- `content` - Article text content
- `url` - Wikipedia URL
- `title_vector` - Dense embeddings for titles (vector(1536))
- `content_vector` - Dense embeddings for content (vector(1536))
- `title_sparse` - Sparse embeddings for titles (sparsevec) - optional
- `content_sparse` - Sparse embeddings for content (sparsevec) - optional

**Indexes included**: HNSW and IVFFlat indexes are pre-created for optimal performance.

### Optional: Add Sparse Embeddings

The database comes with dense embeddings. To add sparse embeddings for hybrid search:

```bash
export DATABASE_URL="postgresql://postgres@localhost/wikipedia"
python3 original/create_emb_sparse_wiki.py
```
**Note**: This adds SPLADE sparse vectors. Takes 2-4 hours for 25k articles.

### Run Wikipedia Searches

#### Simple Wikipedia RAG
```bash
python3 original/RAG_search_wiki.py
```
Basic semantic search with GPT-powered answers.

#### Hybrid Wikipedia RAG (Advanced)
```bash
python3 original/RAG_search_wiki_hybrid.py
```
Features:
- **Automatic query classification**: Factual, conceptual, or exploratory
- **Dense + sparse fusion**: Optimized weight balancing
- **Adaptive search**: Different strategies per query type
- **Source attribution**: Transparent citations

---

## Lab 3: Advanced RAG Techniques

The `lab/search/` directory contains modular implementations of four RAG patterns, demonstrating progressive sophistication.

### 3.1. Naive/Simple RAG

**File**: `lab/search/simple_search.py`

Basic vector similarity search with direct retrieval.

```bash
python3 lab/search/simple_search.py --source wikipedia --query "What is machine learning?"
```

**How it works**:
1. Generate query embedding
2. Find k most similar articles (cosine distance)
3. Retrieve article content
4. Generate answer with LLM

**Best for**: Simple use cases, baseline testing

---

### 3.2. Hybrid RAG

**File**: `lab/search/hybrid_search.py`

Combines dense (semantic) and sparse (keyword) embeddings with fixed weight fusion.

```bash
python3 lab/search/hybrid_search.py --source wikipedia --query "PostgreSQL MVCC"
```

**How it works**:
1. Generate dense query embedding (OpenAI)
2. Generate sparse query embedding (SPLADE)
3. Search both indexes
4. Fuse results: `score = 0.5 * dense_score + 0.5 * sparse_score`
5. Generate answer with top-k fused results

**Best for**: Queries needing both semantic and exact keyword matching

---

### 3.3. Adaptive RAG

**File**: `lab/search/adaptive_search.py`

Query classification with dynamic weight adjustment based on query type.

```bash
python3 lab/search/adaptive_search.py --source wikipedia --query "When was PostgreSQL created?"
```

**How it works**:
1. Classify query type:
   - **Factual**: Specific facts/dates → 0.3 dense, 0.7 sparse (favor keywords)
   - **Conceptual**: Explanations/definitions → 0.7 dense, 0.3 sparse (favor semantics)
   - **Exploratory**: Open-ended → 0.5 dense, 0.5 sparse (balanced)
2. Search with adaptive weights
3. Generate contextual answer

**Best for**: Mixed query workloads where query types vary

---

### 3.4. Agentic RAG (Most Advanced)

**File**: `lab/search/agentic_search.py`

LLM agent autonomously decides whether to search the database or answer directly.

```bash
# Interactive mode
python3 lab/search/agentic_search.py --source wikipedia --interactive

# Single query with decision tracking
python3 lab/search/agentic_search.py \
    --source wikipedia \
    --query "Explain PostgreSQL MVCC" \
    --show-decision \
    --show-sources

# JSON output (for programmatic use)
python3 lab/search/agentic_search.py \
    --source wikipedia \
    --query "What is 2+2?" \
    --json
```

**How it works**:
1. User submits query to LLM agent
2. Agent has access to `search_wikipedia(query, top_k)` function
3. Agent analyzes query and decides:
   - **Simple/common knowledge** → Answer directly (no search)
   - **Specific/complex** → Invoke search function
4. If searched: Agent uses retrieved snippets to formulate grounded answer
5. Response includes decision metadata and source citations

**Decision examples**:
- "What is 2+2?" → Answers directly (no search needed)
- "Explain PostgreSQL MVCC" → Searches database, cites sources
- "What is the capital of France?" → May answer directly (common knowledge)

**Key advantages**:
- **Cost efficient**: Skips retrieval for simple questions
- **Autonomous**: No fixed pipeline, adapts to query
- **Grounded**: When searching, answers based only on retrieved snippets
- **Extensible**: Can add multiple tools (SQL executor, web search, etc.)

**Best for**: Production systems with mixed query complexity, cost-sensitive deployments

---

### Comparison: RAG Approaches

| Feature | Naive | Hybrid | Adaptive | Agentic |
|---------|-------|--------|----------|---------|
| **Retrieval** | Always dense | Always hybrid | Always adaptive | Conditional |
| **Decision Making** | None | Fixed weights | Query classifier | LLM agent |
| **Efficiency** | Low | Medium | High | Highest |
| **Flexibility** | Low | Medium | Medium | Very High |
| **Cost per Query** | Medium | High | High | Variable |
| **Complexity** | Low | Medium | High | Very High |

---

## Lab 4: Evaluation & Optimization

### 4.1. Understanding Metrics

The lab includes comprehensive evaluation tools for measuring search quality.

**Key Metrics**:
- **Precision**: % of retrieved documents that are relevant (quality)
- **Recall**: % of relevant documents that were retrieved (completeness)
- **F1**: Harmonic mean of precision and recall
- **nDCG**: Ranking quality with position awareness (0.0 to 1.0)
- **MRR**: Mean Reciprocal Rank (position of first relevant result)

**Guides**:
- `lab/evaluation/examples/RECALL_VS_PRECISION.md` - Metrics fundamentals
- `lab/evaluation/examples/UNDERSTANDING_NDCG.md` - Deep dive on ranking quality

---

### 4.2. K-Parameter Optimization

Experiment with k_retrieve (retrieval pool size) and k_context (LLM input size) to optimize the recall/precision/cost trade-off.

```bash
# Single configuration test
python3 lab/evaluation/examples/k_balance_experiment.py \
    --test-file lab/evaluation/test_cases.json \
    --k-retrieve 100 \
    --k-context 8

# Compare multiple configurations
python3 lab/evaluation/examples/k_balance_experiment.py \
    --test-file lab/evaluation/test_cases.json \
    --k-retrieve-values 50 100 200 \
    --k-context-values 5 8 10 \
    --output k_results.json
```

**What you'll learn**:
- How k_retrieve affects recall (higher = finds more relevant docs)
- How k_context affects LLM cost and answer quality
- Optimal values for different query types
- Trade-offs between retrieval quality, latency, and cost

**Recommended starting point**: k_retrieve=100-200, k_context=5-10

See `lab/evaluation/examples/README_K_BALANCE.md` for comprehensive guide.

---

### 4.3. Search Configuration Comparison

Compare different search strategies systematically.

```bash
python3 lab/evaluation/examples/compare_search_configs.py \
    --test-file lab/evaluation/test_cases.json \
    --detailed \
    --output comparison_results.json
```

Tests 9 configurations:
- Vector search with k=10, 50, 100, 200
- Sparse search (SPLADE)
- Hybrid search with different weight balances (50/50, 70/30, 30/70)
- Adaptive search

**Outputs**:
- Per-query metrics (Precision, Recall, nDCG, MRR)
- Aggregate statistics
- Latency measurements
- JSON/CSV export for analysis

---

### 4.4. Ranking Improvement Demo

See real-world optimization in action.

```bash
python3 lab/evaluation/examples/demo_ranking_improvement.py
```

**Demonstrates**:
- Title weighting optimization (70% title, 30% content)
- Before/after comparison with real queries
- Up to 49% nDCG improvement on specific queries
- Zero additional cost (SQL formula change only)

**Results**:
- Overall recall: 81.2% → 87.5% (+6%)
- Overall nDCG: 0.795 → 0.861 (+8%)
- Best case: nDCG 0.540 → 0.804 (+49%)

---

## Database Details

### Vector Dimensions

The repository supports multiple embedding dimensions:

- **1536 dimensions**: Legacy embeddings (text-embedding-ada-002, text-embedding-3-small)
  - Storage: ~6KB per vector
  - Included in Wikipedia database

- **3072 dimensions**: Newer embeddings (text-embedding-3-large)
  - Storage: ~12KB per vector
  - Better semantic understanding
  - Requires re-embedding (see migration scripts)

### Index Types

**IVFFlat** (Development):
- Fast to build (30-60 minutes)
- Slower queries (50-100ms)
- Low memory usage
- Recall: ~0.85
- Best for: <100k vectors, development

**HNSW** (Production):
- Medium build time (2-3 hours)
- Fast queries (10-20ms)
- Higher memory usage
- Recall: ~0.97
- Best for: <1M vectors, production

**DiskANN** (Scale):
- Slow build (3-4 hours)
- Fastest queries (5-15ms)
- Medium memory usage
- Recall: ~0.98
- Best for: >1M vectors, production scale
- Requires: pgvectorscale extension

### Index Creation Examples

```sql
-- HNSW (recommended for most use cases)
CREATE INDEX articles_content_hnsw_idx
ON articles
USING hnsw (content_vector vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- DiskANN (for scale)
CREATE INDEX articles_content_diskann_idx
ON articles
USING diskann (content_vector vector_cosine_ops)
WITH (
    storage_layout = memory_optimized,
    num_neighbors = 50,
    search_list_size = 100
);
```

See SQL files in root directory for complete examples.

---

## Troubleshooting

### Common Issues

#### "Extension 'vector' not found"
**Solution**: Install pgvector 0.8.0+
```bash
# Ubuntu/Debian
sudo apt-get install postgresql-17-pgvector

# From source
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

#### "Out of memory during index build"
**Solution**: Increase `maintenance_work_mem`
```sql
SET maintenance_work_mem = '2GB';
```

#### "Query timeout / slow queries"
**Solution**:
1. Ensure index exists: `\d articles`
2. Check index is being used: `EXPLAIN ANALYZE SELECT ...`
3. Adjust ef_search for HNSW: `SET hnsw.ef_search = 100;`

#### "OpenAI API rate limit"
**Solution**: Reduce batch size in embedding scripts
- Edit `BATCH_SIZE` in `create_emb*.py` files
- Add exponential backoff (already implemented)
- Consider upgrading to paid tier

#### "SPLADE model not found"
**Solution**: Model downloads automatically on first run. Ensure internet connection and sufficient disk space (~500MB).

---

## Resources

### Documentation
- **pgvector**: https://github.com/pgvector/pgvector
- **pgvectorscale**: https://github.com/timescale/pgvectorscale
- **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings
- **SPLADE**: https://github.com/naver/splade

### Guides in This Repository
- `lab/setup/pgvectorscale_install.md` - DiskANN installation
- `lab/evaluation/examples/README_K_BALANCE.md` - K-parameter tuning
- `lab/evaluation/examples/UNDERSTANDING_NDCG.md` - nDCG metrics
- `lab/evaluation/examples/RECALL_VS_PRECISION.md` - Basic metrics
- `lab/evaluation/examples/PRESENTATION_GUIDE.md` - Demo guide

### Blog Posts & Presentations
Check `docs/` directory for related blog posts and presentation materials.

---

## Contributing

This is an educational repository. Contributions welcome:
- Bug fixes
- Documentation improvements
- Additional examples
- Performance optimizations

Please ensure all scripts follow Linux (LF) line endings.

---

## License

[Your License Here]

---

## Acknowledgments

- pgvector team for excellent PostgreSQL extension
- OpenAI for embedding and generation APIs
- Timescale for pgvectorscale and DiskANN support
- Wikipedia for article dataset

---

**Questions?** Open an issue or check existing documentation in `lab/evaluation/examples/`.

**Latest Updates**: See `current_progress.md` for development status and recent changes.
