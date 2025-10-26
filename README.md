# pgvector_RAG_search_lab v1.2

This repository is intended for educational purposes, providing a lab environment to explore pgvector and similarity searches on PostgreSQL.

## Release 1.2 Updates
- ✅ Added Wikipedia database with 25,000 articles and pre-computed embeddings
- ✅ New Wikipedia-specific Python scripts with text-embedding-3-small model
- ✅ Advanced hybrid RAG search with query classification
- ✅ Git LFS integration for large database files
- ✅ Comprehensive documentation updates

The lab supports two main scenarios:
1. **Movie Recommendations**: Perform similarity searches to recommend Netflix shows to DVDRental users based on their rental profiles
2. **Wikipedia RAG Search**: Query a database of 25,000 Wikipedia articles using advanced retrieval-augmented generation techniques

## RAG Implementations

This repository implements **four distinct RAG approaches**, from simple to autonomous:

1. **Naive/Simple RAG** (`lab/search/simple_search.py`) - Basic vector similarity search with direct retrieval
2. **Hybrid RAG** (`lab/search/hybrid_search.py`) - Combines dense and sparse embeddings with fixed weights
3. **Adaptive RAG** (`lab/search/adaptive_search.py`) - Query classification with dynamic weight adjustment
4. **Agentic RAG** (`lab/search/agentic_search.py`) - **NEW!** LLM agent autonomously decides when to retrieve information

## Repository Contents

This repository now includes:
- **Original datasets**: `dvdrental.zip` and `netflix_titles.csv.zip` 
- **Wikipedia database**: `vector_database_wikipedia_articles_embedded.zip` (667MB, managed with Git LFS)
- **Python scripts**: Both original movie/Netflix scripts and new Wikipedia-specific scripts
- **SQL files**: Database indexes and sample queries

## Files

### Wikipedia Scripts (New)
- `create_emb_wiki.py`: Generates dense embeddings for Wikipedia articles using OpenAI text-embedding-3-small model
- `create_emb_sparse_wiki.py`: Generates sparse embeddings for Wikipedia articles using SPLADE model
- `RAG_search_wiki.py`: Simple Wikipedia RAG search using dense embeddings
- `RAG_search_wiki_hybrid.py`: Advanced hybrid Wikipedia RAG with query classification and dense/sparse search combination

### Lab Search Modules (Modular Architecture)
- `lab/search/simple_search.py`: Naive RAG implementation
- `lab/search/hybrid_search.py`: Hybrid RAG with dense + sparse combination
- `lab/search/adaptive_search.py`: Adaptive RAG with query classification
- `lab/search/agentic_search.py`: **NEW!** Agentic RAG with LLM-driven retrieval decisions

### Original Movie/Netflix Scripts
- `create_emb.py`: Generates dense embeddings using OpenAI and updates the PostgreSQL database.
- `create_emb_sparse.py`: Generates sparse embeddings using SPLADE and updates the PostgreSQL database.
- `recommend_netflix.py`: Recommends Netflix shows to a customer based on their rental history using dense embeddings.
- `RAG_search.py`: Performs a dense-only retrieval-augmented generation (RAG) search using OpenAI embeddings.
- `RAG_search_hybrid.py`: Performs a hybrid RAG search using both dense and sparse embeddings, with re-ranking and support for structured SQL queries. It classifies queries as structured or semantic and routes them accordingly.
- `RAG_search_hybrid_simple.py`: A simplified version of the hybrid RAG search for experimentation or minimal setups.
- `RAG_search_Open.py`: (Optional) Variant of RAG search using OpenAI with different configurations or models.

## Installation steps

### General Setup
1. Provision a Linux server with PostgreSQL 17.X and pgvector 0.8.0.
2. Set up a Python virtual environment and install dependencies.
3. Set environment variables for `DATABASE_URL` and `OPENAI_API_KEY`.

### For Movie/Netflix Scenario
4a. Import dvdrental and Netflix datasets into a single PostgreSQL database.
5a. Add vector and sparsevec columns to the relevant tables.
6a. Run `create_emb.py` and `create_emb_sparse.py` to populate embeddings.
7a. Use the original RAG search scripts to perform similarity or hybrid searches.

### For Wikipedia Scenario  
4b. Import Wikipedia articles dataset (25,000 articles) into PostgreSQL database.
5b. Ensure the articles table has the required vector columns (already present in provided schema).
6b. Run `create_emb_wiki.py` to generate dense embeddings for titles and content.
7b. Optionally run `create_emb_sparse_wiki.py` to add sparse embeddings.
8b. Use `RAG_search_wiki.py` or `RAG_search_wiki_hybrid.py` for Wikipedia search.

## Python Environment Setup

```bash
python3 -m venv pgvector_lab
source pgvector_lab/bin/activate
pip install psycopg2-binary openai pgvector transformers torch sentencepiece
```

## Usage Examples

### Wikipedia RAG Search

Generate embeddings for Wikipedia articles:
```bash
# Set environment variables
export DATABASE_URL="postgresql://postgres@localhost/wikipedia"
export OPENAI_API_KEY="your_openai_api_key"

# Generate dense embeddings (required)
python3 create_emb_wiki.py

# Generate sparse embeddings (optional)
python3 create_emb_sparse_wiki.py

# Simple Wikipedia search
python3 RAG_search_wiki.py

# Advanced hybrid search with query classification
python3 RAG_search_wiki_hybrid.py
```

### Agentic RAG Search (Latest Addition!)

The Agentic RAG approach represents the cutting edge of retrieval-augmented generation:

```bash
# Set environment variable
export DATABASE_URL="postgresql://postgres@localhost/wikipedia"
export OPENAI_API_KEY="your_openai_api_key"

# Run agentic search
python3 lab/search/agentic_search.py --source wikipedia --interactive

# Or run the demo
python3 lab/search/examples/agentic_demo.py
```

**What makes Agentic RAG special:**
- **Autonomous Decision-Making**: LLM agent decides whether to search or answer directly
- **Function Calling**: Uses OpenAI's function calling to expose search as a tool
- **Cost Efficient**: Skips unnecessary retrievals for simple questions
- **Grounded Answers**: When searching, ensures answers are based on retrieved information
- **Source Citations**: Provides transparent source attribution
- **Flexible**: Can be extended with multiple tools and multi-step retrieval

**How it works:**
1. User asks a question
2. LLM agent analyzes the query
3. Agent decides: "Do I need to search the database?"
   - Simple question → Answer directly
   - Complex/specific question → Use search_wikipedia() tool
4. If searched, agent uses retrieved snippets to formulate answer
5. Answer includes source citations and decision metadata

The hybrid Wikipedia search supports:
- **Adaptive query classification**: Automatically detects factual, conceptual, or exploratory queries
- **Dense and sparse vector search**: Uses both OpenAI embeddings and SPLADE
- **Intelligent re-ranking**: Adjusts weights based on query type
- **Multiple search modes**: dense-only, sparse-only, hybrid, adaptive, or agentic
- **Comprehensive answer generation**: GPT-powered responses with source attribution

### Blog Post Command Reference

The exact commands shown in the Agentic RAG blog post:

#### CLI Usage (JSON Output)
```bash
# Simple query with JSON output - matches blog post format
python lab/search/agentic_search.py \
  --source wikipedia \
  --query "What is PostgreSQL?" \
  --json

# Expected output:
# {"answer": "...\n\nDecision: used search", "tool_used": true, "loops": 1, "latency_ms": 1234}
```

#### CLI Usage (Human-Readable)
```bash
# With decision info and sources
python lab/search/agentic_search.py \
  --source wikipedia \
  --query "What is PostgreSQL?" \
  --show-decision \
  --show-sources
```

#### FastAPI Endpoint
```bash
# POST to /search with method=agentic
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is PostgreSQL?",
    "method": "agentic",
    "source": "wikipedia",
    "top_k": 5
  }'
```

#### Decision Tracking Example
```python
from lab.search.agentic_search import AgenticSearchEngine
from lab.evaluation.metrics import DecisionLog, summarize
from lab.core.database import DatabaseService
from lab.core.config import ConfigService

# Initialize engine
config = ConfigService()
db = DatabaseService(config.database.connection_string, 1, 5)
engine = AgenticSearchEngine(db, config, 'wikipedia')

# Run batch of queries
queries = [
    "What is 2+2?",
    "Explain PostgreSQL MVCC",
    "What is the capital of France?"
]
results = engine.batch_agentic_search(queries, top_k=5)

# Track decisions
logs = [DecisionLog.from_agentic_result(r) for r in results]

# Analyze patterns
summary = summarize(logs)
print(f"Search rate: {summary['search_rate']:.1%}")
print(f"Avg latency: {summary['avg_latency_ms']:.0f}ms")
print(f"Total cost: ${summary['total_cost']:.4f}")
```

### Original Movie/Netflix Hybrid RAG Search

The `RAG_search_hybrid.py` script supports:
- Dense and sparse vector search
- Weighted re-ranking
- Structured query classification and SQL execution
- Transparent output of SQL queries and intermediate results

Example usage:
```bash
export DATABASE_URL="postgresql://postgres@localhost/dvdrental"
python3 RAG_search_hybrid.py
```

You will be prompted to enter a question. The script will classify the query and either:
- Run a semantic search using dense/sparse embeddings
- Or generate and execute a SQL query for structured data retrieval

## Wikipedia Database Setup (Release 1.2)

### Overview
The Wikipedia database contains 25,000 articles with pre-computed embeddings using OpenAI's text-embedding-3-small model. The database includes both title and content vectors for advanced semantic search capabilities.

### Database Structure
The `articles` table contains:
- `id` (integer): Unique article identifier
- `url` (text): Wikipedia article URL
- `title` (text): Article title
- `content` (text): Article content
- `title_vector` (vector(1536)): Dense embeddings for titles
- `content_vector` (vector(1536)): Dense embeddings for content
- `vector_id` (integer): Vector identifier
- `content_tsv` (tsvector): Full-text search vectors
- `title_content_tsvector` (tsvector): Combined title and content text search

### PostgreSQL Installation Steps

#### 1. Prerequisites
Ensure you have PostgreSQL 17.x with pgvector 0.8.0+ installed:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

#### 2. Extract and Import the Database
```bash
# Extract the Wikipedia database (667MB)
unzip vector_database_wikipedia_articles_embedded.zip

# This will create a PostgreSQL dump file
# Import the database
createdb wikipedia
pg_restore -d wikipedia -v vector_database_wikipedia_articles_embedded.sql

# Alternative: if it's a plain SQL file
psql -d wikipedia -f vector_database_wikipedia_articles_embedded.sql
```

#### 3. Verify Installation
```sql
-- Connect to the database
\c wikipedia

-- Check table structure
\d articles

-- Verify data count
SELECT COUNT(*) FROM articles;
-- Expected: 25,000 articles

-- Check embedding coverage
SELECT 
  COUNT(*) as total_articles,
  COUNT(title_vector) as title_embeddings,
  COUNT(content_vector) as content_embeddings
FROM articles;

-- Sample article
SELECT id, title, LENGTH(content) as content_length 
FROM articles 
LIMIT 5;
```

#### 4. Performance Indexes (Already Included)
The database comes with optimized indexes:
```sql
-- Vector indexes for performance
"articles_content_vector_idx" ivfflat (content_vector) WITH (lists='1000')
"idx_articles_vec_hnsw" hnsw (content_vector vector_cosine_ops)
"idx_articles_vec_ivf" ivfflat (content_vector vector_cosine_ops)

-- Full-text search indexes
"idx_articles_content_tsv" gin (content_tsv)
"idx_articles_title_content_tsvector" gin (title_content_tsvector)
```

### Exporting Wikipedia Data to CSV

If you need to export the Wikipedia data for analysis:

```sql
-- Export articles metadata
COPY (
  SELECT id, url, title, LENGTH(content) as content_length,
         CASE WHEN title_vector IS NOT NULL THEN 'YES' ELSE 'NO' END as has_title_embedding,
         CASE WHEN content_vector IS NOT NULL THEN 'YES' ELSE 'NO' END as has_content_embedding
  FROM articles
) TO '/tmp/wikipedia_articles_metadata.csv' WITH CSV HEADER;

-- Export article content (without vectors)
COPY (
  SELECT id, title, content
  FROM articles
) TO '/tmp/wikipedia_articles_content.csv' WITH CSV HEADER;

-- Export embeddings as JSON (for external analysis)
COPY (
  SELECT id, title,
         array_to_json(title_vector) as title_embedding,
         array_to_json(content_vector) as content_embedding
  FROM articles
  WHERE title_vector IS NOT NULL AND content_vector IS NOT NULL
  LIMIT 100  -- Adjust as needed, full export will be very large
) TO '/tmp/wikipedia_sample_embeddings.json' WITH CSV HEADER;
```

### Using the Wikipedia Database

Once installed, you can use the Wikipedia-specific scripts:

```bash
# Set environment variable
export DATABASE_URL="postgresql://postgres@localhost/wikipedia"

# The embeddings are already computed, so you can directly run:
python3 RAG_search_wiki.py              # Simple search
python3 RAG_search_wiki_hybrid.py       # Advanced hybrid search

# Optional: Add sparse embeddings for even better hybrid search
python3 create_emb_sparse_wiki.py
```

### Database File Information
- **File**: `vector_database_wikipedia_articles_embedded.zip`
- **Size**: 667MB (compressed)
- **Articles**: 25,000 Wikipedia articles
- **Embeddings**: Pre-computed using OpenAI text-embedding-3-small
- **Storage**: Managed with Git LFS for efficient repository handling

This pre-computed database saves significant time and OpenAI API costs compared to generating embeddings from scratch.
