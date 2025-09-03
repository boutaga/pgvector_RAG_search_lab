# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PostgreSQL pgvector lab for exploring similarity searches and RAG (Retrieval-Augmented Generation). The lab supports two main scenarios:

1. **Movie/Netflix Recommendations**: Combines DVDRental and Netflix datasets to recommend content based on user rental history using both dense (OpenAI) and sparse (SPLADE) embeddings.

2. **Wikipedia RAG Search**: Query and analyze a database of 25,000 Wikipedia articles using advanced retrieval-augmented generation with query classification and hybrid search techniques.

## Key Architecture Components

### Core Scripts

#### Wikipedia Scripts (New)
- **Embedding Generation**: `create_emb_wiki.py` (dense), `create_emb_sparse_wiki.py` (sparse)
- **RAG Search**: `RAG_search_wiki.py` (simple), `RAG_search_wiki_hybrid.py` (advanced with query classification)

#### Original Movie/Netflix Scripts  
- **Embedding Generation**: `create_emb.py` (dense/OpenAI), `create_emb_sparse.py` (sparse/SPLADE)
- **RAG Search**: Multiple implementations from simple (`RAG_search.py`) to hybrid with query classification (`RAG_search_hybrid.py`)
- **Recommendation Engine**: `recommend_netflix.py` for customer-based recommendations

### Data Flow
1. PostgreSQL database with pgvector extension hosts film and Netflix show data
2. Embeddings are generated via OpenAI API (dense) and SPLADE model (sparse)
3. Similarity searches use vector distance operations with optional re-ranking
4. RAG searches can route to semantic or structured SQL queries based on classification

## Common Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv pgvector_lab
source pgvector_lab/bin/activate

# Install dependencies
pip install psycopg2-binary openai pgvector transformers torch sentencepiece
```

### Running Scripts

#### Wikipedia Scripts
```bash
# Set environment for Wikipedia database
export DATABASE_URL="postgresql://postgres@localhost/wikipedia"

# Generate embeddings (run in order)
python3 create_emb_wiki.py
python3 create_emb_sparse_wiki.py  # optional

# Run Wikipedia RAG search (interactive)
python3 RAG_search_wiki.py
python3 RAG_search_wiki_hybrid.py  # advanced with query classification
```

#### Original Movie/Netflix Scripts
```bash
# Set environment for movie database  
export DATABASE_URL="postgresql://postgres@localhost/dvdrental"

# Generate embeddings (run in order)
python3 create_emb.py
python3 create_emb_sparse.py

# Run RAG search (interactive)
python3 RAG_search_hybrid.py

# Generate recommendations
python3 recommend_netflix.py
```

### Required Environment Variables
```bash
# For Wikipedia scenarios
export DATABASE_URL="postgresql://postgres@localhost/wikipedia"
export OPENAI_API_KEY="your_openai_api_key"

# For Movie/Netflix scenarios  
export DATABASE_URL="postgresql://postgres@localhost/dvdrental"
export OPENAI_API_KEY="your_openai_api_key"
```

## Database Schema Requirements

### Wikipedia Database
- **articles** table with: id, title, content, title_vector (vector), content_vector (vector)
- Optional sparse columns: title_sparse (sparsevec), content_sparse (sparsevec) - added automatically
- HNSW/IVFFlat indexes on vector columns for performance

### Movie/Netflix Database  
- **film** table: film_id, description, embedding (vector)
- **netflix_shows** table: show_id, description, embedding (vector), sparse_embedding (sparsevec)
- **customer**, **rental** tables for recommendations
- HNSW or IVFFlat indexes should be created for performance (see `indexes.sql`)

## Important Configuration

### Batch Processing
- Wikipedia scripts: `BATCH_SIZE = 50` (dense), `BATCH_SIZE = 5` (sparse) - optimized for article sizes
- Original scripts: `BATCH_SIZE = 30` (dense), `BATCH_SIZE = 10` (sparse) - adjust based on OpenAI rate limits
- Handle 429 errors with exponential backoff

### Embedding Models  
- **Wikipedia**: text-embedding-3-small (1536 dimensions) - newer, more efficient model
- **Original**: text-embedding-ada-002 (1536 dimensions) - legacy model
- **SPLADE**: naver/splade-cocondenser-ensembledistil - auto-detects CUDA availability

### Hybrid Search Configuration
- Wikipedia hybrid: Adaptive weights based on query classification (factual/conceptual/exploratory)
- Original hybrid: `DENSE_WEIGHT = 0.5`, `SPARSE_WEIGHT = 0.5` - static weights
- `TOP_K` parameter controls retrieval size before re-ranking

### Query Classification (Wikipedia)
- Factual queries: sparse-heavy (0.3 dense, 0.7 sparse)
- Conceptual queries: dense-heavy (0.7 dense, 0.3 sparse)  
- Exploratory queries: balanced (0.5 dense, 0.5 sparse)
- understand that this repository has been renamed but it will still remain in the same folder locally.
- This repository is meant to support people interested in discovering pgvector and RAG search techniques and support my talks I will do at some PostgreSQL related conferences.
- Understand that you don't have access to the LAB environment so you cannot run any script to test if your dev is ok.