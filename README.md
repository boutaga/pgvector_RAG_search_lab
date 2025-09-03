# Movies_pgvector_lab

This repository is intended for educational purposes, providing a lab environment to explore pgvector and similarity searches on PostgreSQL.

The lab supports two main scenarios:
1. **Movie Recommendations**: Perform similarity searches to recommend Netflix shows to DVDRental users based on their rental profiles
2. **Wikipedia RAG Search**: Query a database of 25,000 Wikipedia articles using advanced retrieval-augmented generation techniques

## Files

### Wikipedia Scripts (New)
- `create_emb_wiki.py`: Generates dense embeddings for Wikipedia articles using OpenAI text-embedding-3-small model
- `create_emb_sparse_wiki.py`: Generates sparse embeddings for Wikipedia articles using SPLADE model  
- `RAG_search_wiki.py`: Simple Wikipedia RAG search using dense embeddings
- `RAG_search_wiki_hybrid.py`: Advanced hybrid Wikipedia RAG with query classification and dense/sparse search combination

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

The hybrid Wikipedia search supports:
- **Adaptive query classification**: Automatically detects factual, conceptual, or exploratory queries
- **Dense and sparse vector search**: Uses both OpenAI embeddings and SPLADE
- **Intelligent re-ranking**: Adjusts weights based on query type
- **Multiple search modes**: dense-only, sparse-only, hybrid, or adaptive
- **Comprehensive answer generation**: GPT-powered responses with source attribution

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
