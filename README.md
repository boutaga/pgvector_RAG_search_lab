# Movies_pgvector_lab

This repository is intended for educational purposes, providing a lab environment to explore pgvector and similarity searches on PostgreSQL.

The goal is to perform similarity searches to recommend Netflix shows to DVDRental users based on their rental profiles.

## Files

- `create_emb.py`: Generates dense embeddings using OpenAI and updates the PostgreSQL database.
- `create_emb_sparse.py`: Generates sparse embeddings using SPLADE and updates the PostgreSQL database.
- `recommend_netflix.py`: Recommends Netflix shows to a customer based on their rental history using dense embeddings.
- `RAG_search.py`: Performs a dense-only retrieval-augmented generation (RAG) search using OpenAI embeddings.
- `RAG_search_hybrid.py`: Performs a hybrid RAG search using both dense and sparse embeddings, with re-ranking and support for structured SQL queries. It classifies queries as structured or semantic and routes them accordingly.
- `RAG_search_hybrid_simple.py`: A simplified version of the hybrid RAG search for experimentation or minimal setups.
- `RAG_search_Open.py`: (Optional) Variant of RAG search using OpenAI with different configurations or models.

## Installation steps

1. Provision a Linux server with PostgreSQL 17.X and pgvector 0.8.0.
2. Import dvdrental and Netflix datasets into a single PostgreSQL database.
3. Add vector and sparsevec columns to the relevant tables.
4. Set up a Python virtual environment and install dependencies.
5. Set environment variables for `DATABASE_URL` and `OPENAI_API_KEY`.
6. Run `create_emb.py` and `create_emb_sparse.py` to populate embeddings.
7. Use the RAG search scripts to perform similarity or hybrid searches.

## Python Environment Setup

```bash
python3 -m venv pgvector_lab
source pgvector_lab/bin/activate
pip install psycopg2-binary openai pgvector transformers torch sentencepiece
```

## Hybrid RAG Search

The `RAG_search_hybrid.py` script supports:
- Dense and sparse vector search
- Weighted re-ranking
- Structured query classification and SQL execution
- Transparent output of SQL queries and intermediate results

Example usage:
```bash
python3 RAG_search_hybrid.py
```

You will be prompted to enter a question. The script will classify the query and either:
- Run a semantic search using dense/sparse embeddings
- Or generate and execute a SQL query for structured data retrieval
