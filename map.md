# Movies_pgvector_lab Script Map

This file indexes all Python scripts in the repository, their functions, and variable linkages.
Updated to include Wikipedia database support with 25,000 articles.

## Script Overview

## Wikipedia Scripts (New)

### 8. create_emb_wiki.py 
**Purpose**: Generate dense embeddings for Wikipedia articles using OpenAI text-embedding-3-small
**Key Functions**:
- `get_batch_embeddings(client, texts, model, max_retries=5)` - Batch embedding generation with retry logic
- `update_wikipedia_embeddings(conn, embedding_type)` - Update embeddings for titles and/or content
- `verify_embeddings(conn)` - Check embedding coverage and dimensions

**Key Variables**:
- `MODEL_NAME = "text-embedding-3-small"` - Updated OpenAI model
- `EMBEDDING_DIMENSION = 1536` - Explicit dimension setting
- `BATCH_SIZE = 50` - Increased batch size for efficiency
- Database columns: `title_vector`, `content_vector` (vector type)

### 9. create_emb_sparse_wiki.py
**Purpose**: Generate sparse embeddings for Wikipedia articles using SPLADE
**Key Functions**:
- `initialize_splade_model()` - Load SPLADE model and tokenizer
- `get_sparse_embedding(tokenizer, model, text)` - Generate sparse embeddings
- `ensure_sparse_columns(conn)` - Add sparse columns if missing
- `update_sparse_embeddings(conn, tokenizer, model, embedding_type)` - Update sparse embeddings

**Key Variables**:
- `MODEL_NAME = "naver/splade-cocondenser-ensembledistil"` - SPLADE model
- `BATCH_SIZE = 5` - Small batch for memory efficiency with large articles
- Database columns: `title_sparse`, `content_sparse` (sparsevec type)

### 10. RAG_search_wiki.py
**Purpose**: Simple Wikipedia RAG search using dense embeddings
**Key Functions**:
- `get_embedding(client, text, model)` - Generate query embedding
- `search_wikipedia_articles(query_embedding, search_type, limit)` - Search titles, content, or both
- `generate_answer(client, query, context_articles)` - GPT-based answer generation
- `display_search_results(results, query)` - Formatted result display

**Key Variables**:
- `MODEL_NAME = "text-embedding-3-small"` - Updated embedding model
- `TOP_K = 5` - Number of results to retrieve
- Searches: Wikipedia `articles` table

### 11. RAG_search_wiki_hybrid.py
**Purpose**: Advanced hybrid Wikipedia RAG with dense/sparse search and query classification
**Key Functions**:
- `classify_query(client, query)` - Classify queries as factual/conceptual/exploratory
- `get_dense_embedding(client, text)` - Dense embedding generation
- `get_sparse_embedding(text)` - Sparse embedding generation  
- `search_dense_articles(query_embedding, search_field, limit)` - Dense search
- `search_sparse_articles(sparse_embedding, search_field, limit)` - Sparse search
- `merge_and_rerank(dense_results, sparse_results)` - Weighted result combination
- `hybrid_search(client, query, search_strategy)` - Orchestrate hybrid search
- `generate_comprehensive_answer(client, query, context_articles)` - Enhanced answer generation

**Key Variables**:
- `DENSE_MODEL = "text-embedding-3-small"` - Updated dense model
- `SPARSE_MODEL = "naver/splade-cocondenser-ensembledistil"` - SPLADE model
- `TOP_K = 8`, `FINAL_K = 5` - Search and final result limits
- `DENSE_WEIGHT = 0.6`, `SPARSE_WEIGHT = 0.4` - Default re-ranking weights
- Adaptive weights based on query classification

## Original Movie/Netflix Scripts

### 1. create_emb.py (109 lines)
**Purpose**: Generate dense embeddings using OpenAI API and update PostgreSQL database
**Key Functions**:
- `get_batch_embeddings(client, texts, model, max_retries=5)` - Batch embedding generation with retry logic
- `update_table_embeddings(conn, table_name, id_column, text_column, model)` - Update embeddings for entire table

**Key Variables**:
- `BATCH_SIZE = 30` - Configurable batch size for API calls
- `model = "text-embedding-ada-002"` - OpenAI embedding model
- Database tables: `film` (film_id, description), `netflix_shows` (show_id, description)

### 2. create_emb_sparse.py (246 lines)
**Purpose**: Generate sparse embeddings using SPLADE model and update PostgreSQL database
**Key Functions**:
- `initialize_splade_model()` - Load SPLADE model and tokenizer
- `get_sparse_embedding(tokenizer, model, text)` - Generate sparse embeddings
- `update_sparse_embeddings(table_name, id_column, text_column)` - Update sparse embeddings for table

**Key Variables**:
- `MODEL_NAME = "naver/splade-cocondenser-ensembledistil"` - SPLADE model
- `BATCH_SIZE = 10` - Smaller batch for memory efficiency
- `DEVICE` - Auto-detect CUDA/CPU
- Database column: `sparse_embedding` (sparsevec type)

### 3. RAG_search.py (84 lines)
**Purpose**: Simple dense-only RAG search using OpenAI embeddings
**Key Functions**:
- `get_embedding(text, model)` - Generate query embedding
- `search_similar_items(query_embedding, limit)` - Dense vector search
- `generate_answer(query, context)` - GPT-based answer generation

**Key Variables**:
- `model = "text-embedding-ada-002"` - Embedding model
- `TOP_K = 5` - Number of results to retrieve
- Searches: `netflix_shows` table only

### 4. RAG_search_Open.py (87 lines)
**Purpose**: Alternative RAG search with different OpenAI configurations
**Key Functions**:
- Similar to RAG_search.py but with different model parameters
- `search_films_and_shows()` - Search both film and netflix_shows tables

### 5. RAG_search_hybrid.py (231 lines)
**Purpose**: Advanced hybrid RAG with dense/sparse search and query classification
**Key Functions**:
- `get_dense_embedding(text, model)` - Dense embedding generation
- `get_sparse_embedding(tokenizer, model, text)` - Sparse embedding generation
- `query_dense_similar_items(query_embedding, limit)` - Dense search
- `query_sparse_similar_items(sparse_vec_str, limit)` - Sparse search
- `merge_and_rerank(dense_results, sparse_results)` - Weighted result combination
- `classify_query(query)` - Determine if query is structured or semantic
- `generate_sql_query(query)` - Generate SQL for structured queries
- `execute_sql_query(sql)` - Execute structured queries

**Key Variables**:
- `TOP_K = 10` - Results per search type
- `DENSE_WEIGHT = 0.5`, `SPARSE_WEIGHT = 0.5` - Re-ranking weights
- `MODEL_NAME = "naver/splade-cocondenser-ensembledistil"` - SPLADE model

### 6. RAG_search_hybrid_simple.py (167 lines)
**Purpose**: Simplified hybrid search for experimentation
**Key Functions**:
- Subset of RAG_search_hybrid.py functionality
- No query classification or SQL generation

### 7. recommend_netflix.py (93 lines)
**Purpose**: Recommend Netflix shows based on customer rental history
**Key Functions**:
- `get_customer_profile(customer_id)` - Get rental history
- `recommend_shows(customer_id, limit)` - Generate recommendations

**Key Variables**:
- Uses DVDRental customer and rental data
- Combines with Netflix shows via embedding similarity

## Variable Dependencies and Environment

### Environment Variables (Required for all scripts)
- `DATABASE_URL` - PostgreSQL connection string  
  - Movies/Netflix: "postgresql://postgres@localhost/dvdrental"
  - Wikipedia: "postgresql://postgres@localhost/wikipedia"
- `OPENAI_API_KEY` - OpenAI API access key

### Database Schema Dependencies

#### Wikipedia Tables (New):
- **articles**: id, url, title, content, title_vector (vector), content_vector (vector), title_sparse (sparsevec), content_sparse (sparsevec), vector_id, content_tsv (tsvector), title_content_tsvector (tsvector)

#### Original Movie/Netflix Tables:
- **film**: film_id, description, embedding (vector)
- **netflix_shows**: show_id, description, embedding (vector), sparse_embedding (sparsevec)
- **customer**, **rental**: For recommendation system

#### Required Extensions:
- pgvector (vector, sparsevec types)
- Indexes: HNSW, IVFFlat for performance

### Model Dependencies
- OpenAI Dense: text-embedding-3-small (1536 dimensions) - **Updated**
- OpenAI Legacy: text-embedding-ada-002 (1536 dimensions) - Original scripts
- SPLADE: naver/splade-cocondenser-ensembledistil
- PyTorch/Transformers for local SPLADE inference
- GPT: gpt-3.5-turbo for answer generation

## Script Execution Flow

### Wikipedia Workflow (New):
1. **Setup Phase**: 
   - `create_emb_wiki.py` → Generate dense embeddings for titles and content
   - `create_emb_sparse_wiki.py` → Generate sparse embeddings (optional, adds columns)

2. **Usage Phase**:
   - `RAG_search_wiki.py` → Simple Wikipedia RAG search
   - `RAG_search_wiki_hybrid.py` → Advanced hybrid search with query classification

### Original Movie/Netflix Workflow:
1. **Setup Phase**: 
   - `create_emb.py` → Generate dense embeddings
   - `create_emb_sparse.py` → Generate sparse embeddings

2. **Usage Phase**:
   - `RAG_search*.py` → Interactive semantic search
   - `recommend_netflix.py` → Customer-based recommendations

## Key Configuration Points
- Batch sizes vary by script (10-30) for rate limit management
- GPU/CPU auto-detection for SPLADE model
- Configurable weights for hybrid re-ranking
- Retry logic with exponential backoff for API calls