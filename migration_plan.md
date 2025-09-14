# Migration Plan: text-embedding-3-large + GPT-5 mini

## Overview
Migrate from current embedding models and GPT versions to:
- **Embedding Model**: `text-embedding-3-large` (3072 dimensions)
- **Generation Model**: `gpt-5-mini` (GPT-5 mini)

### GPT-5 mini Specifications:
- **Context Window**: 400,000 tokens
- **Max Output**: 128,000 tokens
- **Knowledge Cutoff**: May 31, 2024
- **Features**: Reasoning token support
- **Rate Limits**: Varies by tier (Tier 5: 30,000 RPM, 180M TPM)

## Phase 1: Database Schema Changes

### 1.1 Alter Wikipedia Tables
```sql
-- Add new columns for 3072-dimensional vectors
ALTER TABLE articles ADD COLUMN title_vector_3072 vector(3072);
ALTER TABLE articles ADD COLUMN content_vector_3072 vector(3072);

-- After migration, optionally drop old columns and rename
-- ALTER TABLE articles DROP COLUMN title_vector;
-- ALTER TABLE articles DROP COLUMN content_vector;
-- ALTER TABLE articles RENAME COLUMN title_vector_3072 TO title_vector;
-- ALTER TABLE articles RENAME COLUMN content_vector_3072 TO content_vector;
```

### 1.2 Alter Movie/Netflix Tables
```sql
-- Add new columns for 3072-dimensional vectors
ALTER TABLE film ADD COLUMN embedding_3072 vector(3072);
ALTER TABLE netflix_shows ADD COLUMN embedding_3072 vector(3072);

-- After migration, optionally drop old columns and rename
-- ALTER TABLE film DROP COLUMN embedding;
-- ALTER TABLE netflix_shows DROP COLUMN embedding;
-- ALTER TABLE film RENAME COLUMN embedding_3072 TO embedding;
-- ALTER TABLE netflix_shows RENAME COLUMN embedding_3072 TO embedding;
```

### 1.3 Drop and Recreate Indexes
```sql
-- Drop existing indexes
DROP INDEX IF EXISTS film_embedding_idx;
DROP INDEX IF EXISTS netflix_shows_embedding_idx;
DROP INDEX IF EXISTS film_embedding_cosine_idx;
DROP INDEX IF EXISTS netflix_shows_embedding_cosine_idx;
DROP INDEX IF EXISTS film_embedding_ivfflat_idx;
DROP INDEX IF EXISTS netflix_embedding_ivfflat_cosine_idx;

-- Recreate with new dimensions (after column rename)
CREATE INDEX film_embedding_idx ON film USING hnsw (embedding vector_l2_ops);
CREATE INDEX netflix_shows_embedding_idx ON netflix_shows USING hnsw (embedding vector_l2_ops);
-- Add other indexes as needed
```

## Phase 2: Code Changes

### 2.1 Update Embedding Generation Scripts

#### Files to Modify:
1. **`original/create_emb_wiki.py`**
   - Line 25: `MODEL_NAME = "text-embedding-3-large"`
   - Line 26: `EMBEDDING_DIMENSION = 3072`
   - Line 40: Update dimensions parameter
   - Lines 64, 68: Update column names to use `_3072` suffix temporarily

2. **`original/create_emb.py`**
   - Update `MODEL_NAME = "text-embedding-3-large"`
   - Update `EMBEDDING_DIMENSION = 3072`
   - Update column references

3. **`original/create_emb_sparse_wiki.py`**
   - No changes needed (SPLADE remains the same)

4. **`original/create_emb_sparse.py`**
   - No changes needed (SPLADE remains the same)

### 2.2 Update RAG Search Scripts

#### Files to Modify:
1. **`original/RAG_search_wiki.py`**
   - Line 20: `MODEL_NAME = "text-embedding-3-large"`
   - Line 21: `EMBEDDING_DIMENSION = 3072`
   - Line 23: `GPT_MODEL = "gpt-5-mini"`
   - Line 30: Update dimensions parameter
   - Update column references in SQL queries

2. **`original/RAG_search_wiki_hybrid.py`**
   - Update `DENSE_MODEL = "text-embedding-3-large"`
   - Update `EMBEDDING_DIMENSION = 3072`
   - Update `GPT_MODEL = "gpt-5-mini"`
   - Update column references

3. **`original/RAG_search.py`**
   - Update `MODEL_NAME = "text-embedding-3-large"`
   - Update `EMBEDDING_DIMENSION = 3072`
   - Update `GPT_MODEL = "gpt-5-mini"`

4. **`original/RAG_search_Open.py`**
   - Update embedding model references if using OpenAI embeddings

5. **`original/RAG_search_hybrid.py`**
   - Update all model references and dimensions

6. **`original/RAG_search_hybrid_simple.py`**
   - Update all model references and dimensions

### 2.3 Update Lab Core Modules

#### Files to Modify:
1. **`lab/core/config.py`**
   - Line 18: `openai_model: str = "text-embedding-3-large"`
   - Line 19: `openai_dimensions: int = 3072`
   - Line 49: `model: str = "gpt-5-mini"`

2. **`lab/core/embeddings.py`**
   - Line 73-74: Update default parameters in `OpenAIEmbedder.__init__()`
     - `model: str = "text-embedding-3-large"`
     - `dimensions: int = 3072`
   - Ensure dimension parameter is passed to OpenAI API calls

3. **`lab/core/generation.py`**
   - Line 20-21: Add to ModelType enum:
     - `GPT_5_MINI = "gpt-5-mini"`
   - Line 52-53: Add to TOKEN_LIMITS dict:
     - `"gpt-5-mini": 400000` (400k context window)
   - Line 61-62: Add to COST_PER_1K_TOKENS dict:
     - `"gpt-5-mini": {"input": TBD, "output": TBD}` (pricing not yet announced)
   - Line 66: Update default in `GenerationService.__init__()`
     - `model: str = "gpt-5-mini"`

4. **`lab/core/database.py`**
   - Update any vector dimension references
   - Check table creation/alteration logic

5. **`lab/core/search.py`**
   - Verify vector dimension handling in search operations

### 2.4 Update Lab Search Implementations

#### Files to Modify:
1. **`lab/search/simple_search.py`** (Naive RAG)
   - Uses config from `lab/core/config.py`
   - No direct changes needed if using config properly

2. **`lab/search/hybrid_search.py`** (Hybrid RAG)
   - Uses config from `lab/core/config.py`
   - Verify both dense and sparse embedding dimensions

3. **`lab/search/adaptive_search.py`** (Adaptive RAG)
   - Uses config from `lab/core/config.py`
   - Verify query classification still works with new model

### 2.5 Update Lab Embedding Scripts

#### Files to Modify:
1. **`lab/embeddings/embedding_manager.py`**
   - Update to use new embedding dimensions
   - Verify batch processing logic

2. **`lab/embeddings/generate_embeddings.py`**
   - Update embedding model and dimensions
   - Ensure proper column mapping

3. **`lab/embeddings/verify_embeddings.py`**
   - Update expected dimensions to 3072

### 2.6 Update API and UI

#### Files to Modify:
1. **`lab/api/fastapi_server.py`**
   - No direct changes if using config
   - Verify endpoints handle new dimensions

2. **`lab/api/streamlit_app.py`**
   - Update any UI elements showing model info
   - Verify cost calculations with new models

### 2.7 Update Setup Files

#### Files to Modify:
1. **`lab/setup/setup.sql`**
   - Update all `vector(1536)` to `vector(3072)`

2. **`indexes.sql`**
   - Update example queries with new dimensions

## Phase 3: Migration Execution Steps

### Step 1: Backup Current Data
```bash
pg_dump -U postgres -d wikipedia > wikipedia_backup.sql
pg_dump -U postgres -d dvdrental > dvdrental_backup.sql
```

### Step 2: Update Database Schema
Run the ALTER TABLE commands from Phase 1.1 and 1.2

### Step 3: Update Code Files
Apply all changes from Phase 2

### Step 4: Regenerate Embeddings
```bash
# Set environment variables
export DATABASE_URL="postgresql://postgres@localhost/wikipedia"
export OPENAI_API_KEY="your_key"

# Generate new Wikipedia embeddings
python3 original/create_emb_wiki.py

# For movie database
export DATABASE_URL="postgresql://postgres@localhost/dvdrental"
python3 original/create_emb.py
```

### Step 5: Recreate Indexes
Run the index creation commands from Phase 1.3

### Step 6: Test RAG Search
```bash
# Test Wikipedia search
python3 original/RAG_search_wiki.py
python3 original/RAG_search_wiki_hybrid.py

# Test movie search
python3 original/RAG_search.py
python3 original/RAG_search_hybrid.py
```

### Step 7: Clean Up (Optional)
After verifying everything works:
1. Drop old vector columns
2. Rename new columns to original names
3. Update code to use original column names

## Cost Considerations

### Embedding Costs
- **text-embedding-3-small**: $0.020 per 1M tokens
- **text-embedding-3-large**: $0.130 per 1M tokens (6.5x more expensive)

### Generation Costs
- **gpt-3.5-turbo**: $0.50/$1.50 per 1M tokens (input/output)
- **gpt-4o-mini**: $0.15/$0.60 per 1M tokens (input/output)
- **gpt-5-mini**: Pricing TBD (not yet announced)

## Performance Considerations

1. **Storage**: 3072-dimensional vectors require 2x storage (12KB vs 6KB per vector)
2. **Index Size**: HNSW indexes will be approximately 2x larger
3. **Query Performance**: May be slightly slower due to larger dimensions
4. **Embedding Quality**: text-embedding-3-large provides better semantic understanding
5. **GPT-5 mini Advantages**:
   - 400k context window (3x larger than GPT-4o)
   - 128k max output tokens
   - Reasoning token support for better chain-of-thought
   - More recent knowledge cutoff (May 31, 2024)

## Rollback Plan

If issues occur:
1. Keep original columns intact during migration
2. Maintain backup of original code
3. Can switch back by updating configuration
4. Restore from database backups if needed

## Testing Checklist

### Database and Infrastructure
- [ ] Database schema updated successfully (3072 dimensions)
- [ ] New embeddings generated without errors
- [ ] Indexes created successfully
- [ ] Verify embedding dimensions are 3072

### Original Scripts Testing
- [ ] Wikipedia naive RAG search works (`RAG_search_wiki.py`)
- [ ] Wikipedia hybrid RAG search works (`RAG_search_wiki_hybrid.py`)
- [ ] Movie naive RAG search works (`RAG_search.py`)
- [ ] Movie hybrid RAG search works (`RAG_search_hybrid.py`)
- [ ] Simplified hybrid search works (`RAG_search_hybrid_simple.py`)
- [ ] Netflix recommendations work (`recommend_netflix.py`)

### Lab Implementation Testing
- [ ] Simple/Naive RAG works (`lab/search/simple_search.py`)
- [ ] Hybrid RAG works (`lab/search/hybrid_search.py`)
- [ ] Adaptive RAG works (`lab/search/adaptive_search.py`)
- [ ] FastAPI endpoints functional (`lab/api/fastapi_server.py`)
- [ ] Streamlit UI works (`lab/api/streamlit_app.py`)
- [ ] n8n workflows execute properly

### Performance and Cost
- [ ] Query performance acceptable (< 2x slower)
- [ ] Index size reasonable (~ 2x larger)
- [ ] Cost tracking shows expected rates:
  - Embedding: 6.5x increase (but one-time)
  - Generation: 3-4x decrease (ongoing savings)
- [ ] Memory usage acceptable

## Summary of All Files to Update

**Total: ~25 Python files + SQL schemas**

### Critical Files (Must Update):
1. All embedding generation scripts (4 files)
2. All RAG search scripts (10+ files)
3. Core configuration files (5 files)
4. Database schemas and indexes

### RAG Implementation Types:
- **Naive/Simple RAG**: Basic vector similarity search
- **Hybrid RAG**: Combines dense + sparse embeddings
- **Adaptive RAG**: Query classification with dynamic weights