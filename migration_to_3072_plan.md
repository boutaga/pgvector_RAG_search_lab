# Migration Plan: Using 3072-Dimension Vector Columns

## Current State
- Database has both old (1536) and new (3072) vector columns:
  - `title_vector` (1536) → `title_vector_3072` (3072)
  - `content_vector` (1536) → `content_vector_3072` (3072)
- Search scripts currently use old columns
- Configuration shows 3072 dimensions but scripts reference old columns

## Migration Steps

### Phase 1: Configuration Update
Add vector column configuration to specify which columns to use.

### Phase 2: Code Updates

#### 1. Update Search Scripts
Files to modify:
- `lab/search/simple_search.py` - Line 90
- `lab/search/hybrid_search.py` - Line 96
- `lab/search/adaptive_search.py` - Line 307

Change from `content_vector` to `content_vector_3072`

#### 2. Update API Status Checks
Files to modify:
- `lab/api/fastapi_server.py` - Line 432
- `lab/api/streamlit_app.py` - Line 76

Update queries to check new columns.

#### 3. Update Embedding Generation
- `lab/embeddings/embedding_manager.py` - Already targets new columns (line 516)
- `lab/embeddings/generate_embeddings.py` - Update statistics checks (lines 187, 190)

#### 4. Update Verification Scripts
- `lab/embeddings/verify_embeddings.py` - Update column references (lines 158-159)

### Phase 3: Database Optimization

#### Create Indexes for New Columns
```sql
-- HNSW indexes for fast similarity search
CREATE INDEX articles_title_vector_3072_hnsw
ON articles USING hnsw (title_vector_3072 vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX articles_content_vector_3072_hnsw
ON articles USING hnsw (content_vector_3072 vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Optional: L2 distance indexes
CREATE INDEX articles_title_vector_3072_l2
ON articles USING hnsw (title_vector_3072 vector_l2_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX articles_content_vector_3072_l2
ON articles USING hnsw (content_vector_3072 vector_l2_ops)
WITH (m = 16, ef_construction = 64);
```

### Phase 4: Testing & Validation
1. Verify embedding generation works with new columns
2. Test search functionality with new embeddings
3. Compare performance with old vs new embeddings
4. Validate API endpoints return correct results

## Implementation Order
1. Add configuration option for vector columns (backwards compatible)
2. Update all search scripts to use configurable column names
3. Update API status checks
4. Create database indexes
5. Test end-to-end

## Rollback Plan
Since both old and new columns exist:
1. Keep old column references in config as fallback
2. Can switch back by changing configuration
3. No data loss as old embeddings remain intact