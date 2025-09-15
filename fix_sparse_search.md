# Sparse Search Index Out of Bounds Fix

## Problem
The error "sparsevec index out of bounds" occurs when running hybrid or adaptive searches that use sparse embeddings.

## Possible Causes

1. **Index Mismatch**: The SPLADE model generates indices from 0 to 30521, but the database column might expect different indexing.

2. **Empty Sparse Embeddings**: The `content_sparse` column might not be properly populated.

3. **Dimension Mismatch**: The column is defined as `sparsevec(30522)` but the generated embeddings might have different dimensions.

## Quick Workarounds

### Option 1: Disable Sparse Search (Temporary)
Use dense-only search while investigating:
```bash
# For simple search
python3 lab/search/simple_search.py --source wikipedia --search-type dense

# For hybrid search with dense-only
python3 lab/search/hybrid_search.py --source wikipedia --dense-weight 1.0 --sparse-weight 0.0
```

### Option 2: Check Sparse Embeddings
Verify if sparse embeddings exist in the database:
```sql
SELECT COUNT(*) FROM articles WHERE content_sparse IS NOT NULL;
SELECT content_sparse FROM articles WHERE content_sparse IS NOT NULL LIMIT 1;
```

### Option 3: Regenerate Sparse Embeddings
If sparse embeddings are missing or corrupted:
```bash
python3 lab/embeddings/generate_embeddings.py --source wikipedia --type sparse --batch-size 1
```

## Permanent Fix

The issue is likely that:
1. Sparse embeddings were never generated for the `content_sparse` column
2. Or they were generated with a different format/dimension

### Steps to fix:
1. Check if sparse embeddings exist in the database
2. If not, generate them using the embedding generator
3. If they exist but are corrupted, drop and regenerate them

### SQL to check sparse column status:
```sql
-- Check if sparse embeddings exist
SELECT
    COUNT(*) as total,
    COUNT(content_sparse) as with_sparse,
    pg_column_size(content_sparse) as sparse_size
FROM articles
LIMIT 10;

-- Check the format of existing sparse embeddings
SELECT
    id,
    substring(content_sparse::text, 1, 100) as sparse_sample
FROM articles
WHERE content_sparse IS NOT NULL
LIMIT 5;
```