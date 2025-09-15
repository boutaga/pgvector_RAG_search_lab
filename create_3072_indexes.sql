-- Create indexes for 3072-dimension vector columns
-- Run this after populating the new vector columns

-- Drop old indexes if they exist (optional cleanup)
DROP INDEX IF EXISTS idx_articles_content_vec_hnsw;
DROP INDEX IF EXISTS idx_articles_title_vec_hnsw;

-- HNSW index for content vector (cosine similarity)
CREATE INDEX IF NOT EXISTS articles_content_vector_3072_hnsw
ON articles USING hnsw (content_vector_3072 vector_cosine_ops)
WITH (m = 16, ef_construction = 128);

-- HNSW index for title vector (cosine similarity)
CREATE INDEX IF NOT EXISTS articles_title_vector_3072_hnsw
ON articles USING hnsw (title_vector_3072 vector_cosine_ops)
WITH (m = 16, ef_construction = 128);

-- Optional: L2 distance indexes (for different similarity metrics)
-- CREATE INDEX IF NOT EXISTS articles_content_vector_3072_l2
-- ON articles USING hnsw (content_vector_3072 vector_l2_ops)
-- WITH (m = 16, ef_construction = 128);

-- CREATE INDEX IF NOT EXISTS articles_title_vector_3072_l2
-- ON articles USING hnsw (title_vector_3072 vector_l2_ops)
-- WITH (m = 16, ef_construction = 128);

-- Note: If using pgvectorscale with DiskANN, use this syntax instead:
-- CREATE INDEX articles_content_vector_3072_diskann
-- ON articles USING diskann (content_vector_3072)
-- WITH (storage_layout = 'memory_optimized', num_neighbors = 50);

-- Update table statistics for query planner
ANALYZE articles;

-- Verify indexes were created
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'articles'
AND indexname LIKE '%3072%';