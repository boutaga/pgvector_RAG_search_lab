-- Create indexes for 3072-dimension vector columns using pgvectorscale DiskANN
-- DiskANN supports high dimensions and is optimized for large-scale similarity search

-- Drop old indexes if they exist
DROP INDEX IF EXISTS articles_content_vector_3072_diskann;
DROP INDEX IF EXISTS articles_title_vector_3072_diskann;
DROP INDEX IF EXISTS articles_content_vector_3072_ivfflat;
DROP INDEX IF EXISTS articles_title_vector_3072_ivfflat;

-- DiskANN index for content vector (cosine similarity)
-- DiskANN is memory-efficient and supports 3072 dimensions
CREATE INDEX articles_content_vector_3072_diskann
ON articles USING diskann (content_vector_3072 ann_cos_ops)
WITH (
    storage_layout = 'memory_optimized',
    num_neighbors = 50,
    search_list_size = 100,
    max_alpha = 1.2,
    num_dimensions = 3072
);

-- DiskANN index for title vector (cosine similarity)
CREATE INDEX articles_title_vector_3072_diskann
ON articles USING diskann (title_vector_3072 ann_cos_ops)
WITH (
    storage_layout = 'memory_optimized',
    num_neighbors = 50,
    search_list_size = 100,
    max_alpha = 1.2,
    num_dimensions = 3072
);

-- Optional: For L2 distance instead of cosine
-- CREATE INDEX articles_content_vector_3072_diskann_l2
-- ON articles USING diskann (content_vector_3072 ann_l2_ops)
-- WITH (
--     storage_layout = 'memory_optimized',
--     num_neighbors = 50,
--     search_list_size = 100,
--     max_alpha = 1.2,
--     num_dimensions = 3072
-- );

-- Set DiskANN search parameters for optimal performance
-- Adjust these based on your accuracy/speed requirements
SET diskann.query_search_list_size = 100;  -- Higher = more accurate but slower
SET diskann.query_rescore = 50;  -- Number of vectors to rescore for better accuracy

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
AND indexname LIKE '%diskann%';

-- Check index sizes
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size
FROM pg_indexes
WHERE tablename = 'articles'
AND indexname LIKE '%diskann%';

-- Show current DiskANN settings
SHOW diskann.query_search_list_size;
SHOW diskann.query_rescore;