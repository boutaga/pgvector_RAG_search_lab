-- Create indexes for 3072-dimension vector columns using pgvectorscale DiskANN
-- Using correct operator classes for pgvectorscale

-- Drop old indexes if they exist
DROP INDEX IF EXISTS articles_content_vector_3072_diskann;
DROP INDEX IF EXISTS articles_title_vector_3072_diskann;
DROP INDEX IF EXISTS articles_content_vector_3072_ivfflat;
DROP INDEX IF EXISTS articles_title_vector_3072_ivfflat;

-- DiskANN index for content vector (cosine distance)
-- Using vector_cosine_ops which is the correct operator class
CREATE INDEX articles_content_vector_3072_diskann
ON articles USING diskann (content_vector_3072 vector_cosine_ops)
WITH (
    storage_layout = 'memory_optimized',
    num_neighbors = 50,
    search_list_size = 100,
    max_alpha = 1.2
);

-- DiskANN index for title vector (cosine distance)
CREATE INDEX articles_title_vector_3072_diskann
ON articles USING diskann (title_vector_3072 vector_cosine_ops)
WITH (
    storage_layout = 'memory_optimized',
    num_neighbors = 50,
    search_list_size = 100,
    max_alpha = 1.2
);

-- Alternative: Use L2 distance operator
-- CREATE INDEX articles_content_vector_3072_diskann_l2
-- ON articles USING diskann (content_vector_3072 vector_l2_ops)
-- WITH (
--     storage_layout = 'memory_optimized',
--     num_neighbors = 50,
--     search_list_size = 100,
--     max_alpha = 1.2
-- );

-- Alternative: Use inner product (IP) operator
-- CREATE INDEX articles_content_vector_3072_diskann_ip
-- ON articles USING diskann (content_vector_3072 vector_ip_ops)
-- WITH (
--     storage_layout = 'memory_optimized',
--     num_neighbors = 50,
--     search_list_size = 100,
--     max_alpha = 1.2
-- );

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