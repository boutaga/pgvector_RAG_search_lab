-- Create indexes for 3072-dimension vector columns using IVFFlat
-- IVFFlat supports dimensions > 2000 (HNSW is limited to 2000)

-- Drop old indexes if they exist
DROP INDEX IF EXISTS articles_content_vector_3072_ivfflat;
DROP INDEX IF EXISTS articles_title_vector_3072_ivfflat;

-- IVFFlat indexes for content vector (cosine similarity)
-- Note: Data must be present before creating IVFFlat indexes
-- Lists parameter: sqrt(rows) is a good starting point
-- For 25000 rows, 100-200 lists is appropriate
CREATE INDEX articles_content_vector_3072_ivfflat
ON articles USING ivfflat (content_vector_3072 vector_cosine_ops)
WITH (lists = 150);

CREATE INDEX articles_title_vector_3072_ivfflat
ON articles USING ivfflat (title_vector_3072 vector_cosine_ops)
WITH (lists = 150);

-- Optional: L2 distance indexes
-- CREATE INDEX articles_content_vector_3072_ivfflat_l2
-- ON articles USING ivfflat (content_vector_3072 vector_l2_ops)
-- WITH (lists = 150);

-- Set IVFFlat search parameters for better recall
-- Probes controls accuracy vs speed tradeoff (higher = more accurate but slower)
SET ivfflat.probes = 10;

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

-- Check index sizes
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size
FROM pg_indexes
WHERE tablename = 'articles'
AND indexname LIKE '%3072%';