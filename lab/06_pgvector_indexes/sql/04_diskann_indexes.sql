-- =============================================================================
-- Lab 06 — 04_diskann_indexes.sql
-- DiskANN / pgvectorscale: native 3072d, CONCURRENTLY, filtered search
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. DiskANN on vector(3072) — works natively (up to 16K dims)
-- ─────────────────────────────────────────────────────────────────────────────
-- StreamingDiskANN (pgvectorscale) supports up to 16,384 dimensions.
-- No halfvec workaround needed.

DROP INDEX IF EXISTS idx_content_diskann;

\timing on

CREATE INDEX idx_content_diskann
ON articles USING diskann (content_vector vector_cosine_ops)
WITH (
    storage_layout = 'memory_optimized',
    num_neighbors  = 50,
    search_list_size = 100,
    max_alpha      = 1.2
);

\timing off

SELECT pg_size_pretty(pg_relation_size('idx_content_diskann')) AS diskann_size;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. DiskANN build parameters explained
-- ─────────────────────────────────────────────────────────────────────────────
-- storage_layout:
--   'memory_optimized'  → stores full vectors in the index (faster queries, more RAM)
--   'plain'             → stores only graph structure (less RAM, heap fetch needed)
--
-- num_neighbors:   max edges per node (like HNSW m but for the Vamana graph)
-- search_list_size: candidate list during build (like HNSW ef_construction)
-- max_alpha:       pruning strictness (1.0 = strict, higher = more edges kept)

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. CREATE INDEX CONCURRENTLY (pgvectorscale 0.9.0 feature)
-- ─────────────────────────────────────────────────────────────────────────────
-- DiskANN supports CONCURRENTLY since pgvectorscale 0.9.0.
-- This does NOT block writes during the build — essential for production.

-- NOTE: Cannot run CONCURRENTLY inside a transaction block.
-- Run this manually outside of a \i if needed:

-- DROP INDEX IF EXISTS idx_content_diskann_concurrent;
-- CREATE INDEX CONCURRENTLY idx_content_diskann_concurrent
-- ON articles USING diskann (content_vector vector_cosine_ops)
-- WITH (storage_layout = 'memory_optimized', num_neighbors = 50);

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. Query with DiskANN
-- ─────────────────────────────────────────────────────────────────────────────

EXPLAIN ANALYZE
SELECT id, title, content_vector <=> (
    SELECT content_vector FROM articles WHERE id = 1
) AS distance
FROM articles
WHERE content_vector IS NOT NULL
ORDER BY content_vector <=> (
    SELECT content_vector FROM articles WHERE id = 1
)
LIMIT 10;

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. DiskANN query-time tuning
-- ─────────────────────────────────────────────────────────────────────────────
-- diskann.query_search_list_size controls query-time candidate expansion.
-- Higher = better recall, slower queries.

SET diskann.query_search_list_size = 50;
EXPLAIN ANALYZE
SELECT id, title, content_vector <=> (
    SELECT content_vector FROM articles WHERE id = 1
) AS distance
FROM articles
WHERE content_vector IS NOT NULL
ORDER BY content_vector <=> (
    SELECT content_vector FROM articles WHERE id = 1
)
LIMIT 10;

SET diskann.query_search_list_size = 200;
EXPLAIN ANALYZE
SELECT id, title, content_vector <=> (
    SELECT content_vector FROM articles WHERE id = 1
) AS distance
FROM articles
WHERE content_vector IS NOT NULL
ORDER BY content_vector <=> (
    SELECT content_vector FROM articles WHERE id = 1
)
LIMIT 10;

RESET diskann.query_search_list_size;

-- ─────────────────────────────────────────────────────────────────────────────
-- 6. Index size comparison: HNSW (halfvec) vs IVFFlat vs DiskANN
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS size,
    pg_relation_size(indexname::regclass) AS size_bytes
FROM pg_indexes
WHERE tablename = 'articles'
AND (indexname LIKE '%hnsw%' OR indexname LIKE '%ivfflat%' OR indexname LIKE '%diskann%')
ORDER BY pg_relation_size(indexname::regclass);
