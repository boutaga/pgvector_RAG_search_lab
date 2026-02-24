-- =============================================================================
-- Lab 06 — 02_hnsw_indexes.sql
-- HNSW deep dive: dimension limits, halfvec workaround, ef_search tuning
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. HNSW on vector(3072) → FAIL (2000-dimension limit)
-- ─────────────────────────────────────────────────────────────────────────────
-- This WILL fail with:
--   ERROR: column cannot have more than 2000 dimensions for hnsw index
-- This is the key teaching moment: HNSW has a hard 2000-dim limit for vector.

-- Uncomment to demonstrate the error:
-- CREATE INDEX idx_content_hnsw_vector
-- ON articles USING hnsw (content_vector vector_cosine_ops)
-- WITH (m = 16, ef_construction = 128);

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. HNSW on halfvec(3072) → SUCCESS (4000-dimension limit)
-- ─────────────────────────────────────────────────────────────────────────────
-- halfvec raises the limit to 4000 dimensions AND halves storage.
-- The content_halfvec column was populated from content_vector via CAST.

DROP INDEX IF EXISTS idx_content_hnsw_halfvec;

\timing on

CREATE INDEX idx_content_hnsw_halfvec
ON articles USING hnsw (content_halfvec halfvec_cosine_ops)
WITH (m = 16, ef_construction = 128);

\timing off

-- Check index size
SELECT pg_size_pretty(pg_relation_size('idx_content_hnsw_halfvec')) AS hnsw_halfvec_size;

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. HNSW build parameters explained
-- ─────────────────────────────────────────────────────────────────────────────
-- m = 16          → max edges per node (higher = better recall, more storage)
-- ef_construction → search width during build (higher = better graph, slower build)
--
-- Build with higher m and ef_construction for comparison:

DROP INDEX IF EXISTS idx_content_hnsw_halfvec_m32;

\timing on

CREATE INDEX idx_content_hnsw_halfvec_m32
ON articles USING hnsw (content_halfvec halfvec_cosine_ops)
WITH (m = 32, ef_construction = 200);

\timing off

SELECT pg_size_pretty(pg_relation_size('idx_content_hnsw_halfvec_m32')) AS hnsw_m32_size;

-- Compare sizes
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
FROM pg_indexes
WHERE tablename = 'articles'
AND indexname LIKE '%hnsw%'
ORDER BY indexname;

-- Drop the m32 variant to keep one index for further tests
DROP INDEX IF EXISTS idx_content_hnsw_halfvec_m32;

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. ef_search tuning (query-time parameter)
-- ─────────────────────────────────────────────────────────────────────────────
-- ef_search controls how many candidates HNSW considers at query time.
-- Higher ef_search → better recall, slower queries.
-- Default is 40.

-- Baseline: ef_search = 40 (default)
SET hnsw.ef_search = 40;
EXPLAIN ANALYZE
SELECT id, title, content_halfvec <=> (
    SELECT content_halfvec FROM articles WHERE id = 1
) AS distance
FROM articles
WHERE content_halfvec IS NOT NULL
ORDER BY content_halfvec <=> (
    SELECT content_halfvec FROM articles WHERE id = 1
)
LIMIT 10;

-- ef_search = 100
SET hnsw.ef_search = 100;
EXPLAIN ANALYZE
SELECT id, title, content_halfvec <=> (
    SELECT content_halfvec FROM articles WHERE id = 1
) AS distance
FROM articles
WHERE content_halfvec IS NOT NULL
ORDER BY content_halfvec <=> (
    SELECT content_halfvec FROM articles WHERE id = 1
)
LIMIT 10;

-- ef_search = 200
SET hnsw.ef_search = 200;
EXPLAIN ANALYZE
SELECT id, title, content_halfvec <=> (
    SELECT content_halfvec FROM articles WHERE id = 1
) AS distance
FROM articles
WHERE content_halfvec IS NOT NULL
ORDER BY content_halfvec <=> (
    SELECT content_halfvec FROM articles WHERE id = 1
)
LIMIT 10;

-- ef_search = 400
SET hnsw.ef_search = 400;
EXPLAIN ANALYZE
SELECT id, title, content_halfvec <=> (
    SELECT content_halfvec FROM articles WHERE id = 1
) AS distance
FROM articles
WHERE content_halfvec IS NOT NULL
ORDER BY content_halfvec <=> (
    SELECT content_halfvec FROM articles WHERE id = 1
)
LIMIT 10;

-- Reset to default
RESET hnsw.ef_search;

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. Parallel index build (PG 18)
-- ─────────────────────────────────────────────────────────────────────────────
-- pgvector 0.8+ supports parallel HNSW builds.
-- Check current settings:
SHOW max_parallel_maintenance_workers;
SHOW maintenance_work_mem;

-- Rebuild with parallel workers to observe build time difference
DROP INDEX IF EXISTS idx_content_hnsw_halfvec;

\timing on

-- Workers are controlled by max_parallel_maintenance_workers (set in docker-compose)
CREATE INDEX idx_content_hnsw_halfvec
ON articles USING hnsw (content_halfvec halfvec_cosine_ops)
WITH (m = 16, ef_construction = 128);

\timing off
