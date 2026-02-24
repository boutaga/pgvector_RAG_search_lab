-- =============================================================================
-- Lab 06 — 03_ivfflat_indexes.sql
-- IVFFlat deep dive: dimension limit (same as HNSW), probes tuning, optimizer
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. IVFFlat on vector(3072) → FAIL (also 2000-dimension limit)
-- ─────────────────────────────────────────────────────────────────────────────
-- Like HNSW, IVFFlat has a 2000-dimension limit for the vector type.
-- Same workaround: use halfvec(3072) (limit 4000).

-- Uncomment to demonstrate the error:
-- CREATE INDEX idx_content_ivfflat_fail
-- ON articles USING ivfflat (content_vector vector_cosine_ops)
-- WITH (lists = 150);
-- ERROR:  column cannot have more than 2000 dimensions for ivfflat index

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. IVFFlat on halfvec(3072) → SUCCESS
-- ─────────────────────────────────────────────────────────────────────────────
-- lists = number of Voronoi cells. Rule of thumb: sqrt(n) for < 1M rows.
-- For 25,000 articles: sqrt(25000) ≈ 158, we use 150.

DROP INDEX IF EXISTS idx_content_ivfflat;

\timing on

CREATE INDEX idx_content_ivfflat
ON articles USING ivfflat (content_halfvec halfvec_cosine_ops)
WITH (lists = 150);

\timing off

SELECT pg_size_pretty(pg_relation_size('idx_content_ivfflat')) AS ivfflat_size;

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Probes sweep: 1, 5, 10, 20, 50
-- ─────────────────────────────────────────────────────────────────────────────
-- probes = how many Voronoi cells to search. Higher = better recall, slower.
-- Default is 1 (fast but low recall).

-- probes = 1 (default — fast, low recall)
SET ivfflat.probes = 1;
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

-- probes = 5
SET ivfflat.probes = 5;
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

-- probes = 10
SET ivfflat.probes = 10;
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

-- probes = 20
SET ivfflat.probes = 20;
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

-- probes = 50
SET ivfflat.probes = 50;
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

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. Optimizer flip-flop: SeqScan vs IndexScan
-- ─────────────────────────────────────────────────────────────────────────────
-- At high probes values, the planner may decide a sequential scan is cheaper.
-- Watch for the switch point:

SET ivfflat.probes = 100;
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

-- Force index scan to compare:
SET enable_seqscan = off;
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

SET enable_seqscan = on;

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. SET LOCAL pattern for production
-- ─────────────────────────────────────────────────────────────────────────────
-- In production, use SET LOCAL inside a transaction so the setting reverts
-- automatically after COMMIT/ROLLBACK. No global state leakage.

BEGIN;
    SET LOCAL ivfflat.probes = 20;

    SELECT id, title, content_halfvec <=> (
        SELECT content_halfvec FROM articles WHERE id = 1
    ) AS distance
    FROM articles
    WHERE content_halfvec IS NOT NULL
    ORDER BY content_halfvec <=> (
        SELECT content_halfvec FROM articles WHERE id = 1
    )
    LIMIT 10;
COMMIT;

-- Verify probes reverted to default
SHOW ivfflat.probes;

-- Reset
RESET ivfflat.probes;
