-- =============================================================================
-- Lab 06 — 07_operators_and_filtering.sql
-- Distance operators, operator classes, sargable rewrites, partial indexes
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. The four distance operators and their operator classes
-- ─────────────────────────────────────────────────────────────────────────────
--
-- Operator  | Name               | Operator Class      | Lower is better?
-- ----------|--------------------|--------------------|------------------
-- <=>       | Cosine distance    | vector_cosine_ops  | Yes (0 = identical)
-- <->       | L2 (Euclidean)     | vector_l2_ops      | Yes (0 = identical)
-- <#>       | Negative inner prod| vector_ip_ops      | Yes (more negative = closer)
-- <+>       | L1 (Manhattan)     | vector_l1_ops      | Yes (0 = identical)
--
-- For halfvec: halfvec_cosine_ops, halfvec_l2_ops, halfvec_ip_ops
-- For bit:     bit_hamming_ops, bit_jaccard_ops
-- For sparsevec: sparsevec_cosine_ops, sparsevec_l2_ops

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Wrong operator → index not used
-- ─────────────────────────────────────────────────────────────────────────────
-- The HNSW index on content_halfvec uses halfvec_cosine_ops.
-- Using the L2 operator (<->) will NOT use this index.

-- This SHOULD use the HNSW cosine index:
EXPLAIN (COSTS OFF)
SELECT id, title
FROM articles
ORDER BY content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1)
LIMIT 10;

-- This will NOT use the cosine index (wrong operator):
EXPLAIN (COSTS OFF)
SELECT id, title
FROM articles
ORDER BY content_halfvec <-> (SELECT content_halfvec FROM articles WHERE id = 1)
LIMIT 10;

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Sargable rewrite: cross-join → scalar subquery
-- ─────────────────────────────────────────────────────────────────────────────
-- A common pattern that PREVENTS index use: self-join to get the probe vector.
-- The optimizer cannot push the ORDER BY into the index scan.

-- BAD: Cross-join pattern — forces sequential scan
EXPLAIN (COSTS OFF)
SELECT a.id, a.title
FROM articles a, articles b
WHERE b.id = 1
ORDER BY a.content_halfvec <=> b.content_halfvec
LIMIT 10;

-- GOOD: Scalar subquery pattern — enables index scan
EXPLAIN (COSTS OFF)
SELECT id, title
FROM articles
ORDER BY content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1)
LIMIT 10;

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. Passing a literal vector (application pattern)
-- ─────────────────────────────────────────────────────────────────────────────
-- In production, your application passes the query embedding as a parameter.
-- This is always sargable because there's no subquery ambiguity.

-- Simulated application query (using a zero vector as placeholder):
EXPLAIN (COSTS OFF)
SELECT id, title
FROM articles
ORDER BY content_halfvec <=> $1::halfvec(3072)
LIMIT 10;

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. B-tree + vector combo (composite filtering)
-- ─────────────────────────────────────────────────────────────────────────────
-- PostgreSQL can combine a B-tree filter with a vector index scan.
-- The iterative scan feature (script 05) handles this elegantly.

-- Without iterative scan: planner chooses between strategies
RESET hnsw.iterative_scan;

EXPLAIN ANALYZE
SELECT id, title, category,
       content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1) AS distance
FROM articles
WHERE category = 'Science'
ORDER BY content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1)
LIMIT 10;

-- With iterative scan: guaranteed correct result count
SET hnsw.iterative_scan = relaxed_order;

EXPLAIN ANALYZE
SELECT id, title, category,
       content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1) AS distance
FROM articles
WHERE category = 'Science'
ORDER BY content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1)
LIMIT 10;

RESET hnsw.iterative_scan;

-- ─────────────────────────────────────────────────────────────────────────────
-- 6. Partial indexes — index only what you need
-- ─────────────────────────────────────────────────────────────────────────────
-- If you always filter on a specific category, a partial index is smaller
-- and faster than indexing the full table.

DROP INDEX IF EXISTS idx_content_hnsw_science;

\timing on

CREATE INDEX idx_content_hnsw_science
ON articles USING hnsw (content_halfvec halfvec_cosine_ops)
WITH (m = 16, ef_construction = 128)
WHERE category = 'Science';

\timing off

-- Check size vs full index
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
FROM pg_indexes
WHERE tablename = 'articles'
AND indexname IN ('idx_content_hnsw_halfvec', 'idx_content_hnsw_science')
ORDER BY indexname;

-- Query using the partial index (WHERE must match the index predicate)
EXPLAIN ANALYZE
SELECT id, title,
       content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1) AS distance
FROM articles
WHERE category = 'Science'
ORDER BY content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1)
LIMIT 10;

-- Cleanup
DROP INDEX IF EXISTS idx_content_hnsw_science;
