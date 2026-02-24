-- =============================================================================
-- Lab 06 — 05_iterative_scans.sql
-- Iterative index scans: THE star feature for vector + WHERE combinations
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- The Problem
-- ─────────────────────────────────────────────────────────────────────────────
-- Vector search returns the K nearest neighbors, then PostgreSQL applies WHERE.
-- If your filter is selective, you get fewer results than requested.
--
-- Example: Ask for 10 results about "Science" articles, but the index only
-- returns 10 nearest neighbors total — maybe only 2 are "Science".
--
-- Before iterative scans, you had two bad options:
--   1. Over-fetch (LIMIT 1000) and hope enough rows match → wasteful, unreliable
--   2. Sequential scan → slow on large tables

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Demonstrate the truncation problem
-- ─────────────────────────────────────────────────────────────────────────────

-- First, check category distribution
SELECT category, COUNT(*) AS cnt
FROM articles
WHERE category IS NOT NULL
GROUP BY category
ORDER BY cnt DESC
LIMIT 20;

-- Standard vector search + WHERE: may return fewer than 10 rows
SET hnsw.ef_search = 40;
EXPLAIN ANALYZE
SELECT id, title, category,
       content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1) AS distance
FROM articles
WHERE category = 'Science'
ORDER BY content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1)
LIMIT 10;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Iterative scan — relaxed_order mode (HNSW)
-- ─────────────────────────────────────────────────────────────────────────────
-- hnsw.iterative_scan = relaxed_order
-- The index keeps fetching more candidates until the filter is satisfied.
-- "relaxed_order" means results are approximately ordered (not strict ranking).

SET hnsw.iterative_scan = relaxed_order;
SET hnsw.ef_search = 40;

EXPLAIN ANALYZE
SELECT id, title, category,
       content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1) AS distance
FROM articles
WHERE category = 'Science'
ORDER BY content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1)
LIMIT 10;

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Iterative scan — strict_order mode (HNSW)
-- ─────────────────────────────────────────────────────────────────────────────
-- strict_order guarantees exact distance ordering.
-- Slightly slower than relaxed_order but results are perfectly ranked.

SET hnsw.iterative_scan = strict_order;

EXPLAIN ANALYZE
SELECT id, title, category,
       content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1) AS distance
FROM articles
WHERE category = 'Science'
ORDER BY content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1)
LIMIT 10;

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. max_scan_tuples — safety valve
-- ─────────────────────────────────────────────────────────────────────────────
-- Controls how many index tuples the iterative scan examines before giving up.
-- Prevents runaway scans on very selective filters.
-- Default: 20000. Set to 0 for unlimited.

SET hnsw.iterative_scan = relaxed_order;

-- Restrictive limit: may return fewer rows but fast
SET hnsw.max_scan_tuples = 500;
EXPLAIN ANALYZE
SELECT id, title, category,
       content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1) AS distance
FROM articles
WHERE category = 'Science'
ORDER BY content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1)
LIMIT 10;

-- Generous limit: should find all 10 rows
SET hnsw.max_scan_tuples = 20000;
EXPLAIN ANALYZE
SELECT id, title, category,
       content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1) AS distance
FROM articles
WHERE category = 'Science'
ORDER BY content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1)
LIMIT 10;

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. Iterative scan with IVFFlat
-- ─────────────────────────────────────────────────────────────────────────────
-- Same concept, different GUC prefix.

SET ivfflat.iterative_scan = relaxed_order;
SET ivfflat.probes = 10;

EXPLAIN ANALYZE
SELECT id, title, category,
       content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1) AS distance
FROM articles
WHERE category = 'Science'
ORDER BY content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1)
LIMIT 10;

-- ─────────────────────────────────────────────────────────────────────────────
-- 6. Multi-filter + vector search combo
-- ─────────────────────────────────────────────────────────────────────────────
-- Iterative scans shine when combining multiple filters with vector search.

SET hnsw.iterative_scan = relaxed_order;
SET hnsw.ef_search = 100;

EXPLAIN ANALYZE
SELECT id, title, category, word_count,
       content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1) AS distance
FROM articles
WHERE category = 'Science'
  AND word_count > 500
ORDER BY content_halfvec <=> (SELECT content_halfvec FROM articles WHERE id = 1)
LIMIT 10;

-- Reset all
RESET hnsw.iterative_scan;
RESET hnsw.max_scan_tuples;
RESET hnsw.ef_search;
RESET ivfflat.iterative_scan;
RESET ivfflat.probes;
