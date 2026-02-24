-- =============================================================================
-- Lab 06 — 06_halfvec_quantization.sql
-- Storage optimization: halfvec, binary_quantize, subvector, re-ranking
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. CAST vector → halfvec → bit
-- ─────────────────────────────────────────────────────────────────────────────
-- These columns were populated by the Python pipeline, but here's how:

-- vector(3072) → halfvec(3072): reduces storage from 4 bytes to 2 bytes per dim
-- UPDATE articles SET content_halfvec = content_vector::halfvec WHERE content_vector IS NOT NULL;

-- vector(3072) → bit(3072): reduces storage from 4 bytes to 1 bit per dim
-- UPDATE articles SET content_bq = binary_quantize(content_vector)::bit(3072) WHERE content_vector IS NOT NULL;

-- Verify population
SELECT
    COUNT(*) AS total,
    COUNT(content_vector) AS dense,
    COUNT(content_halfvec) AS halfvec,
    COUNT(content_bq) AS binary_quantized
FROM articles;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Storage math
-- ─────────────────────────────────────────────────────────────────────────────
-- For 3072 dimensions:
--   vector(3072):  3072 × 4 bytes = 12,288 bytes per row (~12 KB)
--   halfvec(3072): 3072 × 2 bytes =  6,144 bytes per row (~6 KB)
--   bit(3072):     3072 / 8       =    384 bytes per row (~0.4 KB)
--
-- For 25,000 articles:
--   vector:   ~300 MB
--   halfvec:  ~150 MB
--   bit:       ~10 MB

-- Actual column storage (approximate via pg_column_size on a sample)
SELECT
    pg_size_pretty(AVG(pg_column_size(content_vector))::bigint) AS avg_vector_size,
    pg_size_pretty(AVG(pg_column_size(content_halfvec))::bigint) AS avg_halfvec_size,
    pg_size_pretty(AVG(pg_column_size(content_bq))::bigint) AS avg_bq_size
FROM articles
WHERE content_vector IS NOT NULL;

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Index size comparison
-- ─────────────────────────────────────────────────────────────────────────────
-- Build HNSW indexes on halfvec and binary quantized columns

-- HNSW on halfvec (already exists from script 02)
-- CREATE INDEX idx_content_hnsw_halfvec
-- ON articles USING hnsw (content_halfvec halfvec_cosine_ops)
-- WITH (m = 16, ef_construction = 128);

-- HNSW on binary quantized (Hamming distance)
DROP INDEX IF EXISTS idx_content_hnsw_bq;

\timing on

CREATE INDEX idx_content_hnsw_bq
ON articles USING hnsw (content_bq bit_hamming_ops)
WITH (m = 16, ef_construction = 128);

\timing off

-- Compare all index sizes
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS size,
    pg_relation_size(indexname::regclass) AS size_bytes
FROM pg_indexes
WHERE tablename = 'articles'
AND indexname LIKE '%content%'
AND (indexname LIKE '%hnsw%' OR indexname LIKE '%ivfflat%' OR indexname LIKE '%diskann%')
ORDER BY pg_relation_size(indexname::regclass);

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. Binary quantize + re-ranking pattern
-- ─────────────────────────────────────────────────────────────────────────────
-- Strategy: Fast coarse search on binary → re-rank top candidates with full vectors
-- This gives binary speed with near-full-precision quality.

-- Step 1: Coarse search using binary quantized index (Hamming distance)
-- Step 2: Re-rank top candidates using exact cosine on original vectors

EXPLAIN ANALYZE
WITH coarse AS (
    SELECT id, title, content_vector
    FROM articles
    WHERE content_bq IS NOT NULL
    ORDER BY content_bq <~> (
        SELECT binary_quantize(content_vector)::bit(3072) FROM articles WHERE id = 1
    )
    LIMIT 100  -- over-fetch for re-ranking
)
SELECT id, title,
       content_vector <=> (SELECT content_vector FROM articles WHERE id = 1) AS exact_distance
FROM coarse
ORDER BY content_vector <=> (SELECT content_vector FROM articles WHERE id = 1)
LIMIT 10;

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. l2_normalize() — unit vectors for cosine optimization
-- ─────────────────────────────────────────────────────────────────────────────
-- If vectors are pre-normalized, inner product (<#>) equals cosine distance.
-- This lets you use ip_ops indexes instead of cosine_ops.

-- Check if vectors are already normalized (OpenAI embeddings usually are)
SELECT
    id,
    l2_norm(content_vector) AS norm
FROM articles
WHERE content_vector IS NOT NULL
LIMIT 5;

-- Normalize example (not needed for OpenAI, but useful for custom embeddings)
-- UPDATE articles SET content_vector = l2_normalize(content_vector)
-- WHERE content_vector IS NOT NULL;

-- ─────────────────────────────────────────────────────────────────────────────
-- 6. subvector() — dimension reduction
-- ─────────────────────────────────────────────────────────────────────────────
-- Extract first N dimensions for faster approximate search.
-- text-embedding-3-large supports Matryoshka representation:
-- first 1024 dims carry most of the information.

-- subvector(vector, offset, dimensions) — 1-indexed offset
SELECT
    a.id AS id_a,
    b.id AS id_b,
    a.content_vector <=> b.content_vector AS full_3072_distance,
    subvector(a.content_vector, 1, 1024)::vector(1024) <=>
    subvector(b.content_vector, 1, 1024)::vector(1024) AS sub_1024_distance
FROM articles a, articles b
WHERE a.id = 1 AND b.id = 2
AND a.content_vector IS NOT NULL
AND b.content_vector IS NOT NULL;

-- Cleanup
DROP INDEX IF EXISTS idx_content_hnsw_bq;
