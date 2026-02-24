-- =============================================================================
-- Lab 06 — 01_create_schema.sql
-- Articles table with all column types needed for index experiments
-- =============================================================================

-- Main articles table
-- Dense vectors stored as vector(3072) — native type
-- Half-precision copies stored as halfvec(3072) — for HNSW (2000-dim limit on vector)
-- Binary quantized copy stored as bit(3072) — for ultra-compact search
-- Sparse vectors stored as sparsevec(30522) — SPLADE vocabulary size
CREATE TABLE IF NOT EXISTS articles (
    id          INTEGER PRIMARY KEY,
    url         TEXT,
    title       TEXT,
    content     TEXT,

    -- Dense embeddings (text-embedding-3-large, 3072 dimensions)
    title_vector    vector(3072),
    content_vector  vector(3072),

    -- Half-precision copies (populated via CAST after dense embedding)
    title_halfvec   halfvec(3072),
    content_halfvec halfvec(3072),

    -- Binary quantized (populated via binary_quantize() after dense embedding)
    content_bq      bit(3072),

    -- Sparse embeddings (SPLADE, vocab size 30522)
    title_sparse    sparsevec(30522),
    content_sparse  sparsevec(30522),

    -- Metadata for filtering demos
    category    TEXT,
    word_count  INTEGER
);

-- B-tree indexes on metadata columns (for filtered vector search demos)
CREATE INDEX IF NOT EXISTS idx_articles_category   ON articles (category);
CREATE INDEX IF NOT EXISTS idx_articles_word_count ON articles (word_count);

-- NOTE: No vector indexes are created here.
-- Building indexes IS the lab exercise — see scripts 02 through 08.
