-- Migration Script: Update to text-embedding-3-large (3072 dimensions)
-- Run this script to update existing databases from 1536 to 3072 dimensions
-- IMPORTANT: Backup your databases before running this script!

-- ============================================================================
-- STEP 1: ADD NEW COLUMNS FOR 3072-DIMENSIONAL VECTORS
-- ============================================================================

-- For Wikipedia Database
-- ------------------------
\c wikipedia;

-- Add new columns to articles table
ALTER TABLE articles
    ADD COLUMN IF NOT EXISTS title_vector_3072 vector(3072),
    ADD COLUMN IF NOT EXISTS content_vector_3072 vector(3072);

-- Check if columns were added successfully
SELECT column_name, data_type, character_maximum_length
FROM information_schema.columns
WHERE table_name = 'articles'
AND column_name IN ('title_vector', 'content_vector', 'title_vector_3072', 'content_vector_3072');

-- For Movie/Netflix Database
-- ---------------------------
\c dvdrental;

-- Add new columns to film table
ALTER TABLE film
    ADD COLUMN IF NOT EXISTS embedding_3072 vector(3072);

-- Add new columns to netflix_shows table
ALTER TABLE netflix_shows
    ADD COLUMN IF NOT EXISTS embedding_3072 vector(3072);

-- Check if columns were added successfully
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_name IN ('film', 'netflix_shows')
AND column_name IN ('embedding', 'embedding_3072');

-- ============================================================================
-- STEP 2: DROP ALL EXISTING VECTOR INDEXES
-- ============================================================================

-- Wikipedia Database Indexes
\c wikipedia;

-- Drop HNSW indexes
DROP INDEX IF EXISTS articles_title_vector_idx;
DROP INDEX IF EXISTS articles_content_vector_idx;
DROP INDEX IF EXISTS idx_articles_title_vector_hnsw;
DROP INDEX IF EXISTS idx_articles_content_vector_hnsw;

-- Drop IVFFlat indexes
DROP INDEX IF EXISTS articles_title_vector_ivfflat_idx;
DROP INDEX IF EXISTS articles_content_vector_ivfflat_idx;

-- Drop any cosine similarity indexes
DROP INDEX IF EXISTS articles_title_vector_cosine_idx;
DROP INDEX IF EXISTS articles_content_vector_cosine_idx;

-- Movie/Netflix Database Indexes
\c dvdrental;

-- Drop film indexes
DROP INDEX IF EXISTS film_embedding_idx;
DROP INDEX IF EXISTS film_embedding_cosine_idx;
DROP INDEX IF EXISTS film_embedding_ivfflat_idx;
DROP INDEX IF EXISTS film_embedding_ivfflat_cosine_idx;
DROP INDEX IF EXISTS film_embedding_hnsw_idx;

-- Drop netflix_shows indexes
DROP INDEX IF EXISTS netflix_shows_embedding_idx;
DROP INDEX IF EXISTS netflix_shows_embedding_cosine_idx;
DROP INDEX IF EXISTS netflix_embedding_ivfflat_cosine_idx;
DROP INDEX IF EXISTS netflix_embedding_cosine_idx;
DROP INDEX IF EXISTS netflix_embedding_idx;
DROP INDEX IF EXISTS netflix_shows_embedding_hnsw_idx;

-- ============================================================================
-- STEP 3: AFTER REGENERATING EMBEDDINGS - MIGRATE DATA
-- ============================================================================
-- Run these commands AFTER you've regenerated embeddings with the new model

-- Wikipedia Database
\c wikipedia;

-- Copy new embeddings to original columns (if you've populated the _3072 columns)
-- UPDATE articles SET
--     title_vector = title_vector_3072,
--     content_vector = content_vector_3072
-- WHERE title_vector_3072 IS NOT NULL;

-- OR if you're updating the original columns directly, skip the above

-- Movie/Netflix Database
\c dvdrental;

-- Copy new embeddings to original columns (if you've populated the _3072 columns)
-- UPDATE film SET embedding = embedding_3072 WHERE embedding_3072 IS NOT NULL;
-- UPDATE netflix_shows SET embedding = embedding_3072 WHERE embedding_3072 IS NOT NULL;

-- ============================================================================
-- STEP 4: CREATE NEW INDEXES FOR 3072 DIMENSIONS
-- ============================================================================

-- Wikipedia Database - HNSW Indexes (Recommended for accuracy)
\c wikipedia;

-- Create HNSW indexes for title vectors
CREATE INDEX idx_articles_title_vector_hnsw
    ON articles USING hnsw (title_vector vector_l2_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_articles_title_vector_cosine_hnsw
    ON articles USING hnsw (title_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Create HNSW indexes for content vectors
CREATE INDEX idx_articles_content_vector_hnsw
    ON articles USING hnsw (content_vector vector_l2_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_articles_content_vector_cosine_hnsw
    ON articles USING hnsw (content_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat Indexes (Faster build, slightly less accurate)
-- Uncomment if you prefer IVFFlat over HNSW

-- CREATE INDEX idx_articles_title_vector_ivfflat
--     ON articles USING ivfflat (title_vector vector_l2_ops)
--     WITH (lists = 100);

-- CREATE INDEX idx_articles_content_vector_ivfflat
--     ON articles USING ivfflat (content_vector vector_l2_ops)
--     WITH (lists = 100);

-- Movie/Netflix Database - HNSW Indexes
\c dvdrental;

-- Film table indexes
CREATE INDEX idx_film_embedding_hnsw
    ON film USING hnsw (embedding vector_l2_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_film_embedding_cosine_hnsw
    ON film USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Netflix shows indexes
CREATE INDEX idx_netflix_shows_embedding_hnsw
    ON netflix_shows USING hnsw (embedding vector_l2_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_netflix_shows_embedding_cosine_hnsw
    ON netflix_shows USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat Indexes
-- Uncomment if you prefer IVFFlat over HNSW

-- CREATE INDEX idx_film_embedding_ivfflat
--     ON film USING ivfflat (embedding vector_l2_ops)
--     WITH (lists = 100);

-- CREATE INDEX idx_netflix_shows_embedding_ivfflat
--     ON netflix_shows USING ivfflat (embedding vector_l2_ops)
--     WITH (lists = 100);

-- ============================================================================
-- STEP 5: UPDATE FUNCTIONS FOR NEW DIMENSIONS
-- ============================================================================

\c dvdrental;

-- Drop and recreate the get_similar_movies function with new dimensions
DROP FUNCTION IF EXISTS get_similar_movies(vector(1536), integer);
DROP FUNCTION IF EXISTS get_similar_movies(vector(3072), integer);

CREATE OR REPLACE FUNCTION get_similar_movies(
    query_embedding vector(3072),
    limit_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    film_id INTEGER,
    title VARCHAR(255),
    description TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.film_id,
        f.title,
        f.description,
        1 - (f.embedding <=> query_embedding) AS similarity
    FROM film f
    WHERE f.embedding IS NOT NULL
    ORDER BY f.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Similar function for Netflix shows
DROP FUNCTION IF EXISTS get_similar_netflix_shows(vector(1536), integer);
DROP FUNCTION IF EXISTS get_similar_netflix_shows(vector(3072), integer);

CREATE OR REPLACE FUNCTION get_similar_netflix_shows(
    query_embedding vector(3072),
    limit_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    show_id VARCHAR(20),
    title VARCHAR(500),
    description TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        n.show_id,
        n.title,
        n.description,
        1 - (n.embedding <=> query_embedding) AS similarity
    FROM netflix_shows n
    WHERE n.embedding IS NOT NULL
    ORDER BY n.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- STEP 6: CLEANUP (OPTIONAL - Run after verifying everything works)
-- ============================================================================

-- After confirming the migration is successful, you can drop the temporary columns

-- \c wikipedia;
-- ALTER TABLE articles
--     DROP COLUMN IF EXISTS title_vector_3072,
--     DROP COLUMN IF EXISTS content_vector_3072;

-- \c dvdrental;
-- ALTER TABLE film DROP COLUMN IF EXISTS embedding_3072;
-- ALTER TABLE netflix_shows DROP COLUMN IF EXISTS embedding_3072;

-- ============================================================================
-- STEP 7: VERIFY THE MIGRATION
-- ============================================================================

-- Check vector dimensions in Wikipedia database
\c wikipedia;

SELECT
    'articles' as table_name,
    COUNT(*) as total_rows,
    COUNT(title_vector) as rows_with_title_vector,
    COUNT(content_vector) as rows_with_content_vector,
    pg_column_size(title_vector) as title_vector_bytes,
    pg_column_size(content_vector) as content_vector_bytes
FROM articles
LIMIT 1;

-- Check vector dimensions in Movie database
\c dvdrental;

SELECT
    'film' as table_name,
    COUNT(*) as total_rows,
    COUNT(embedding) as rows_with_embedding,
    pg_column_size(embedding) as embedding_bytes
FROM film;

SELECT
    'netflix_shows' as table_name,
    COUNT(*) as total_rows,
    COUNT(embedding) as rows_with_embedding,
    pg_column_size(embedding) as embedding_bytes
FROM netflix_shows;

-- Check index status
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename IN ('articles', 'film', 'netflix_shows')
AND indexname LIKE '%vector%' OR indexname LIKE '%embedding%'
ORDER BY tablename, indexname;

-- ============================================================================
-- PERFORMANCE TUNING RECOMMENDATIONS
-- ============================================================================

-- For HNSW indexes, you can tune these parameters:
-- m: Maximum number of connections (default 16, range 2-100)
-- ef_construction: Size of dynamic list (default 64, range 4-1000)
-- Higher values = better recall but slower build time

-- For IVFFlat indexes, tune the lists parameter:
-- lists: Number of clusters (default 100)
-- Recommended: sqrt(number of rows) for optimal performance

-- Set search parameters for better query performance:
-- For HNSW:
SET hnsw.ef_search = 100;  -- Higher = better accuracy, slower search

-- For IVFFlat:
SET ivfflat.probes = 10;   -- Higher = better accuracy, slower search

-- ============================================================================
-- NOTES
-- ============================================================================
-- 1. This migration assumes you're updating from 1536 to 3072 dimensions
-- 2. Backup your data before running this script
-- 3. Regenerate embeddings using the updated Python scripts after Step 1
-- 4. The script uses HNSW indexes by default (better accuracy)
-- 5. IVFFlat alternatives are provided (faster build, slightly less accurate)
-- 6. Adjust index parameters based on your dataset size and performance needs
-- 7. Vector size: 3072 dimensions * 4 bytes = 12KB per vector (vs 6KB for 1536)