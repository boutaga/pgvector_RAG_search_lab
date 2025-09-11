-- PostgreSQL pgvector RAG Lab Setup
-- This script sets up the complete database schema

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create the main articles table with all search capabilities
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT,
    content TEXT NOT NULL,
    
    -- Vector embeddings (pgvector)
    title_vector vector(1536),
    content_vector vector(1536),
    
    -- Sparse embeddings (pgvectorscale sparsevec)
    title_sparse sparsevec(30522),
    content_sparse sparsevec(30522),
    
    -- Full-text search vectors
    content_tsv tsvector,
    title_content_tsvector tsvector,
    
    -- Metadata
    vector_id INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Advanced full-text setup with weighted ranking (titles priority over content)
CREATE OR REPLACE FUNCTION update_article_tsvector() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    NEW.title_content_tsvector := 
        setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') || 
        setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'B');
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER articles_tsvector_update 
    BEFORE INSERT OR UPDATE ON articles 
    FOR EACH ROW EXECUTE FUNCTION update_article_tsvector();

-- Performance indexes
-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_articles_content_tsv 
    ON articles USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS idx_articles_title_content_tsvector 
    ON articles USING GIN (title_content_tsvector);

-- Traditional indexes for exact matches
CREATE INDEX IF NOT EXISTS idx_articles_title_gin 
    ON articles USING GIN (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_articles_id 
    ON articles USING btree (id);

-- Note: Vector indexes will be created after embeddings are populated
-- Dense vector indexes (HNSW for speed) - uncomment after populating embeddings
/*
CREATE INDEX IF NOT EXISTS idx_articles_title_vec_hnsw 
    ON articles USING hnsw (title_vector vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_articles_content_vec_hnsw 
    ON articles USING hnsw (content_vector vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);

-- Sparse vector indexes (pgvectorscale) - uncomment after populating sparse embeddings
CREATE INDEX IF NOT EXISTS idx_articles_title_sparse_hnsw 
    ON articles USING hnsw (title_sparse sparsevec_ip_ops) 
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_articles_content_sparse_hnsw 
    ON articles USING hnsw (content_sparse sparsevec_ip_ops) 
    WITH (m = 16, ef_construction = 64);

-- DiskANN indexes for large-scale performance (pgvectorscale required)
CREATE INDEX IF NOT EXISTS idx_articles_content_diskann 
    ON articles USING diskann (content_vector vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_articles_content_sparse_diskann 
    ON articles USING diskann (content_sparse sparsevec_ip_ops);
*/

-- Enhanced performance metrics table
CREATE TABLE IF NOT EXISTS search_metrics (
    log_id SERIAL PRIMARY KEY,
    query_id TEXT,                    -- Hash of the query for grouping
    description TEXT,                 -- First 20 chars of query for readability
    query_time TIMESTAMPTZ DEFAULT NOW(),
    mode TEXT,                        -- Search method used
    top_score REAL,                   -- Best similarity score/distance
    token_usage INTEGER,              -- Total tokens consumed
    precision REAL DEFAULT 0,        -- Proportion of relevant results
    embedding_ms REAL,                -- Embedding generation time
    db_ms REAL,                       -- Database query execution time
    llm_ms REAL,                      -- LLM response generation time
    total_ms REAL,                    -- Total end-to-end latency
    
    -- Extended metrics for advanced analysis
    context_tokens INTEGER,           -- Tokens used in context
    output_tokens INTEGER,            -- Tokens generated in response
    chunk_count INTEGER,              -- Number of chunks retrieved
    rerank_ms REAL,                   -- Reranking execution time
    index_used TEXT,                  -- Which index was used by query planner
    buffer_hits INTEGER,              -- PostgreSQL buffer cache hits
    disk_reads INTEGER                -- Disk reads for performance analysis
);

-- Create index for metrics analysis
CREATE INDEX IF NOT EXISTS idx_search_metrics_query_time 
    ON search_metrics(query_time DESC);
CREATE INDEX IF NOT EXISTS idx_search_metrics_mode 
    ON search_metrics(mode);
CREATE INDEX IF NOT EXISTS idx_search_metrics_query_id 
    ON search_metrics(query_id);

-- Metric descriptions for UI tooltips
CREATE TABLE IF NOT EXISTS metric_descriptions (
    metric_name TEXT PRIMARY KEY,
    description TEXT NOT NULL
);

INSERT INTO metric_descriptions(metric_name, description) VALUES
    ('query_id', 'Short hash representing the query text'),
    ('description', 'First 20 characters of the query'),
    ('query_time', 'Timestamp when the query was executed'),
    ('mode', 'Search mode used for this query'),
    ('top_score', 'Best similarity distance or score'),
    ('token_usage', 'Total tokens used in the LLM call'),
    ('precision', 'Proportion of relevant results'),
    ('embedding_ms', 'Milliseconds spent generating the embedding'),
    ('db_ms', 'Milliseconds spent executing the database search'),
    ('llm_ms', 'Milliseconds spent generating the LLM answer'),
    ('total_ms', 'Total execution time in milliseconds'),
    ('context_tokens', 'Number of tokens used in the context'),
    ('output_tokens', 'Number of tokens generated in the response'),
    ('chunk_count', 'Number of document chunks retrieved'),
    ('rerank_ms', 'Time spent on result reranking'),
    ('index_used', 'PostgreSQL index used by the query planner'),
    ('buffer_hits', 'Number of buffer cache hits'),
    ('disk_reads', 'Number of disk reads performed')
ON CONFLICT (metric_name) DO NOTHING;

-- Performance monitoring views
CREATE OR REPLACE VIEW search_performance AS
SELECT 
    mode,
    COUNT(*) as query_count,
    AVG(total_ms) as avg_latency,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_ms) as p95_latency,
    AVG(CASE WHEN top_score > 0.8 THEN 1.0 ELSE 0.0 END) as high_confidence_rate
FROM search_metrics 
WHERE query_time >= NOW() - INTERVAL '24 hours'
GROUP BY mode;

-- Movie/Netflix specific tables (if using movie dataset)
CREATE TABLE IF NOT EXISTS film (
    film_id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    release_year INTEGER,
    language_id INTEGER,
    rental_duration INTEGER,
    rental_rate NUMERIC(4,2),
    length INTEGER,
    replacement_cost NUMERIC(5,2),
    rating TEXT,
    last_update TIMESTAMP DEFAULT NOW(),
    special_features TEXT[],
    fulltext tsvector,
    
    -- Embeddings
    embedding vector(1536),
    sparse_embedding sparsevec(30522)
);

CREATE TABLE IF NOT EXISTS netflix_shows (
    show_id TEXT PRIMARY KEY,
    type TEXT,
    title TEXT NOT NULL,
    director TEXT,
    cast_list TEXT,
    country TEXT,
    date_added DATE,
    release_year INTEGER,
    rating TEXT,
    duration TEXT,
    listed_in TEXT,
    description TEXT,
    
    -- Embeddings
    embedding vector(1536),
    sparse_embedding sparsevec(30522)
);

-- Customer and rental tables for recommendations
CREATE TABLE IF NOT EXISTS customer (
    customer_id SERIAL PRIMARY KEY,
    store_id INTEGER,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    address_id INTEGER,
    activebool BOOLEAN DEFAULT true,
    create_date DATE DEFAULT CURRENT_DATE,
    last_update TIMESTAMP DEFAULT NOW(),
    active INTEGER
);

CREATE TABLE IF NOT EXISTS rental (
    rental_id SERIAL PRIMARY KEY,
    rental_date TIMESTAMP NOT NULL,
    inventory_id INTEGER NOT NULL,
    customer_id INTEGER REFERENCES customer(customer_id),
    return_date TIMESTAMP,
    staff_id INTEGER,
    last_update TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS inventory (
    inventory_id SERIAL PRIMARY KEY,
    film_id INTEGER REFERENCES film(film_id),
    store_id INTEGER,
    last_update TIMESTAMP DEFAULT NOW()
);

-- Indexes for movie/Netflix tables
CREATE INDEX IF NOT EXISTS idx_film_title ON film(title);
CREATE INDEX IF NOT EXISTS idx_netflix_title ON netflix_shows(title);
CREATE INDEX IF NOT EXISTS idx_rental_customer ON rental(customer_id);
CREATE INDEX IF NOT EXISTS idx_rental_date ON rental(rental_date);
CREATE INDEX IF NOT EXISTS idx_inventory_film ON inventory(film_id);

-- Function to get similar movies
CREATE OR REPLACE FUNCTION get_similar_movies(
    query_embedding vector(1536),
    limit_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    film_id INTEGER,
    title TEXT,
    description TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        f.film_id,
        f.title,
        f.description,
        1 - (f.embedding <=> query_embedding) as similarity
    FROM film f
    WHERE f.embedding IS NOT NULL
    ORDER BY f.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Evaluation support tables
CREATE TABLE IF NOT EXISTS evaluation_test_cases (
    test_id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    expected_doc_ids INTEGER[],
    expected_answer TEXT,
    category TEXT,
    difficulty TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    result_id SERIAL PRIMARY KEY,
    test_id INTEGER REFERENCES evaluation_test_cases(test_id),
    search_method TEXT,
    k_value INTEGER,
    precision_at_k REAL,
    recall_at_k REAL,
    f1_score REAL,
    mrr REAL,
    ndcg REAL,
    answer_relevance REAL,
    answer_faithfulness REAL,
    latency_ms REAL,
    token_cost REAL,
    evaluated_at TIMESTAMP DEFAULT NOW()
);

-- Benchmark results table
CREATE TABLE IF NOT EXISTS benchmark_results (
    benchmark_id SERIAL PRIMARY KEY,
    benchmark_name TEXT,
    operation TEXT,
    duration_ms REAL,
    throughput_ops REAL,
    cpu_percent REAL,
    memory_mb REAL,
    success BOOLEAN,
    error_message TEXT,
    metadata JSONB,
    executed_at TIMESTAMP DEFAULT NOW()
);

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_app_user;

SELECT 'PostgreSQL pgvector RAG Lab setup completed successfully!' as status;