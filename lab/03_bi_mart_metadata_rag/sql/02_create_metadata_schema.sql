-- Create metadata catalog schema for storing database structure information
-- This schema will store enriched metadata with embeddings for RAG search

DROP SCHEMA IF EXISTS catalog CASCADE;
CREATE SCHEMA catalog;

-- Table-level metadata with embeddings
CREATE TABLE catalog.table_metadata (
    id SERIAL PRIMARY KEY,
    schema_name VARCHAR(100) NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    table_type VARCHAR(50), -- BASE TABLE, VIEW, MATERIALIZED VIEW
    row_count BIGINT,
    table_size_bytes BIGINT,
    description TEXT,
    table_comment TEXT,
    has_primary_key BOOLEAN,
    column_count INTEGER,
    foreign_key_count INTEGER,
    index_count INTEGER,
    metadata_text TEXT, -- Combined text for embedding
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(schema_name, table_name)
);

-- Column-level metadata with embeddings
CREATE TABLE catalog.column_metadata (
    id SERIAL PRIMARY KEY,
    schema_name VARCHAR(100) NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    column_name VARCHAR(100) NOT NULL,
    ordinal_position INTEGER,
    data_type VARCHAR(100),
    character_maximum_length INTEGER,
    numeric_precision INTEGER,
    numeric_scale INTEGER,
    is_nullable BOOLEAN,
    column_default TEXT,
    is_primary_key BOOLEAN,
    is_foreign_key BOOLEAN,
    is_unique BOOLEAN,
    referenced_schema VARCHAR(100),
    referenced_table VARCHAR(100),
    referenced_column VARCHAR(100),
    column_comment TEXT,
    -- Statistical information from pg_stats
    n_distinct NUMERIC,
    null_fraction NUMERIC,
    avg_width INTEGER,
    correlation NUMERIC,
    most_common_values TEXT[],
    most_common_freqs NUMERIC[],
    histogram_bounds TEXT[],
    -- Combined metadata for embedding
    metadata_text TEXT,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(schema_name, table_name, column_name)
);

-- Relationship metadata (foreign keys)
CREATE TABLE catalog.relationship_metadata (
    id SERIAL PRIMARY KEY,
    constraint_name VARCHAR(100),
    source_schema VARCHAR(100),
    source_table VARCHAR(100),
    source_column VARCHAR(100),
    target_schema VARCHAR(100),
    target_table VARCHAR(100),
    target_column VARCHAR(100),
    relationship_type VARCHAR(50), -- ONE_TO_ONE, ONE_TO_MANY, MANY_TO_MANY
    delete_rule VARCHAR(20), -- CASCADE, SET NULL, RESTRICT, etc.
    update_rule VARCHAR(20),
    metadata_text TEXT,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index metadata for performance optimization
CREATE TABLE catalog.index_metadata (
    id SERIAL PRIMARY KEY,
    schema_name VARCHAR(100),
    table_name VARCHAR(100),
    index_name VARCHAR(100),
    index_type VARCHAR(50), -- btree, hash, gin, gist, etc.
    is_unique BOOLEAN,
    is_primary BOOLEAN,
    columns TEXT[], -- Array of column names
    index_size_bytes BIGINT,
    index_scans BIGINT, -- From pg_stat_user_indexes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Query patterns and usage statistics
CREATE TABLE catalog.query_patterns (
    id SERIAL PRIMARY KEY,
    pattern_hash VARCHAR(64), -- MD5 hash of normalized query
    pattern_text TEXT, -- Normalized query pattern
    tables_used TEXT[], -- Array of table names
    columns_used TEXT[], -- Array of column names
    join_conditions TEXT[],
    aggregations TEXT[],
    execution_count INTEGER DEFAULT 1,
    avg_execution_time_ms NUMERIC,
    last_execution TIMESTAMP,
    metadata_text TEXT,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Suggested KPIs based on analysis
CREATE TABLE catalog.suggested_kpis (
    id SERIAL PRIMARY KEY,
    kpi_name VARCHAR(200),
    kpi_description TEXT,
    kpi_category VARCHAR(100), -- Sales, Inventory, Customer, etc.
    measure_expression TEXT,
    dimension_columns TEXT[],
    required_tables TEXT[],
    query_template TEXT,
    metadata_text TEXT,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Mart execution history
CREATE TABLE catalog.mart_execution_history (
    id SERIAL PRIMARY KEY,
    execution_id UUID DEFAULT uuid_generate_v4(),
    user_question TEXT,
    retrieved_metadata JSONB, -- Top-K metadata results
    mart_plan JSONB, -- Generated mart plan
    mart_schema VARCHAR(100),
    execution_status VARCHAR(50), -- PLANNING, EXECUTING, COMPLETED, FAILED
    error_message TEXT,
    tables_created INTEGER,
    rows_processed BIGINT,
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Create indexes for vector similarity search
CREATE INDEX idx_table_metadata_embedding ON catalog.table_metadata
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_column_metadata_embedding ON catalog.column_metadata
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_relationship_metadata_embedding ON catalog.relationship_metadata
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_query_patterns_embedding ON catalog.query_patterns
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_suggested_kpis_embedding ON catalog.suggested_kpis
    USING hnsw (embedding vector_cosine_ops);

-- Create regular indexes for lookups
CREATE INDEX idx_column_metadata_table ON catalog.column_metadata(schema_name, table_name);
CREATE INDEX idx_column_metadata_fk ON catalog.column_metadata(is_foreign_key) WHERE is_foreign_key = true;
CREATE INDEX idx_relationship_source ON catalog.relationship_metadata(source_schema, source_table);
CREATE INDEX idx_relationship_target ON catalog.relationship_metadata(target_schema, target_table);
CREATE INDEX idx_query_patterns_tables ON catalog.query_patterns USING gin(tables_used);
CREATE INDEX idx_mart_execution_status ON catalog.mart_execution_history(execution_status);

-- Helper view for complete column information with relationships
CREATE VIEW catalog.v_column_relationships AS
SELECT
    c.*,
    r.target_schema,
    r.target_table,
    r.target_column,
    r.relationship_type
FROM catalog.column_metadata c
LEFT JOIN catalog.relationship_metadata r
    ON c.schema_name = r.source_schema
    AND c.table_name = r.source_table
    AND c.column_name = r.source_column;

-- Helper view for table relationships graph
CREATE VIEW catalog.v_table_relationships AS
SELECT DISTINCT
    source_schema,
    source_table,
    target_schema,
    target_table,
    COUNT(*) as relationship_count
FROM catalog.relationship_metadata
GROUP BY source_schema, source_table, target_schema, target_table;

-- Function to generate metadata text for embeddings
CREATE OR REPLACE FUNCTION catalog.generate_column_metadata_text(
    p_schema_name VARCHAR,
    p_table_name VARCHAR,
    p_column_name VARCHAR
) RETURNS TEXT AS $$
DECLARE
    v_metadata_text TEXT;
    v_fk_info TEXT;
    v_stats_info TEXT;
BEGIN
    -- Get foreign key information if exists
    SELECT
        CASE WHEN r.target_table IS NOT NULL
        THEN format(' references %s.%s(%s)', r.target_schema, r.target_table, r.target_column)
        ELSE ''
        END INTO v_fk_info
    FROM catalog.column_metadata c
    LEFT JOIN catalog.relationship_metadata r
        ON c.schema_name = r.source_schema
        AND c.table_name = r.source_table
        AND c.column_name = r.source_column
    WHERE c.schema_name = p_schema_name
        AND c.table_name = p_table_name
        AND c.column_name = p_column_name;

    -- Get statistical information
    SELECT
        format(' distinct_values:%s null_fraction:%s',
            COALESCE(n_distinct::TEXT, 'unknown'),
            COALESCE(null_fraction::TEXT, 'unknown')
        ) INTO v_stats_info
    FROM catalog.column_metadata
    WHERE schema_name = p_schema_name
        AND table_name = p_table_name
        AND column_name = p_column_name;

    -- Combine all information
    v_metadata_text := format(
        '%s.%s.%s type:%s%s%s%s',
        p_schema_name,
        p_table_name,
        p_column_name,
        (SELECT data_type FROM catalog.column_metadata
         WHERE schema_name = p_schema_name
         AND table_name = p_table_name
         AND column_name = p_column_name),
        v_fk_info,
        v_stats_info,
        COALESCE(
            (SELECT ' comment:' || column_comment
             FROM catalog.column_metadata
             WHERE schema_name = p_schema_name
             AND table_name = p_table_name
             AND column_name = p_column_name
             AND column_comment IS NOT NULL),
            ''
        )
    );

    RETURN v_metadata_text;
END;
$$ LANGUAGE plpgsql;