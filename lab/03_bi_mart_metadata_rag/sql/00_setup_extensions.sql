-- Enable required PostgreSQL extensions for BI Mart Metadata RAG lab
-- Run this first to ensure all required extensions are available

-- Enable pgvector for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation for unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable additional statistics capabilities
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Enable crosstab functionality for pivot operations
CREATE EXTENSION IF NOT EXISTS tablefunc;

-- Verify extensions are installed
SELECT
    extname AS extension_name,
    extversion AS version
FROM pg_extension
WHERE extname IN ('vector', 'uuid-ossp', 'pg_stat_statements', 'tablefunc')
ORDER BY extname;