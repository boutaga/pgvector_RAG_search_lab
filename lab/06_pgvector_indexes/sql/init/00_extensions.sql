-- =============================================================================
-- Lab 06 — 00_extensions.sql
-- Create required extensions and verify setup
-- =============================================================================

-- pgvector: vector data type, distance operators, HNSW, IVFFlat
CREATE EXTENSION IF NOT EXISTS vector;

-- pgvectorscale: StreamingDiskANN indexes (Timescale)
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;

-- Verify versions
SELECT extname, extversion
FROM pg_extension
WHERE extname IN ('vector', 'vectorscale')
ORDER BY extname;

-- Verify available index access methods
SELECT amname
FROM pg_am
WHERE amname IN ('hnsw', 'ivfflat', 'diskann')
ORDER BY amname;
