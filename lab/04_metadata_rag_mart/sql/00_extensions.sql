-- 00_extensions.sql — PostgreSQL 18 extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Schemas — no raw data in PG, only metadata + governance + compute
CREATE SCHEMA IF NOT EXISTS catalog;      -- metadata + embeddings (Agent 1 reads)
CREATE SCHEMA IF NOT EXISTS governance;   -- audit trail, registry, masking, roles
CREATE SCHEMA IF NOT EXISTS rag_monitor;  -- RAG quality monitoring
CREATE SCHEMA IF NOT EXISTS data_mart;    -- agent-provisioned analytical tables
CREATE SCHEMA IF NOT EXISTS lake;         -- ephemeral staging (Parquet → PG → mart)
