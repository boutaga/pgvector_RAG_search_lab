-- 00_extensions.sql - extensions for the bank database (lab 16)
-- pgaudit, anon and pg_stat_statements are preloaded on the server command line.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgaudit;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS anon CASCADE;
