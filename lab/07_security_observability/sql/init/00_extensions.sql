-- 00_extensions.sql — extensions for the security observability lab
-- Runs once at container init. pgaudit + anon are preloaded via the server
-- command line (shared_preload_libraries=pgaudit,anon).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pgaudit;

-- postgresql_anonymizer
CREATE EXTENSION IF NOT EXISTS anon CASCADE;
SELECT anon.init();

-- Fixed salt so anon.pseudo_company() is reproducible across rebuilds
-- (the masked re-embedding must be deterministic for a repeatable demo).
ALTER DATABASE secobs SET anon.salt = 'swiss-pgday-2026';
