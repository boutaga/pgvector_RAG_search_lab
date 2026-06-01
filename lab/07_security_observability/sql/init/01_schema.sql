-- 01_schema.sql — multi-tenant finance corpus + RLS baseline
-- =============================================================================
-- Three tenants (banks). RLS is enabled on `documents` from the start — that is
-- the "source isolation works" beat. The `embeddings` table deliberately has
-- NO RLS at baseline: it holds the chunk text + vector, so an unscoped ANN
-- search leaks other tenants' content. That missing policy IS the security debt.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS app;

CREATE TABLE app.documents (
    id             bigserial PRIMARY KEY,
    tenant_id      text NOT NULL,            -- bank_a / bank_b / bank_c
    client_name    text NOT NULL,            -- the sensitive field anon will mask
    title          text NOT NULL,
    body           text NOT NULL,
    classification text NOT NULL DEFAULT 'internal',
    created_at     timestamptz NOT NULL DEFAULT now()
);

-- One chunk per document keeps relevance labels exact for the demo. The
-- embeddings table carries the chunk text itself (as real RAG chunk stores do),
-- which is exactly why forgetting RLS here leaks the sensitive text.
CREATE TABLE app.embeddings (
    id                bigserial PRIMARY KEY,
    tenant_id         text NOT NULL,
    document_id       bigint NOT NULL REFERENCES app.documents(id) ON DELETE CASCADE,
    chunk_text        text NOT NULL,         -- text embedded at baseline (real client name)
    chunk_text_masked text,                  -- text embedded after anonymization
    embedding         vector(1536),          -- baseline embedding (text-embedding-3-small)
    embedding_masked  vector(1536)           -- masked embedding
);

CREATE INDEX embeddings_hnsw        ON app.embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX embeddings_masked_hnsw ON app.embeddings USING hnsw (embedding_masked vector_cosine_ops);
CREATE INDEX embeddings_tenant_idx  ON app.embeddings (tenant_id);
CREATE INDEX documents_tenant_idx   ON app.documents (tenant_id);

-- App login role: NON-superuser, so RLS applies to it. The app connects as
-- app_user and SETs app.tenant_id per request. Seeding uses superuser dba_admin.
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app_user') THEN
    CREATE ROLE app_user LOGIN PASSWORD 'dbi2026!' NOSUPERUSER;
  END IF;
END$$;
GRANT USAGE ON SCHEMA app TO app_user;
GRANT SELECT ON ALL TABLES IN SCHEMA app TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT SELECT ON TABLES TO app_user;

-- RLS on documents (source isolation works) ----------------------------------
ALTER TABLE app.documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY documents_tenant_isolation ON app.documents
    USING (tenant_id = current_setting('app.tenant_id', true));

-- NOTE: app.embeddings has NO row level security yet. The fix lives in
-- sql/demo/10_rls_embeddings.sql and is applied live during the talk.
