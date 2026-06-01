-- 10_rls_embeddings.sql — THE FIX (applied live, Act 2 step 2.1)
-- Add the row level security the embeddings table was missing. One policy,
-- one table. After this, the same RAG query no longer leaks other tenants.

ALTER TABLE app.embeddings ENABLE ROW LEVEL SECURITY;

CREATE POLICY embeddings_tenant_isolation ON app.embeddings
    USING (tenant_id = current_setting('app.tenant_id', true));
