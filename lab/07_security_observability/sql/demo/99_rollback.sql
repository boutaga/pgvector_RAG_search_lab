-- 99_rollback.sql — reset to the baseline (leaking, unmasked) state.
-- Lets the demo be rehearsed repeatedly. Safe to run if nothing was applied.

-- Undo anonymization (ignore errors if masking was never started)
DO $$ BEGIN PERFORM anon.stop_dynamic_masking(); EXCEPTION WHEN OTHERS THEN NULL; END $$;
SECURITY LABEL FOR anon ON COLUMN app.documents.client_name IS NULL;
DO $$
BEGIN
  IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'bank_masked') THEN
    EXECUTE 'SECURITY LABEL FOR anon ON ROLE bank_masked IS NULL';
  END IF;
END$$;

-- Undo the embeddings RLS fix
DROP POLICY IF EXISTS embeddings_tenant_isolation ON app.embeddings;
ALTER TABLE app.embeddings DISABLE ROW LEVEL SECURITY;
