-- 20_anonymize.sql — anonymize client names (applied live, Act 2 step 2.3)
-- =============================================================================
-- Demonstrates postgresql_anonymizer on stage. A masking rule is attached to
-- the sensitive column, and a masked role sees faked company names instead of
-- real client names.
--
-- The measurement path (measure_security_cost.py) uses the pre-computed
-- `embedding_masked` column, produced by `embed.py --mode masked` with
-- anon.pseudo_company() (deterministic). This live block is what the audience
-- sees; the re-embedding is what justifies the recall cost.
-- =============================================================================

-- Tag the sensitive column with a masking function.
SECURITY LABEL FOR anon ON COLUMN app.documents.client_name
    IS 'MASKED WITH FUNCTION anon.fake_company()';

-- A masked role: reads of client_name come back faked for this role.
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'bank_masked') THEN
    CREATE ROLE bank_masked LOGIN PASSWORD 'dbi2026!' NOSUPERUSER;
  END IF;
END$$;
GRANT USAGE ON SCHEMA app TO bank_masked;
GRANT SELECT ON app.documents, app.embeddings TO bank_masked;
SECURITY LABEL FOR anon ON ROLE bank_masked IS 'MASKED';

-- Turn on dynamic masking so masked roles read faked values.
SELECT anon.start_dynamic_masking();

-- On stage: connect as bank_masked and SELECT client_name to show faked values:
--   \c secobs bank_masked
--   SELECT client_name FROM app.documents LIMIT 5;
