-- 0_rls.sql - layer 1, row-level security: who sees which bank's rows
-- Run as app_agent:  psql -h localhost -p 5437 -U app_agent -d bank -f 0_rls.sql
\set ECHO queries

-- No tenant set: zero rows, not everything.
SELECT count(*) AS documents_without_tenant FROM bank.documents;
SELECT count(*) AS embeddings_without_tenant FROM bank.embeddings;

-- Tenant set for this transaction only (set_config(..., true) = SET LOCAL).
BEGIN;
SELECT set_config('app.bank_id', 'bank_a', true);
SELECT bank_id, count(*) AS documents FROM bank.documents GROUP BY bank_id;
SELECT bank_id, count(*) AS embeddings FROM bank.embeddings GROUP BY bank_id;
COMMIT;

-- After the transaction the setting is gone: a pooler cannot leak it to the next request.
SELECT count(*) AS documents_after_commit FROM bank.documents;

-- RLS is on for both tables (the Swiss PGDay lab's leak was embeddings without it).
SELECT relname, relrowsecurity FROM pg_class
WHERE oid IN ('bank.documents'::regclass, 'bank.embeddings'::regclass);
