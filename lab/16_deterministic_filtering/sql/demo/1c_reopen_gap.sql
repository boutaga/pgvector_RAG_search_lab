-- 1c_reopen_gap.sql - put the gap back, to replay the demo from the start
-- Run as lab_admin, then re-run python/tokenize_corpus.py.
\set ECHO queries
DELETE FROM gov.sensitive_columns WHERE table_name = 'clients' AND column_name = 'contact_phone';
SECURITY LABEL FOR anon ON COLUMN bank.clients.contact_phone IS NULL;
SELECT table_name, column_name, labelled FROM gov.label_coverage WHERE NOT labelled;
