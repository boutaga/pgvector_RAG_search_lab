-- 1_labels.sql - layer 2, labelling: what is sensitive, and where are the gaps
-- Run as lab_admin.
\set ECHO queries

-- The registry, and the security labels generated from it.
SELECT * FROM gov.sensitive_columns ORDER BY table_name, column_name;
SELECT objname, label FROM pg_seclabels WHERE provider = 'anon' AND objtype = 'column' ORDER BY objname;

-- Coverage: every business text column. The gap is the point.
SELECT table_name, column_name, category, labelled FROM gov.label_coverage ORDER BY labelled, table_name;

-- A human analyst sees labelled columns masked, unlabelled ones in clear.
-- Row-level security applies to the analyst too, so a bank must be set.
\connect bank analyst_masked
BEGIN;
SELECT set_config('app.bank_id', 'bank_a', true);
SELECT client_name, contact_phone, domicile FROM bank.clients ORDER BY client_id LIMIT 3;
COMMIT;
