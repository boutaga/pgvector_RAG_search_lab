-- 1b_close_gap.sql - close the labelling gap the scanner found (contact_phone)
-- Run as lab_admin, then re-run python/tokenize_corpus.py.
\set ECHO queries
INSERT INTO gov.sensitive_columns VALUES ('clients', 'contact_phone', 'PHONE', 'phone_token')
ON CONFLICT DO NOTHING;
CALL gov.apply_labels();
SELECT table_name, column_name, labelled FROM gov.label_coverage WHERE NOT labelled;
