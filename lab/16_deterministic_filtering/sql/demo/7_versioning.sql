-- 7_versioning.sql - embedding versions: which one answers, and switching back
-- Run as lab_admin.
\set ECHO queries
SELECT v.version_id, v.state, v.dense_model, v.dims, v.key_version, v.is_active, count(e.*) AS rows
FROM bank.embedding_versions v LEFT JOIN bank.embeddings e USING (version_id)
GROUP BY v.version_id ORDER BY v.version_id;

-- Rollback = one transaction: deactivate the current tokenized version, activate the previous one.
-- Only valid if the previous version's key_version matches the tokens in bank.documents.
-- BEGIN;
-- UPDATE bank.embedding_versions SET is_active = false WHERE state = 'tokenized' AND is_active;
-- UPDATE bank.embedding_versions SET is_active = true  WHERE version_id = <previous id>;
-- COMMIT;
