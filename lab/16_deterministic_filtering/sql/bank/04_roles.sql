-- 04_roles.sql - roles, privileges, audit, and the maturity scorecard
-- =============================================================================
--   lab_admin      superuser (container owner), DDL and seeding
--   gateway        the pipeline: reads raw data to tokenize it, writes tokens,
--                  embeddings and the egress log. BYPASSRLS because it serves
--                  all three banks; it is never exposed to the agent.
--   app_agent      what the agent's tools run as: row-level security applies,
--                  no privilege on any raw sensitive column, no account on the vault
--   analyst_masked a human who reads labelled columns masked (created in 02)
-- =============================================================================

CREATE ROLE gateway   LOGIN PASSWORD 'dbi2026!' BYPASSRLS;
CREATE ROLE app_agent LOGIN PASSWORD 'dbi2026!';

GRANT USAGE ON SCHEMA bank, gov TO gateway;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA bank TO gateway;
GRANT SELECT, INSERT ON gov.egress_log, gov.quality_runs, gov.answer_runs TO gateway;
GRANT UPDATE (outcome) ON gov.egress_log TO gateway;   -- the outcome only: the rest of the log is append-only
GRANT SELECT ON gov.sensitive_columns, gov.label_coverage TO gateway;
GRANT EXECUTE ON FUNCTION bank.retrieve(vector, sparsevec, text, text, int) TO gateway;

GRANT USAGE ON SCHEMA bank TO app_agent;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA bank FROM PUBLIC;
GRANT EXECUTE ON FUNCTION
    bank.retrieve(vector, sparsevec, text, text, int),
    bank.search_documents(vector, sparsevec, int),
    bank.client_profile(text),
    bank.host_documents(text)
  TO app_agent;

-- Column privileges: tokens and non-sensitive attributes only, never raw values.
GRANT SELECT (doc_id, bank_id, doc_type, created_at, title_tokenized, body_tokenized, key_version)
    ON bank.documents TO app_agent;
GRANT SELECT ON bank.document_mentions, bank.embeddings, bank.embedding_versions TO app_agent;
GRANT SELECT (client_id, bank_id, client_type, domicile, rm_id, client_token) ON bank.clients TO app_agent;
GRANT SELECT (rm_id, bank_id, rm_token) ON bank.relationship_managers TO app_agent;
GRANT SELECT (account_id, client_id, bank_id, currency, balance, iban_token) ON bank.accounts TO app_agent;
GRANT SELECT (server_id, bank_id, role, environment, host_token, fqdn_token, ip_token) ON bank.servers TO app_agent;

-- The masked analyst reads the business tables; anon masks the labelled columns.
GRANT USAGE ON SCHEMA bank TO analyst_masked;
GRANT SELECT ON bank.clients, bank.relationship_managers, bank.accounts, bank.servers TO analyst_masked;

-- Audit: every read and every function call made by the agent's role.
ALTER ROLE app_agent SET pgaudit.log = 'read, function';
ALTER ROLE app_agent SET pgaudit.log_parameter = 'off';  -- parameters are vectors, keep the log readable

-- Stale vectors: an active embedding whose text no longer matches what was embedded.
-- Re-tokenizing (a new label, a key rotation) changes the text; the vectors must follow.
CREATE VIEW gov.embedding_staleness AS
SELECT v.state, e.version_id, e.doc_id,
       e.text_sha256 IS NOT NULL AND e.text_sha256 = encode(sha256(convert_to(
           CASE v.state WHEN 'raw'       THEN d.title || '. ' || d.body
                        WHEN 'redacted'  THEN d.title_redacted || '. ' || d.body_redacted
                        ELSE                  d.title_tokenized || '. ' || d.body_tokenized END,
           'UTF8')), 'hex') AS fresh
FROM bank.embeddings e
JOIN bank.embedding_versions v USING (version_id)
JOIN bank.documents d USING (doc_id)
WHERE v.is_active;
GRANT SELECT ON gov.embedding_staleness TO gateway;

-- Lab control checklist ("maturity scorecard") ------------------------------------------
-- It checks the controls this lab implements. It cannot see what nobody labelled (the
-- planted names pass it), so it is a checklist of lab controls, not an assessment of an
-- organization's AI maturity.
-- Each level needs every check of the levels below it. The view reports each check
-- and the level reached. Levels:
--   1 isolation   row-level security on documents and embeddings
--   2 labelling   sensitive columns labelled, no unregistered gap
--   3 filtering   an active tokenized embedding version, every document tokenized
--   4 egress      outbound requests logged and scanned, no filtered request sent with a hit
--   5 separation  vault unreachable from here, agent cannot read raw columns,
--                 agent audited, retrieval quality measured on the tokenized state
CREATE VIEW gov.maturity_checks AS
SELECT * FROM (VALUES
  (1, 'RLS enabled on documents and embeddings',
      (SELECT bool_and(relrowsecurity) FROM pg_class
        WHERE oid IN ('bank.documents'::regclass, 'bank.embeddings'::regclass))),
  (2, 'Sensitive columns are labelled',
      (SELECT count(*) > 0 FROM gov.label_coverage WHERE labelled)),
  (2, 'No unlabelled text column left (gaps listed in gov.label_coverage)',
      (SELECT count(*) = 0 FROM gov.label_coverage WHERE NOT labelled)),
  (3, 'An active tokenized embedding version exists',
      EXISTS (SELECT FROM bank.embedding_versions WHERE state = 'tokenized' AND is_active)),
  (3, 'Every document has a tokenized copy',
      NOT EXISTS (SELECT FROM bank.documents WHERE body_tokenized IS NULL)),
  (3, 'Active embeddings match the current text (no stale or unverifiable vectors)',
      NOT EXISTS (SELECT FROM gov.embedding_staleness WHERE NOT fresh)),
  (4, 'Outbound requests are logged with filtering on',
      EXISTS (SELECT FROM gov.egress_log WHERE filtering = 'tokenized')),
  (4, 'No filtered request left with a scanner hit (blocked ones never left)',
      NOT EXISTS (SELECT FROM gov.egress_log
                  WHERE filtering = 'tokenized' AND hits > 0 AND outcome IN ('sent', 'attempted'))),
  (5, 'No foreign server or dblink: the vault is unreachable from this database',
      NOT EXISTS (SELECT FROM pg_foreign_server)
      AND NOT EXISTS (SELECT FROM pg_extension WHERE extname IN ('postgres_fdw', 'dblink'))),
  (5, 'The agent role cannot read raw sensitive columns',
      NOT has_column_privilege('app_agent', 'bank.documents', 'body', 'SELECT')
      AND NOT has_column_privilege('app_agent', 'bank.clients', 'client_name', 'SELECT')
      AND NOT has_column_privilege('app_agent', 'bank.servers', 'hostname', 'SELECT')),
  (5, 'The agent role is audited (pgaudit)',
      EXISTS (SELECT FROM pg_db_role_setting s JOIN pg_roles r ON r.oid = s.setrole
               WHERE r.rolname = 'app_agent' AND s.setconfig::text LIKE '%pgaudit.log=%')),
  (5, 'Retrieval quality measured on the tokenized state',
      EXISTS (SELECT FROM gov.quality_runs WHERE state = 'tokenized'))
) AS t(level, check_name, passed);

CREATE VIEW gov.maturity AS
SELECT coalesce(min(level) - 1, 5) AS level_reached,
       (SELECT count(*) FILTER (WHERE passed) FROM gov.maturity_checks) AS checks_passed,
       (SELECT count(*) FROM gov.maturity_checks) AS checks_total
FROM gov.maturity_checks
WHERE NOT coalesce(passed, false);

GRANT SELECT ON gov.maturity_checks, gov.maturity TO gateway;
