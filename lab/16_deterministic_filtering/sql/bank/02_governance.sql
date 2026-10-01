-- 02_governance.sql - the governance layers, in the order they stack
-- =============================================================================
--   1. Row-level security: who sees which bank's rows (the Swiss PGDay lab
--      pattern, applied to documents AND embeddings from the start)
--   2. Labelling: which columns are sensitive, and in which category
--   3. Egress log: what left the perimeter (hashes and scan results, never payloads)
--   4. Quality runs: retrieval quality measured per text state
--   5. Coverage and maturity: SQL views that say where the database stands
-- =============================================================================

-- 1. Row-level security ---------------------------------------------------------
-- The tenant comes from app.bank_id, set per transaction (set_config(..., true))
-- so a connection pooler cannot carry one bank's setting into the next request.
-- current_setting(..., true) returns NULL when unset: no tenant = zero rows.
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['relationship_managers','clients','accounts','servers',
                           'documents','document_mentions','embeddings'] LOOP
    EXECUTE format('ALTER TABLE bank.%I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format($p$CREATE POLICY tenant_isolation ON bank.%I
                     USING (bank_id = current_setting('app.bank_id', true))$p$, t);
  END LOOP;
END$$;

-- 2. Labelling ------------------------------------------------------------------
-- The registry says what is sensitive and where its token goes. The security
-- labels (postgresql_anonymizer provider) are generated FROM the registry, so a
-- masked human role sees the category instead of the value. The catalog
-- (pg_seclabels) is then the single place to ask "is this column labelled?".
CREATE TABLE gov.sensitive_columns (
    table_name   text NOT NULL,
    column_name  text NOT NULL,
    category     text NOT NULL,   -- CLIENT, PERSON, EMAIL, IBAN, HOST, IP, FREE_TEXT
    token_column text,            -- where the deterministic token is written (NULL for free text)
    PRIMARY KEY (table_name, column_name)
);

INSERT INTO gov.sensitive_columns VALUES
    ('clients',               'client_name', 'CLIENT',    'client_token'),
    ('relationship_managers', 'full_name',   'PERSON',    'rm_token'),
    ('relationship_managers', 'email',       'EMAIL',     'email_token'),
    ('accounts',              'iban',        'IBAN',      'iban_token'),
    ('servers',               'hostname',    'HOST',      'host_token'),
    ('servers',               'fqdn',        'HOST',      'fqdn_token'),
    ('servers',               'ip_address',  'IP',        'ip_token'),
    ('documents',             'title',       'FREE_TEXT', NULL),
    ('documents',             'body',        'FREE_TEXT', NULL);
-- clients.contact_phone is sensitive and deliberately NOT registered:
-- the coverage view must report it as a gap.

CREATE PROCEDURE gov.apply_labels()
LANGUAGE plpgsql AS $$
DECLARE r record;
BEGIN
  FOR r IN SELECT * FROM gov.sensitive_columns LOOP
    EXECUTE format('SECURITY LABEL FOR anon ON COLUMN bank.%I.%I IS %L',
                   r.table_name, r.column_name,
                   format('MASKED WITH VALUE %L', '[' || r.category || ']'));
  END LOOP;
END$$;
CALL gov.apply_labels();

-- Coverage: every text column of the business tables, labelled or not.
-- Derived columns (tokens, redacted and tokenized copies) and identifiers are excluded.
CREATE VIEW gov.label_coverage AS
SELECT c.table_name,
       c.column_name,
       sl.label,
       sc.category,
       (sl.label IS NOT NULL) AS labelled
FROM information_schema.columns c
LEFT JOIN pg_seclabels sl
       ON sl.provider = 'anon' AND sl.objtype = 'column'
      AND sl.objoid = format('bank.%I', c.table_name)::regclass
      AND sl.objsubid = c.ordinal_position
LEFT JOIN gov.sensitive_columns sc
       ON sc.table_name = c.table_name AND sc.column_name = c.column_name
WHERE c.table_schema = 'bank'
  AND c.data_type = 'text'
  AND c.table_name NOT IN ('embedding_versions', 'document_mentions', 'banks')
  AND c.column_name NOT LIKE '%\_token' ESCAPE '\'
  AND c.column_name NOT LIKE '%\_redacted' ESCAPE '\'
  AND c.column_name NOT LIKE '%\_tokenized' ESCAPE '\'
  AND c.column_name NOT IN ('bank_id', 'doc_type', 'client_type', 'currency',
                            'role', 'environment', 'domicile', 'category', 'token');

-- A human analyst who sees labelled columns masked (anon transparent dynamic masking).
CREATE ROLE analyst_masked LOGIN PASSWORD 'dbi2026!';
SECURITY LABEL FOR anon ON ROLE analyst_masked IS 'MASKED';
ALTER DATABASE bank SET anon.transparent_dynamic_masking = true;

-- 3. Egress log -----------------------------------------------------------------
-- One row per request that leaves for an external API. The payload itself is
-- never stored (a log of payloads would be a second leak): only its hash, size,
-- and what the scanner found in it.
CREATE TABLE gov.egress_log (
    egress_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts             timestamptz NOT NULL DEFAULT clock_timestamp(),
    destination    text NOT NULL,          -- e.g. openai:gpt-5.4-mini
    purpose        text NOT NULL,          -- agent_turn, embedding, ...
    bank_id        text,
    filtering      text NOT NULL CHECK (filtering IN ('off', 'tokenized')),
    payload_sha256 text NOT NULL,
    payload_chars  int  NOT NULL,
    hits           int  NOT NULL,
    hit_categories text[] NOT NULL DEFAULT '{}',
    blocked        boolean NOT NULL,
    -- what happened to the request: only 'sent' left the perimeter.
    -- 'attempted' = logged before the call, outcome unknown (process died mid-call).
    outcome        text NOT NULL DEFAULT 'attempted'
                   CHECK (outcome IN ('attempted', 'sent', 'failed', 'blocked', 'dry_run')),
    run_label      text            -- groups the requests of one evaluation run
);

-- 4. Quality runs ---------------------------------------------------------------
CREATE TABLE gov.quality_runs (
    run_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts          timestamptz NOT NULL DEFAULT now(),
    version_id  int  NOT NULL REFERENCES bank.embedding_versions,
    state       text NOT NULL,
    method      text NOT NULL,             -- dense, sparse, hybrid
    question_set text NOT NULL,            -- all, entity, generic
    k           int  NOT NULL,
    recall      numeric(5,3) NOT NULL,
    ndcg        numeric(5,3) NOT NULL,
    n_questions int  NOT NULL
);

-- 4b. Answer runs: relevance of the agent's answers, naive versus governed ----------
CREATE TABLE gov.answer_runs (
    run_id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts              timestamptz NOT NULL DEFAULT now(),
    label           text NOT NULL,             -- baseline, entity_retrieval, ...
    mode            text NOT NULL,             -- off (naive) or tokenized (governed)
    question_set    text NOT NULL,
    n_questions     int  NOT NULL,
    cited_recall    numeric(5,3) NOT NULL,
    cited_precision numeric(5,3) NOT NULL,
    context_recall  numeric(5,3) NOT NULL,
    explicit_entity numeric(5,3),            -- answer names the asked entity (token or name)
    blocked         int  NOT NULL
);
