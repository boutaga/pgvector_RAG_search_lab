-- =============================================================================
-- Lab 16 walkthrough: the same question asked two ways, with the real output
-- =============================================================================
-- Run on 2026-10-01 against the local lab (lab16_pg18 :5437, lab16_vault :5438).
-- Every output block below was captured from that run, not written by hand.
--
-- Example A, the naive setup: raw question, raw documents read by an account
--   that sees every bank, sent as is to the model.
-- Example B, the governed setup: the same question through six layers,
--   row-level security, labelling, deterministic tokens, the egress gate,
--   the vault, and audit, then measured for relevance and scored.
--
-- How to replay: SQL steps run in psql as lab_admin unless a step says
--   otherwise. Python steps are shown as comments with their command; run them
--   from the lab folder with ~/venvs/lab16/bin/python.
--
--   psql -h localhost -p 5437 -U lab_admin -d bank        (password dbi2026!)
--
-- The model: OpenAI gpt-6-luna (set in .env), called through Chat Completions
-- with function tools (reasoning_effort = none, which that endpoint requires
-- for tools). Steps A1, B5 and B7 use --dry-run, which runs the agent's first
-- step for real (question, database search, exact payload, egress scan, log
-- entry) and stops before sending, so the gate's decision can be shown on its
-- own. Step B7b and section B10b are live: the model answered.
--
-- Data: three synthetic banks (bank_a Alder Private Bank, bank_b Birchwood
-- Bank, bank_c Cedar Trust), 18 relationship managers, 240 clients, 481
-- accounts, 30 servers, 1,521 documents (advisor notes, emails, incident
-- tickets). All names, IBANs, hosts and addresses are invented.
-- =============================================================================


-- #############################################################################
-- EXAMPLE A - the naive setup
-- #############################################################################

-- -----------------------------------------------------------------------------
-- A1. Ask the question the naive way.
--     The pipeline account searches the raw embeddings, keeps bank_a's rows by
--     an application-side WHERE, and builds the payload from raw text.
--
--   python python/agent.py --dry-run --bank bank_a --filtering off \
--     "What did Loic Kuhn discuss about renewing know-your-customer documents?"
-- -----------------------------------------------------------------------------
-- Output:
--   question sent : What did Loic Kuhn discuss about renewing know-your-customer documents?
--     [tool] search_documents returned:
--       doc 350 advisor_note: Valentina Zanetti met Loic Kuhn. The client was reminded that the
--                             beneficial owner declaration has expired. ...
--       doc 351 email: From valentina.zanetti@alder-bank.example. Dear client, following our call,
--                      Loic Kuhn has not yet returned the updated passport copy for the KYC review ...
--       doc 354 email: ... Loic Kuhn wants the mandate aligned with a sustainability rating ...
--       doc 352 advisor_note: Valentina Zanetti met Loic Kuhn. The client needs to confirm the source
--                             of wealth for the periodic review ...
--       doc 353 advisor_note: ... Loic Kuhn. The client asked to exclude fossil fuel producers ...
--     [egress] 2996 chars, filtering=off, scanner hits=29 ['CLIENT', 'EMAIL', 'IBAN', 'PERSON']
--     payload cleared the gate (dry run: not sent)
--
-- Reading: the answer would be good (all three expected documents, 350, 351,
-- 352, are in the top five), and one request carries 29 sensitive values:
-- the client's name, the advisor's name, the advisor's email and an IBAN.

-- -----------------------------------------------------------------------------
-- A2. What the egress log recorded for that request.
-- -----------------------------------------------------------------------------
SELECT egress_id, purpose, bank_id, filtering, payload_chars, hits, hit_categories, blocked
FROM gov.egress_log WHERE filtering = 'off';
-- Output:
--  egress_id |  purpose   | bank_id | filtering | payload_chars | hits |       hit_categories       | blocked
-- -----------+------------+---------+-----------+---------------+------+----------------------------+---------
--          4 | agent_turn | bank_a  | off       |          2996 |   29 | {CLIENT,EMAIL,IBAN,PERSON} | f
--
-- The log keeps the hash and size of the payload, never the payload itself
-- (a log of payloads would be a second copy of the leak).

-- -----------------------------------------------------------------------------
-- A3. Across the whole corpus (Python, no model call): how many sensitive
--     values does raw text carry?
-- -----------------------------------------------------------------------------
-- Output of the scanner over all 1,521 documents, raw text:
--   CLIENT 3024, PERSON 1833, EMAIL 960, HOST 630, IBAN 480, PHONE 154, IP 90  -> 7,171 values


-- #############################################################################
-- EXAMPLE B - the governed setup
-- #############################################################################

-- -----------------------------------------------------------------------------
-- B1. Layer 1, row-level security. Run as app_agent, the role the agent's
--     tools use.   psql -h localhost -p 5437 -U app_agent -d bank
-- -----------------------------------------------------------------------------
SELECT count(*) AS documents_without_tenant FROM bank.documents;
SELECT count(*) AS embeddings_without_tenant FROM bank.embeddings;
-- Output: 0 and 0. No tenant set means zero rows, not everything.

BEGIN;
SELECT set_config('app.bank_id', 'bank_a', true);   -- transaction-local, like SET LOCAL
SELECT bank_id, count(*) AS documents FROM bank.documents GROUP BY bank_id;
-- Output:
--  bank_id | documents
-- ---------+-----------
--  bank_a  |       508
SELECT bank_id, count(*) AS embeddings FROM bank.embeddings GROUP BY bank_id;
-- Output: bank_a | 1524   (508 documents x 3 embedding versions: raw, redacted, tokenized)
COMMIT;

SELECT count(*) AS documents_after_commit FROM bank.documents;
-- Output: 0. The setting died with the transaction, so a connection pooler
-- cannot hand bank_a's setting to the next request.

-- Back to lab_admin for the rest:  \connect bank lab_admin

-- -----------------------------------------------------------------------------
-- B2. Layer 2, labelling. The registry says what is sensitive; the security
--     labels are generated from it; the coverage view finds the gap.
--     State at the start of the demo (before step B6):
-- -----------------------------------------------------------------------------
SELECT table_name, column_name, category, labelled FROM gov.label_coverage
ORDER BY labelled, table_name, column_name;
-- Output (start of demo):
--       table_name       |  column_name  | category  | labelled
-- -----------------------+---------------+-----------+----------
--  clients               | contact_phone |           | f         <- the gap
--  accounts              | iban          | IBAN      | t
--  clients               | client_name   | CLIENT    | t
--  documents             | body          | FREE_TEXT | t
--  documents             | title         | FREE_TEXT | t
--  relationship_managers | email         | EMAIL     | t
--  relationship_managers | full_name     | PERSON    | t
--  servers               | fqdn          | HOST      | t
--  servers               | hostname      | HOST      | t
--  servers               | ip_address    | IP        | t

-- What a masked human analyst sees (postgresql_anonymizer dynamic masking):
--   psql -h localhost -p 5437 -U analyst_masked -d bank
--   BEGIN; SELECT set_config('app.bank_id', 'bank_a', true);
--   SELECT client_name, contact_phone, domicile FROM bank.clients ORDER BY client_id LIMIT 3;
-- Output:
--  client_name |  contact_phone   | domicile
-- -------------+------------------+----------
--  [CLIENT]    | +41 77 934 52 95 | Lausanne
--  [CLIENT]    | +41 77 470 43 63 | Geneva
--  [CLIENT]    | +41 77 340 72 29 | Lugano
-- Labelled column masked, unlabelled column in clear: the gap, made visible.

-- -----------------------------------------------------------------------------
-- B3. Layer 3, deterministic tokens. Same value, same token, everywhere.
-- -----------------------------------------------------------------------------
SELECT client_name, client_token FROM bank.clients WHERE client_name = 'Emmental Immobilien AG';
-- Output:  Emmental Immobilien AG | CLIENT_43aa5d8366b5

SELECT d.doc_id, d.doc_type, left(d.body_tokenized, 95) AS tokenized
FROM bank.documents d JOIN bank.document_mentions m USING (doc_id)
WHERE m.token = (SELECT client_token FROM bank.clients WHERE client_name = 'Emmental Immobilien AG')
ORDER BY d.doc_id;
-- Output:
--  doc_id |   doc_type   | tokenized
-- --------+--------------+------------------------------------------------------------------------------
--      10 | advisor_note | PERSON_65a94b3e0336 met CLIENT_43aa5d8366b5. The client complained that the custody fees rose w
--      11 | email        | From EMAIL_288e9f2f11a1. Dear client, following our call, CLIENT_43aa5d8366b5 complained that t
--      12 | advisor_note | PERSON_65a94b3e0336 met CLIENT_43aa5d8366b5. The client complained that the custody fees rose w
--      13 | advisor_note | PERSON_65a94b3e0336 met CLIENT_43aa5d8366b5. The client requested an ESG screening of current e
--      14 | email        | From EMAIL_288e9f2f11a1. Dear client, following our call, CLIENT_43aa5d8366b5 requested an ESG
--      15 | advisor_note | PERSON_65a94b3e0336 met CLIENT_43aa5d8366b5. The client asked to exclude fossil fuel producers

-- Entities, not strings: a server's hostname and FQDN share one token.
SELECT hostname, fqdn, ip_address, host_token, fqdn_token, ip_token FROM bank.servers ORDER BY server_id LIMIT 2;
-- Output:
--     hostname    |             fqdn              |  ip_address  |    host_token     |    fqdn_token     |    ip_token
-- ----------------+-------------------------------+--------------+-------------------+-------------------+-----------------
--  alb-pg-prd-01  | alb-pg-prd-01.alder.internal  | 10.11.14.237 | HOST_f774b8916acd | HOST_f774b8916acd | IP_a7517e21e8cb
--  alb-ora-prd-02 | alb-ora-prd-02.alder.internal | 10.11.25.151 | HOST_3e20b95cc524 | HOST_3e20b95cc524 | IP_56ed1e540874

-- Redacted versus tokenized, same document:
SELECT doc_id, left(body_redacted, 80) AS redacted, left(body_tokenized, 80) AS tokenized
FROM bank.documents WHERE doc_id = 10;
-- Output:
--  10 | [REDACTED] met [REDACTED]. The client complained that the custody fees rose with
--     | PERSON_65a94b3e0336 met CLIENT_43aa5d8366b5. The client complained that the cust
-- Redaction loses who; the token keeps who without saying who.

-- -----------------------------------------------------------------------------
-- B4. Layer 5, the vault: a separate PostgreSQL server holds the key and the
--     mapping. From the bank, there is no path to it.
-- -----------------------------------------------------------------------------
SELECT count(*) AS foreign_servers FROM pg_foreign_server;                          -- Output: 0
SELECT extname FROM pg_extension WHERE extname IN ('postgres_fdw', 'dblink');      -- Output: (0 rows)
SELECT has_column_privilege('app_agent', 'bank.clients', 'client_name',  'SELECT') AS agent_reads_client_name,
       has_column_privilege('app_agent', 'bank.clients', 'client_token', 'SELECT') AS agent_reads_client_token;
-- Output:  f | t      The agent can read tokens, never the raw name.

-- On the vault (port 5438), by role:
--   app_agent:    psql -h localhost -p 5438 -U app_agent -d vault
--                 -> FATAL: password authentication failed for user "app_agent"
--                    (the role does not exist on the vault; roles there: reidentifier, tokenizer)
--   tokenizer:    SELECT * FROM vault.keys;
--                 -> ERROR: permission denied for table keys   (nobody reads the key)
--   tokenizer:    SELECT * FROM vault.tokenize('client', ARRAY['Brenta Kontor AG', 'brenta kontor ag ']);
--                 -> both spellings get the same token; computed inside the vault
--   reidentifier: SELECT token, value FROM vault.mapping
--                 WHERE token IN ('CLIENT_43aa5d8366b5', 'PERSON_65a94b3e0336');
--                 ->  CLIENT_43aa5d8366b5 | Emmental Immobilien AG
--                     PERSON_65a94b3e0336 | Valentina Zanetti
--   reidentifier: SELECT * FROM vault.keys;
--                 -> ERROR: permission denied for table keys

-- -----------------------------------------------------------------------------
-- B5. Layer 4, the egress gate, with the labelling gap still open.
--
--   python python/agent.py --dry-run --bank bank_a \
--     "What did Emmental Immobilien AG complain about regarding custody fees?"
-- -----------------------------------------------------------------------------
-- Output:
--   question sent : What did CLIENT_43aa5d8366b5 complain about regarding custody fees?
--     [tool] search_documents returned:
--       doc 10  advisor_note: PERSON_65a94b3e0336 met CLIENT_43aa5d8366b5. The client complained that the custody fees ...
--       doc 110 advisor_note: PERSON_de6876d5f585 met CLIENT_1028ccd3d88c. The client complained that the custody fees ...
--       doc 11  email: From EMAIL_288e9f2f11a1. ... CLIENT_43aa5d8366b5 complained that the custody fees ...
--       doc 12  advisor_note: PERSON_65a94b3e0336 met CLIENT_43aa5d8366b5. The client complained ... (contains the phone)
--       doc 249 email: From EMAIL_764856d120dc. ... CLIENT_378c4734d8be complained that the custody fees ...
--     [egress] 3689 chars, filtering=tokenized, scanner hits=1 ['PHONE']  BLOCKED
--     stopped at the egress gate: 1 sensitive value(s) in outbound payload: ['PHONE']
--
-- Reading: every labelled value was tokenized, but doc 12 still carries the
-- client's phone number, from the column nobody labelled. The scanner's phone
-- pattern caught it and the request never left.

-- -----------------------------------------------------------------------------
-- B6. Close the gap: label the column, then re-tokenize.
-- -----------------------------------------------------------------------------
INSERT INTO gov.sensitive_columns VALUES ('clients', 'contact_phone', 'PHONE', 'phone_token')
ON CONFLICT DO NOTHING;
CALL gov.apply_labels();
SELECT table_name, column_name, labelled FROM gov.label_coverage WHERE NOT labelled;
-- Output: (0 rows)

--   python python/tokenize_corpus.py
-- Output:
--   tokenizing registered columns through the vault
--     accounts.iban                    IBAN     481 tokens
--     clients.client_name              CLIENT   240 tokens
--     clients.contact_phone            PHONE    240 tokens      <- new
--     relationship_managers.email      EMAIL     18 tokens
--     relationship_managers.full_name  PERSON    18 tokens
--     servers.fqdn                     HOST      30 tokens
--     servers.hostname                 HOST      30 tokens
--     servers.ip_address               IP        30 tokens
--   filtering free text (key version 1)
--     1521 documents filtered, 3940 mentions, 1568 dictionary entries

-- -----------------------------------------------------------------------------
-- B7. Same question again, gap closed, with the reviewer's view.
--
--   python python/agent.py --dry-run --bank bank_a --reviewer \
--     "What did Emmental Immobilien AG complain about regarding custody fees?"
-- -----------------------------------------------------------------------------
-- Output:
--   question sent : What did CLIENT_43aa5d8366b5 complain about regarding custody fees?
--     [tool] search_documents returned: docs 10, 110, 11, 12, 249 (same as B5)
--     [egress] 3691 chars, filtering=tokenized, scanner hits=0
--     payload cleared the gate (dry run: not sent)
--     reviewer view, re-identified through the vault:
--       doc 10:  Valentina Zanetti met Emmental Immobilien AG. The client complained that the custody fees ...
--       doc 110: Reto Chappuis met Silvretta Software Sarl. The client complained that the custody fees ...
--
-- Reading: zero sensitive values in the outbound request, and only the
-- reidentifier role, on the vault server, can turn the tokens back into names.
-- Note on relevance: docs 10, 11 and 12 are the right client; 110 and 249 are
-- other clients with the same complaint. That was plain hybrid search on
-- tokens. Step B10 adds the entity-aware search that fixes it.

-- -----------------------------------------------------------------------------
-- B7b. The same question, live: the model answers, in tokens, and only the
--      reviewer sees names. (Run before the entity-aware search was added.)
--
--   python python/agent.py --bank bank_a --reviewer \
--     "What did Emmental Immobilien AG complain about regarding custody fees?"
-- -----------------------------------------------------------------------------
-- Output:
--   question sent : What did CLIENT_43aa5d8366b5 complain about regarding custody fees?
--     [egress] 1564 chars, filtering=tokenized, scanner hits=0
--     [tool] search_documents({"query": "CLIENT_43aa5d8366b5 custody fees", "k": 5})
--     [egress] 3699 chars, filtering=tokenized, scanner hits=0
--
--   answer (as the model wrote it):
--   CLIENT_43aa5d8366b5 complained that the custody fees had risen without notice.
--
--   answer re-identified through the vault (reviewer only):
--   Emmental Immobilien AG complained that the custody fees had risen without notice.
--
-- Reading: the model never saw the name, wrote a correct answer with the
-- token, and the vault turned it back for the one role allowed to see it.

-- -----------------------------------------------------------------------------
-- B8. Layer 4, the egress log for the whole run.
-- -----------------------------------------------------------------------------
SELECT egress_id, ts::time(0), purpose, bank_id, filtering, payload_chars, hits, hit_categories, blocked
FROM gov.egress_log ORDER BY egress_id;
-- Output:
--  egress_id |    ts    |  purpose   | bank_id | filtering | payload_chars | hits |       hit_categories       | blocked
-- -----------+----------+------------+---------+-----------+---------------+------+----------------------------+---------
--          4 | 14:26:08 | agent_turn | bank_a  | off       |          2996 |   29 | {CLIENT,EMAIL,IBAN,PERSON} | f       <- A1
--          5 | 14:26:23 | agent_turn | bank_a  | tokenized |          3764 |    0 | {}                         | f       <- Loic Kuhn, governed
--          6 | 14:26:51 | agent_turn | bank_a  | tokenized |          3689 |    1 | {PHONE}                    | t       <- B5
--          7 | 14:27:15 | agent_turn | bank_a  | tokenized |          3691 |    0 | {}                         | f       <- B7
-- Totals after the live runs and the answer evaluation (B7b, B10b):
-- (column outcome added after this run: only outcome = 'sent' left the perimeter)
SELECT filtering, outcome, count(*) AS requests, sum(hits) AS scanner_hits
FROM gov.egress_log GROUP BY filtering, outcome ORDER BY 1, 2;
-- Output:
--  filtering | requests | values_sent | blocked
--  off       |       88 |        1571 |       0
--  tokenized |      104 |           0 |       1

-- -----------------------------------------------------------------------------
-- B9. Layer 6, audit: pgAudit logs every function call made by app_agent.
--     From the host:  docker logs lab16_pg18 2>&1 | grep AUDIT | grep FUNCTION | tail
-- -----------------------------------------------------------------------------
-- Output (excerpt):
--   AUDIT: SESSION,2,2,FUNCTION,EXECUTE,FUNCTION,bank.retrieve,"SELECT doc_id, doc_type, created_at, title,
--          body FROM bank.search_documents($1::vector, $2::sparsevec, $3)",<not logged>
--   AUDIT: SESSION,2,4,FUNCTION,EXECUTE,FUNCTION,public.cosine_distance, ...
--   AUDIT: SESSION,2,5,FUNCTION,EXECUTE,FUNCTION,public.sparsevec_negative_inner_product, ...
-- Parameters are not logged on purpose (they are 1024-dimension vectors).

-- -----------------------------------------------------------------------------
-- B10. Relevance of RETRIEVAL: what did filtering cost? 36 labelled questions (30 name a
--      client or a server, 6 generic), each run against raw, redacted and
--      tokenized embeddings, as app_agent with its bank as tenant.
--
--   python python/measure.py
-- -----------------------------------------------------------------------------
-- Retrieval, recall@5 and nDCG@5 (entity questions = 30 that name a client or a server):
--
--  state      method   entity recall@5  entity nDCG@5  generic recall@5
--  ---------  -------  ---------------  -------------  ----------------
--  raw        dense              0.808          0.829             1.000
--  raw        sparse             0.967          0.932             1.000
--  raw        hybrid             0.922          0.915             1.000
--  raw        entity             1.000          0.989             1.000
--  redacted   dense              0.189          0.158             1.000
--  redacted   sparse             0.181          0.160             1.000
--  redacted   hybrid             0.175          0.167             1.000
--  redacted   entity             0.175          0.167             1.000   (a redacted question names no one)
--  tokenized  dense              0.414          0.432             1.000
--  tokenized  sparse             0.772          0.789             1.000
--  tokenized  hybrid             0.611          0.641             1.000
--  tokenized  entity             1.000          0.990             1.000
--
-- Reading:
-- 1. Redaction destroys entity questions (0.18): the documents no longer say who.
-- 2. Opaque tokens with semantic search alone recover part of it (hybrid 0.61):
--    the dense model barely uses a hex token, the sparse path does (0.77).
-- 3. Entity-aware search (sql/bank/05_entity_retrieval.sql) uses the mention
--    index that the labels already built: documents that mention the entity the
--    question names come first. Tokenized reaches 1.000, the same as raw. The
--    labels that stop the leak are the labels that win the relevance back.
-- 4. Caveats: each entity question names exactly one entity, and the six generic
--    questions score 1.000 everywhere, so they are too easy to discriminate.
--    This proves the mechanism, not a production figure.

-- From SQL:
SELECT state, method, question_set, k, recall, ndcg
FROM gov.quality_runs
WHERE run_id > (SELECT max(run_id) - 144 FROM gov.quality_runs)
  AND k = 5 AND question_set = 'entity'
ORDER BY state, method;

-- -----------------------------------------------------------------------------
-- B10b. Relevance of the ANSWERS, before and after filtering. The 36 questions
--       answered by the model twice: naive (raw text, plain hybrid search) and
--       governed (tokens only, entity-aware search, egress gate). The model
--       cites [doc N]; scoring is deterministic, no LLM judge.
--
--   python python/evaluate_answers.py --label entity_retrieval
-- -----------------------------------------------------------------------------
-- Output:
--  mode       set      cited rec  cited prec  context rec  right entity  blocked
--  off        entity       0.667       0.989        0.969           1.0        0
--  off        generic      1.000       1.000        1.000          None        0
--  off        all          0.722       0.991        0.975           1.0        0
--  tokenized  entity       0.803       0.989        1.000         0.933        0
--  tokenized  generic      1.000       1.000        1.000          None        0
--  tokenized  all          0.836       0.991        1.000         0.933        0
--  egress: off       75 requests, 1,114 sensitive values sent, 0 blocked
--          tokenized 87 requests,     0 sensitive values sent, 0 blocked
--  tokens: 124,758 prompt, 7,827 completion
--
-- Reading:
-- - Leak: the naive run sent 1,114 sensitive values to the provider in 75
--   requests; the governed run sent none in 87.
-- - Relevance: the governed answers cite more of the expected documents (0.80
--   against 0.67 on entity questions) because the governed search is
--   entity-aware and the naive one is not. Same precision (0.99).
-- - Right entity 0.933: in 2 of 30 governed answers the model wrote "the client"
--   instead of the token. Correct content, but the answer does not name who.
-- - Fairness note: the retrieval table shows raw text with entity-aware search
--   also reaches 1.000. The gain belongs to the labels, which is the point:
--   governance delivers the leak control and the relevance at the same time.

-- From SQL:
SELECT label, mode, question_set, n_questions, cited_recall, cited_precision, context_recall, explicit_entity, blocked
FROM gov.answer_runs WHERE label = 'entity_retrieval' ORDER BY mode, question_set;

-- -----------------------------------------------------------------------------
-- B11. Embedding versions. Closing the gap changed the tokenized text, so the
--      tokenized state was re-embedded as a new version; the old one stays
--      for rollback.
-- -----------------------------------------------------------------------------
SELECT v.version_id, v.state, v.dense_model, v.dims, v.key_version, v.is_active, count(e.*) AS rows
FROM bank.embedding_versions v LEFT JOIN bank.embeddings e USING (version_id)
GROUP BY v.version_id ORDER BY v.version_id;
-- Output:
--  version_id |   state   |      dense_model       | dims | key_version | is_active | rows
-- ------------+-----------+------------------------+------+-------------+-----------+------
--           3 | tokenized | voyageai/voyage-4-nano | 1024 |           1 | f         | 1521   <- before the gap was closed
--           4 | raw       | voyageai/voyage-4-nano | 1024 |             | t         | 1521
--           6 | redacted  | voyageai/voyage-4-nano | 1024 |             | t         | 1521
--           7 | tokenized | voyageai/voyage-4-nano | 1024 |           1 | t         | 1521   <- phones tokenized
-- Rollback to version 3 is one transaction (see sql/demo/7_versioning.sql),
-- valid only with the tokens it was built from.

-- -----------------------------------------------------------------------------
-- B12. Maturity scorecard.
-- -----------------------------------------------------------------------------
SELECT level, check_name, passed FROM gov.maturity_checks ORDER BY level, check_name;
SELECT * FROM gov.maturity;
-- Output (end of run):
--  level |                                check_name                                | passed
-- -------+--------------------------------------------------------------------------+--------
--      1 | RLS enabled on documents and embeddings                                  | t
--      2 | No unlabelled text column left (gaps listed in gov.label_coverage)       | t
--      2 | Sensitive columns are labelled                                           | t
--      3 | An active tokenized embedding version exists                             | t
--      3 | Every document has a tokenized copy                                      | t
--      4 | No filtered request left with a scanner hit (blocked ones never left)    | t
--      4 | Outbound requests are logged with filtering on                           | t
--      5 | No foreign server or dblink: the vault is unreachable from this database | t
--      5 | Retrieval quality measured on the tokenized state                        | t
--      5 | The agent role cannot read raw sensitive columns                         | t
--      5 | The agent role is audited (pgaudit)                                      | t
--
--  level_reached | checks_passed | checks_total
-- ---------------+---------------+--------------
--              5 |            11 |           11
-- At the start of the demo (gap open) the same query returns level 1: the
-- labelling check fails, and each level requires every level below it.

-- -----------------------------------------------------------------------------
-- B13. Query plans, as app_agent with bank_a set (full script: sql/demo/9_plans.sql).
-- -----------------------------------------------------------------------------
-- Dense path, ORDER BY dense <=> query LIMIT 10, output (vector literals cut):
--  Limit (actual time=3.196..3.199 rows=10.00 loops=1)
--    Buffers: shared hit=1753
--    ->  Sort  Sort Method: top-N heapsort  Memory: 25kB
--          ->  Nested Loop (actual time=0.060..3.060 rows=508.00 loops=1)
--                ->  Index Scan using embedding_versions_one_active on embedding_versions v
--                      Index Cond: (state = 'tokenized'::text)
--                ->  Bitmap Heap Scan on embeddings e (rows=508.00)
--                      Recheck Cond: ((version_id = v.version_id)
--                                     AND (bank_id = current_setting('app.bank_id'::text, true)))
--                      ->  Bitmap Index Scan on embeddings_version_id_bank_id_idx
--  Execution Time: 3.358 ms
-- Sparse path, ORDER BY sparse <#> query LIMIT 10: same shape, Execution Time: 0.628 ms
-- Hybrid, bank.retrieve(...): Function Scan, Execution Time: 3.116 ms
--
-- Reading: at 508 rows per bank and version, the planner skips the HNSW index
-- and scores every candidate exactly (top-N heapsort over 508 rows). The
-- row-level security predicate is not a filter applied afterwards: it is part
-- of the index condition on (version_id, bank_id), next to the version. At this
-- size the tenant filter rides on the same index scan. With much larger tables
-- the planner would switch to the HNSW index (threshold not measured), and
-- hnsw.iterative_scan (set on bank.retrieve) keeps the RLS filter from starving
-- the result list.

-- =============================================================================
-- Known limit, shown on purpose: 9 planted documents name a lawyer who exists
-- in no table (data/planted.json). Dictionary filtering only knows labelled
-- values and no pattern describes a name, so these pass the gate. Labelling
-- coverage is the ceiling of what deterministic filtering can promise.
-- =============================================================================
