# Lab 16 runbook: run it by hand and capture the output

Every step in the order the post tells it, from an empty lab to the last number. Each
step says what it does, gives the command to type, shows what my 2026-10-01 run printed,
and says what to look at. Your output should match except where noted: vault tokens are
random per vault (different hex, same behaviour), and model answers vary in wording.

Time: about 1 h 30 min of machine time, most of it embeddings (40 min plus a 25 min
rebuild) and the answer evaluation (5 to 20 min). They run in the background; see
"Background steps" below before starting the first one.

## Capturing output

Shell steps: pipe through `tee` into `captures/` (gitignored):

```bash
cd ~/path/to/Movies_pgvector_lab/lab/16_deterministic_filtering
mkdir -p captures
PY=~/venvs/lab16/bin/python
$PY data/generate_bank.py | tee captures/01_generate.txt
```

psql steps: open one session per role and log it.

```bash
export PGPASSWORD='dbi2026!'
psql -h localhost -p 5437 -U lab_admin -d bank -X
```
```
\pset pager off
\timing on
SET search_path = bank, gov, public;
\o | tee -a captures/psql_admin.txt
```

The tables live in schemas `bank` and `gov`, so without the search path `\dt` shows
nothing (or use `\dt bank.*` and `\dt gov.*`). The runbook's queries are schema-qualified
and work either way.

`\o | tee` writes every result to the file and to the screen. `\o` alone stops it.
VS Code works too (connections in the README); psql is closer to what the post prints.

## Background steps: one at a time, and how to watch them

Three steps take long and run in the background: the embeddings (step 7), the rebuild
after closing the gap (step 12) and the answer evaluation (step 14). Rules:

- **One Python step that loads the models at a time.** Each loads about 3 GB; two do not
  fit in the WSL VM and the kernel kills one. A lock enforces it: a second job stops at
  once with `Another lab job is using the embedding models (...)`. That includes
  `agent.py` and `measure.py`, so they wait for the background job too.
- **Watch it in a second terminal:**

  ```bash
  watch -n 15 ~/venvs/lab16/bin/python python/status.py     # versions, stale vectors, running job
  tail -f captures/07_embed.txt                              # progress lines every 160 documents
  ```

  `status.py` prints `model job: none running, you can start the next Python step`
  when it is safe to go on. psql steps can run any time; they do not use the models.
- **Do not start a step that needs a version that is still building.** Each step below
  says what it needs.

## Step 0. Start from an empty lab

This deletes the current lab data (both volumes). Skip it if you want to keep my run.

```bash
cd docker
docker compose down -v
docker compose up -d --build
docker compose ps
cd ..
```

Expected: `lab16_pg18` on 5437 and `lab16_vault` on 5438, both `healthy`. The bank
runs `sql/bank/00..05` at first start, the vault runs `sql/vault/01_vault.sql`. The two
servers are on separate Docker networks.

On a machine where the models were never downloaded, set `HF_HUB_OFFLINE=0` in `.env`
for the first run (step 7), then back to `1`: offline mode skips the Hugging Face check
on every model load, which otherwise fails on a flaky network.

## Step 1. Generate the synthetic bank

```bash
$PY data/generate_bank.py | tee captures/01_generate.txt
```
```
banks 3, relationship managers 18, clients 240, accounts 481, servers 30, documents 1521, questions 36, planted residuals 9
```

In psql (lab_admin), look at what the data looks like:

```sql
SELECT bank_id, doc_type, count(*) FROM bank.documents GROUP BY 1, 2 ORDER BY 1, 2;
SELECT doc_id, left(body, 140) FROM bank.documents WHERE doc_id IN (10, 12, 1500);
```

What to look at: real-looking names, emails, IBANs, hostnames and IPs in free text,
and in doc 12 a phone number. That phone number is the story of step 9.

## Step 2. Layer 1, row-level security

Open a second psql session as the agent's role:

```bash
psql -h localhost -p 5437 -U app_agent -d bank -X
```
```sql
SELECT count(*) AS documents_without_tenant FROM bank.documents;

BEGIN;
SELECT set_config('app.bank_id', 'bank_a', true);
SELECT bank_id, count(*) AS documents FROM bank.documents GROUP BY bank_id;
COMMIT;

SELECT count(*) AS documents_after_commit FROM bank.documents;

-- and the agent cannot read raw columns at all:
BEGIN;
SELECT set_config('app.bank_id', 'bank_a', true);
SELECT client_name FROM bank.clients LIMIT 1;
ROLLBACK;
```
```
 documents_without_tenant: 0      bank_a | 508      documents_after_commit: 0
 ERROR:  permission denied for table clients
```

What to look at: no tenant means zero rows, not everything; the setting dies with
the transaction; the raw column is refused by privilege, before RLS even matters.

## Step 3. Layer 2, labelling and the gap

Back in the lab_admin session:

```sql
SELECT * FROM gov.sensitive_columns ORDER BY table_name, column_name;
SELECT objname, label FROM pg_seclabels WHERE provider = 'anon' AND objtype = 'column' ORDER BY objname;
SELECT table_name, column_name, category, labelled FROM gov.label_coverage ORDER BY labelled, table_name;
```

Expected: nine registered columns, nine labels like `MASKED WITH VALUE '[CLIENT]'`, and
`clients.contact_phone` with `labelled = f` at the top of the coverage view.

The masked analyst (third session):

```bash
psql -h localhost -p 5437 -U analyst_masked -d bank -X
```
```sql
BEGIN;
SELECT set_config('app.bank_id', 'bank_a', true);
SELECT client_name, contact_phone, domicile FROM bank.clients ORDER BY client_id LIMIT 3;
COMMIT;
```
```
 client_name |  contact_phone   | domicile
-------------+------------------+----------
 [CLIENT]    | +41 77 934 52 95 | Lausanne
```

## Step 4. The vault

```bash
export VPW='vault2026!'
PGPASSWORD=$VPW psql -h localhost -p 5438 -U tokenizer -d vault -X
```
```sql
SELECT * FROM vault.keys;                                   -- ERROR: permission denied
SELECT * FROM vault.tokenize('client', ARRAY['Brenta Kontor AG', 'brenta kontor ag ']);
SELECT * FROM vault.mapping;                                -- ERROR: permission denied
```

Then try the agent on the vault, and the bank side:

```bash
PGPASSWORD='dbi2026!' psql -h localhost -p 5438 -U app_agent -d vault -c "select 1"
```
```
FATAL:  password authentication failed for user "app_agent"     (the role does not exist there)
```
```sql
-- lab_admin on the bank:
SELECT count(*) FROM pg_foreign_server;
SELECT extname FROM pg_extension WHERE extname IN ('postgres_fdw', 'dblink');
```

Clean up the test value (vault_admin) so it does not sit in the mapping:

```bash
docker exec lab16_vault psql -U vault_admin -d vault -c "DELETE FROM vault.mapping WHERE value ILIKE 'alder holding ag%'"
```

## Step 5. Tokenize the corpus

```bash
$PY python/tokenize_corpus.py | tee captures/05_tokenize.txt
```
```
tokenizing registered columns through the vault
  accounts.iban         IBAN     481 tokens
  clients.client_name  CLIENT   240 tokens
  relationship_managers.email        EMAIL     18 tokens
  relationship_managers.full_name    PERSON    18 tokens
  servers.fqdn         HOST      30 tokens
  servers.hostname     HOST      30 tokens
  servers.ip_address   IP        30 tokens
filtering free text (key version 1)
  1521 documents filtered, 3786 mentions, 1328 dictionary entries
```

In psql (lab_admin), the queries of the post's layer 3 section:

```sql
SELECT client_name, client_token FROM bank.clients WHERE client_name = 'Emmental Immobilien AG';

SELECT d.doc_id, d.doc_type, left(d.body_tokenized, 95) AS tokenized
FROM bank.documents d JOIN bank.document_mentions m USING (doc_id)
WHERE m.token = (SELECT client_token FROM bank.clients WHERE client_name = 'Emmental Immobilien AG')
ORDER BY d.doc_id;

SELECT hostname, fqdn, ip_address, host_token, fqdn_token, ip_token FROM bank.servers ORDER BY server_id LIMIT 2;

SELECT doc_id, left(body_redacted, 80) AS redacted, left(body_tokenized, 80) AS tokenized
FROM bank.documents WHERE doc_id = 10;
```

What to look at: one token for one client across six documents; hostname and FQDN with
the same token; `[REDACTED]` versus a token on the same sentence.

## Step 6. Scan the corpus

```bash
$PY python/scan_corpus.py | tee captures/06_scan.txt
```
```
raw         7171 values in 1521 documents: CLIENT 3024, PERSON 1833, EMAIL 960, HOST 630, IBAN 480, PHONE 154, IP 90
redacted     154 values in  154 documents: PHONE 154
tokenized    154 values in  154 documents: PHONE 154
```

What to look at: after tokenization, only the unlabelled phone numbers remain. The nine
planted lawyer names (`data/planted.json`) are not counted: no dictionary entry and no
pattern knows them. That is the limit section of the post.

## Step 7. Embed locally (background, about 40 minutes)

```bash
nohup $PY python/embed.py --state all > captures/07_embed.txt 2>&1 &
watch -n 15 $PY python/status.py
```

Tokenized first, then raw, then redacted. Capture the final state when it is done:

```bash
$PY python/status.py | tee captures/07_status.txt
```
```
embedding versions
    1  tokenized   1521/1521  active   key_version=1
    2  raw         1521/1521  active   key_version=None
    3  redacted    1521/1521  active   key_version=None
stale active vectors: none
model job: none running, you can start the next Python step
```

**Wait for `model job: none running` before step 8.** Steps 2 to 6 (psql) can be done
while it runs.

## Step 8. Example A, the naive setup

```bash
$PY python/agent.py --dry-run --bank bank_a --filtering off \
  "What did Loic Kuhn discuss about renewing know-your-customer documents?" | tee captures/08_naive.txt
```
```
question sent : What did Loic Kuhn discuss about renewing know-your-customer documents?
  [tool] search_documents returned:
    doc 350 advisor_note: Valentina Zanetti met Loic Kuhn. The client was reminded that ...
  [egress] 2996 chars, filtering=off, scanner hits=29 ['CLIENT', 'EMAIL', 'IBAN', 'PERSON']
  payload cleared the gate (dry run: not sent)
```

Needs the raw embeddings (step 7 done). Run before step 7 finished, the search returns
nothing and the request shows 2 hits instead of 29. `--dry-run` runs the first agent
step for real and stops before sending; its log row has `outcome = dry_run`. Remove `--dry-run` to send
it to the model for real (raw names included; synthetic data, so harmless).

## Step 9. Example B, the governed setup: the gate blocks, the label fixes it

```bash
Q="What did Emmental Immobilien AG complain about regarding custody fees?"
$PY python/agent.py --dry-run --bank bank_a "$Q" | tee captures/09a_gate_blocks.txt
```
```
question sent : What did CLIENT_43aa5d8366b5 complain about regarding custody fees?
  [egress] 3689 chars, filtering=tokenized, scanner hits=1 ['PHONE']  BLOCKED
  stopped at the egress gate: 1 sensitive value(s) in outbound payload: ['PHONE']
```

If your run does not block, the retrieved set did not include a note with a phone
number; the hex of your tokens changes the sparse scores slightly. Try another client
listed by `SELECT c.client_name FROM bank.clients c JOIN bank.documents d ON d.body LIKE '%'||c.contact_phone||'%' WHERE d.bank_id = 'bank_a';`

Close the gap (lab_admin), then re-tokenize:

```sql
INSERT INTO gov.sensitive_columns VALUES ('clients', 'contact_phone', 'PHONE', 'phone_token');
CALL gov.apply_labels();
SELECT table_name, column_name, labelled FROM gov.label_coverage WHERE NOT labelled;   -- 0 rows
```
```bash
$PY python/tokenize_corpus.py | tee captures/09b_retokenize.txt     # now with clients.contact_phone PHONE 240 tokens
$PY python/agent.py --dry-run --bank bank_a "$Q" | tee captures/09c_gate_clears.txt
```
```
  [egress] 3691 chars, filtering=tokenized, scanner hits=0
  payload cleared the gate (dry run: not sent)
```

Closing the gap changed the tokenized and the redacted text. `status.py` now reports
those vectors as stale; step 12 rebuilds them. Steps 10 and 11 do not need it.

## Step 10. The live answer, re-identified by the vault

```bash
$PY python/agent.py --bank bank_a --reviewer "$Q" | tee captures/10_live.txt
```
```
question sent : What did CLIENT_43aa5d8366b5 complain about regarding custody fees?
  [egress] 1564 chars, filtering=tokenized, scanner hits=0
  [tool] search_documents({"query": "CLIENT_43aa5d8366b5 custody fees", "k": 5})
  [egress] 3699 chars, filtering=tokenized, scanner hits=0

answer (as the model wrote it):
CLIENT_43aa5d8366b5 complained that the custody fees had risen without notice. [doc 10] ...

answer re-identified through the vault (reviewer only):
Emmental Immobilien AG complained that the custody fees had risen without notice. ...
```

Re-identification on its own, from the vault:

```bash
$PY python/reidentify.py "CLIENT_43aa5d8366b5 met PERSON_65a94b3e0336"     # use your own tokens
```

## Step 11. The egress log and the audit trail

```sql
SELECT egress_id, ts::time(0), filtering, payload_chars, hits, hit_categories, blocked, outcome
FROM gov.egress_log ORDER BY egress_id;
```
```bash
docker logs lab16_pg18 2>&1 | grep AUDIT | grep FUNCTION | tail -5 | tee captures/11_audit.txt
docker logs lab16_vault 2>&1 | grep "reidentifier@vault" | tail -3   # every re-identification, with the tokens asked for
```

What to look at: the naive dry run with 29 hits (`outcome = dry_run`, it never left),
the blocked row with `{PHONE}`, the live rows with 0 hits and `outcome = sent`; one
pgAudit line per agent function call, parameters not logged.

## Step 12. Rebuild the filtered embeddings (background, about 25 minutes)

```bash
nohup $PY python/embed.py --state tokenized redacted > captures/12_reembed.txt 2>&1 &
watch -n 15 $PY python/status.py
```

Two new versions become active; the previous tokenized and redacted versions stay,
inactive. Wait for `stale active vectors: none` and `model job: none running`.

## Step 13. Relevance of retrieval (after step 12)

It needs an active raw, redacted and tokenized version; it stops with a message if one
is missing. Capture tip: `$PY python/measure.py | tee captures/13_measure.txt`.

```bash
$PY python/measure.py | tee captures/13_measure.txt
```
```sql
SELECT state, method, recall, ndcg FROM gov.quality_runs
WHERE run_id > (SELECT max(run_id) - 144 FROM gov.quality_runs)
  AND k = 5 AND question_set = 'entity' ORDER BY state, method;
```

My run, entity questions recall@5: raw hybrid 0.922, raw entity 1.000, redacted hybrid
0.175, tokenized hybrid 0.611, tokenized entity 1.000. Tokenized numbers can move a few
points with your tokens' hex; the entity rows should stay at 1.000.

## Step 14. Relevance of the answers (background, about 20 minutes)

72 agent runs (36 questions, naive and governed). My run used 124,758 prompt and 7,827
completion tokens; check the cost on your OpenAI usage page afterwards.

```bash
nohup $PY python/evaluate_answers.py --label hand_run > captures/14_answers.txt 2>&1 &
watch -n 15 $PY python/status.py      # or: tail -f captures/14_answers.txt
```

The `done` line of a background job appears at your next prompt, not when it ends;
`status.py` is the reliable signal.
```sql
SELECT mode, question_set, cited_recall, cited_precision, context_recall, explicit_entity, blocked
FROM gov.answer_runs WHERE label = 'hand_run' ORDER BY mode, question_set;

-- what this run actually sent: only outcome = 'sent' left the perimeter
SELECT filtering, outcome, count(*) AS requests, sum(hits) AS sensitive_values
FROM gov.egress_log WHERE run_label = 'hand_run'
GROUP BY filtering, outcome ORDER BY 1, 2;
```

Two runs on 2026-10-01 (old question wording): cited recall on entity questions naive
0.667 / governed 0.803, then naive 0.706 / governed 0.756. Naive sent 1,114 and 1,267
sensitive values; governed 0 both times. Answers vary run to run; compare the direction,
not one gap. `explicit_entity` counts answers that name the asked entity (token or
name); "the client" does not count.

## Step 15. Embedding versions and stale vectors

```sql
SELECT v.version_id, v.state, v.dense_model, v.dims, v.key_version, v.is_active, count(e.*) AS rows
FROM bank.embedding_versions v LEFT JOIN bank.embeddings e USING (version_id)
GROUP BY v.version_id ORDER BY v.version_id;
```

```sql
SELECT state, count(*) FILTER (WHERE NOT fresh) AS stale, count(*) AS active_vectors
FROM gov.embedding_staleness GROUP BY state;
```

Two tokenized and two redacted versions: the first of each inactive (built before the
gap was closed), the second active, and zero stale vectors. The inactive versions are
**not** a complete rollback: re-tokenizing overwrote the text and the mention index they
were built from (see "Rollback limit" in the README).

## Step 16. Lab control checklist

```sql
SELECT level, check_name, passed FROM gov.maturity_checks ORDER BY level, check_name;
SELECT * FROM gov.maturity;
```

Expected at the end: level 5, 12 of 12. It is a checklist of the controls this lab
implements, not an AI-maturity assessment: it cannot see what nobody labelled (the nine
planted names). Between steps 9 and 12 it stops at level 2, because of the stale
vectors; with the gap open it stops at level 1.

## Step 17. Query plans

```bash
psql -h localhost -p 5437 -U app_agent -d bank -X -f sql/demo/9_plans.sql | tee captures/16_plans.txt
```

What to look at: no HNSW at this size, a bitmap scan on `(version_id, bank_id)` with the
RLS predicate inside `Recheck Cond`, top-N heapsort over 508 rows, about 3 ms dense and
under 1 ms sparse.

## Step 18. Trace the agent with Langfuse, then scan the trace store (about 10 minutes)

Self-hosted Langfuse from its official Docker Compose file (web, worker, PostgreSQL,
ClickHouse, Redis, MinIO; about 3 GB of images). Its own `.env` holds random secrets and a
pre-created project, so no sign-up is needed:

```bash
cd docker/langfuse
r(){ openssl rand -hex $1; }
cat > .env <<EOT
SALT=$(r 16)
ENCRYPTION_KEY=$(r 32)
NEXTAUTH_SECRET=$(r 24)
POSTGRES_PASSWORD=$(r 12)
DATABASE_URL=postgresql://postgres:\${POSTGRES_PASSWORD}@postgres:5432/postgres
CLICKHOUSE_PASSWORD=$(r 12)
REDIS_AUTH=$(r 12)
MINIO_ROOT_PASSWORD=$(r 12)
LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=\${MINIO_ROOT_PASSWORD}
LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY=\${MINIO_ROOT_PASSWORD}
LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY=\${MINIO_ROOT_PASSWORD}
LANGFUSE_INIT_ORG_ID=lab16
LANGFUSE_INIT_ORG_NAME=Lab 16
LANGFUSE_INIT_PROJECT_ID=lab16-governance
LANGFUSE_INIT_PROJECT_NAME=Lab 16 governance
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-lab16-$(r 8)
LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-lab16-$(r 8)
LANGFUSE_INIT_USER_EMAIL=lab@example.com
LANGFUSE_INIT_USER_NAME=Lab
LANGFUSE_INIT_USER_PASSWORD=$(r 10)
TELEMETRY_ENABLED=false
EOT
docker compose -p lab16-langfuse up -d
curl -s http://localhost:3000/api/public/health      # {"status":"OK",...}
cd ../..
```

Then add to the lab `.env` (tracing is on as soon as the public key is set, off without it):

```bash
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_PUBLIC_KEY=<LANGFUSE_INIT_PROJECT_PUBLIC_KEY from docker/langfuse/.env>
LANGFUSE_SECRET_KEY=<LANGFUSE_INIT_PROJECT_SECRET_KEY>
LANGFUSE_CLICKHOUSE_URL=http://localhost:8123
LANGFUSE_CLICKHOUSE_USER=clickhouse
LANGFUSE_CLICKHOUSE_PASSWORD=<CLICKHOUSE_PASSWORD>
LANGFUSE_S3_URL=http://localhost:9090
LANGFUSE_S3_USER=minio
LANGFUSE_S3_PASSWORD=<MINIO_ROOT_PASSWORD>
```

Run the answers again with tracing on (`pip install "langfuse>=4.16,<5" boto3` first), then scan
both trace stores with the egress scanner:

```bash
$PY python/evaluate_answers.py --label langfuse | tee captures/17_answers_langfuse.txt
$PY python/scan_traces.py --label langfuse | tee captures/18_scan_traces.txt
```

What to look at: the naive traces hold more sensitive values than the naive requests sent
(each retrieved document is recorded as the search result and again in every later model
call), once in ClickHouse and once in object storage; the governed traces hold none. The
traces themselves are at http://localhost:3000 (user and password from `docker/langfuse/.env`),
and the two SQL queries of the post run in ClickHouse:

```bash
docker exec -it lab16-langfuse-clickhouse-1 clickhouse-client --user clickhouse --password <CLICKHOUSE_PASSWORD>
```

## Step 19. The fair comparisons and the checks behind the post (about 15 minutes)

Same search, same tools, raw text against tokens (the post's answer table), then the three
setups together (the naive reference and the trace scan):

```bash
$PY python/evaluate_answers.py --label fair2 --modes raw_entity,tokenized_search | tee captures/27_answers_equal_tools.txt
$PY python/scan_traces.py --label fair2 | tee -a captures/27_answers_equal_tools.txt
$PY python/evaluate_answers.py --label fair --modes off,raw_entity,tokenized | tee captures/23_answers_fair.txt
$PY python/scan_traces.py --label fair | tee captures/25_scan_traces_fair.txt
```

`raw_entity`: raw text, but the governed path's entity-aware search as `app_agent`, same
candidate limits, search tool only. `tokenized_search`: the governed path with the search
tool only. What to look at: context recall 1.000 for both, cited recall within run-to-run
spread, and detected sensitive values sent only by the raw arm.

The entity-first lookup on its own, and repeated timings of the three retrieval plans:

```bash
psql -h localhost -p 5437 -U app_agent -d bank -X <<'EOT' | tee captures/30_entity_plan.txt
BEGIN;
SELECT set_config('app.bank_id', 'bank_a', true);
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
SELECT m.doc_id FROM bank.document_mentions m
WHERE m.token = ANY(ARRAY['CLIENT_43aa5d8366b5'])
GROUP BY m.doc_id HAVING count(DISTINCT m.token) = 1;
COMMIT;
EOT
for i in 1 2 3 4 5; do
  psql -h localhost -p 5437 -U app_agent -d bank -X -f sql/demo/9_plans.sql \
    | grep "Execution Time" | awk '{printf "%s ", $3}'; echo
done | tee captures/31_plan_timings.txt     # dense, sparse, hybrid in ms
```

The token `CLIENT_43aa5d8366b5` is from the lab's vault key; take any `client_token` from yours.

PostgreSQL behaviour and storage the post relies on:

```bash
psql -h localhost -p 5437 -U lab_admin -d bank -X \
  -c "SELECT current_setting('app.bank_id', true) IS NULL AS null_when_never_set" \
  -c "BEGIN" -c "SELECT set_config('app.bank_id', 'bank_a', true)" -c "COMMIT" \
  -c "SELECT quote_literal(current_setting('app.bank_id', true)) AS value_after_txn"   # '' not NULL
psql -h localhost -p 5437 -U lab_admin -d bank -Atc \
  "SELECT attname, attstorage FROM pg_attribute WHERE attrelid = 'bank.embeddings'::regclass AND attname IN ('dense','sparse')" -c \
  "SELECT pg_size_pretty(pg_relation_size('bank.embeddings')), pg_size_pretty(pg_relation_size(reltoastrelid)) FROM pg_class WHERE oid = 'bank.embeddings'::regclass"
```

## Replay the gap

```bash
psql -h localhost -p 5437 -U lab_admin -d bank -X -f sql/demo/1c_reopen_gap.sql
$PY python/tokenize_corpus.py
nohup $PY python/embed.py --state tokenized redacted > captures/replay_reembed.txt 2>&1 &
```

## Optional: key rotation

```bash
PGPASSWORD='vault2026!' psql -h localhost -p 5438 -U vault_admin -d vault -c "SELECT vault.rotate_key()"
$PY python/tokenize_corpus.py                 # every token changes, key version 2
nohup $PY python/embed.py --state tokenized > captures/rotation_embed.txt 2>&1 &
```

Then step 15 shows a tokenized version with `key_version = 2`. Not run on 2026-10-01:
an unvalidated procedure. Rebuild the redacted state too if you want every active set fresh.
