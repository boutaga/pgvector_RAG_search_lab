# Lab 16 - Deterministic filtering: sensitive data out of AI retrieval, relevance kept

An enterprise database holds client names, staff names, IBANs, hostnames and IP
addresses. This lab shows that it can feed embeddings and an AI agent without any
of those values leaving the perimeter, and measures what that costs in retrieval
quality. It builds on lab 07 (Swiss PGDay security observability): same three
synthetic banks as tenants, same PostgreSQL image, same pooler-safe tenant setting.

Everything is synthetic. Everything runs locally except the agent's chat call to
OpenAI, which only ever receives tokens.

## The layers, in the order they stack

| Layer | What it answers | Where |
|---|---|---|
| 1. Row-level security | Who sees which bank's rows, on documents and embeddings | `sql/bank/02_governance.sql`, demo `0_rls.sql` |
| 2. Labelling | What is sensitive, in which category, and where the gaps are | `gov.sensitive_columns`, `gov.label_coverage`, demo `1_labels.sql` |
| 3. Deterministic tokens | Same value, same token, in documents and questions | `python/tokenize_corpus.py`, demo `2_tokens.sql` |
| 4. Egress gate | What left the perimeter, proved by a scan of every request | `python/egress.py`, `gov.egress_log`, demo `4_egress.sql` |
| 5. Vault | Who may turn tokens back into values, on a separate server | `sql/vault/01_vault.sql`, `python/reidentify.py`, demo `3_vault.sql` |
| 6. Audit | Which tool calls the agent made | pgAudit on `app_agent`, demo `5_audit.sql` |
| Measure | What filtering costs in recall and nDCG | `python/measure.py`, demo `6_quality.sql` |
| Answers | Relevance of the model's answers, naive versus governed | `python/evaluate_answers.py`, `gov.answer_runs` |
| Score | Where the database stands, level 0 to 5 | `gov.maturity`, demo `8_maturity.sql` |

## Architecture

```
                 bank server (lab16_pg18, :5437)                 vault server (lab16_vault, :5438)
  ┌──────────────────────────────────────────────────┐        ┌──────────────────────────────┐
  │ bank.*  raw data + tokenized copies + embeddings │        │ vault.keys     secret key    │
  │ gov.*   registry, labels, egress log, quality,   │        │ vault.mapping  token → value │
  │         maturity scorecard                       │        │ vault.tokenize() computes    │
  │ RLS on every tenant table, column privileges     │        │ tokens inside the vault      │
  └───────────────▲──────────────────────▲───────────┘        └──────▲───────────────▲───────┘
                  │ gateway (pipeline)   │ app_agent (tools only)    │ tokenizer     │ reidentifier
                  │                      │                           │ (execute only)│ (read mapping)
          tokenize_corpus.py, embed.py   agent.py ── egress gate ── OpenAI chat (tokens only)
          (local voyage-4-nano + SPLADE)
```

No foreign server, no `dblink`: the bank has no path to the vault. Nobody reads
the key, not even the tokenizer: tokens are computed inside the vault by a
`SECURITY DEFINER` function. The agent's role has no account on the vault.

## Versions (as run on 2026-10-01)

PostgreSQL 18.3, pgvector 0.8.6, pgAudit 18.0, postgresql_anonymizer 3.2.2.
Dense embeddings: `voyageai/voyage-4-nano` at 1024 dimensions, local, CPU, float32.
Sparse embeddings: `prithivida/Splade_PP_en_v1` through fastembed, local.
Agent: OpenAI `gpt-6-luna` (set in `.env`; tools on Chat Completions need `reasoning_effort="none"`). These are the versions that were run,
not a support statement.

## Setup

```bash
cp .env.example .env            # fill OPENAI_API_KEY
python3 -m venv ~/venvs/lab16   # keep the venv on the Linux filesystem, not /mnt/c
~/venvs/lab16/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/venvs/lab16/bin/pip install -r requirements.txt

cd docker && docker compose up -d --build && cd ..
~/venvs/lab16/bin/python data/generate_bank.py      # 3 banks, 240 clients, 1521 documents, 36 questions
~/venvs/lab16/bin/python python/tokenize_corpus.py  # tokens through the vault, filtered copies
~/venvs/lab16/bin/python python/embed.py            # three states, about 30 min on CPU
```

Then walk the demo: `RUNBOOK.md` runs every step by hand with capture commands,
`./run_demo.sh` runs the beats with pauses, and `walkthrough.sql` holds every step with
the output of the 2026-10-01 run.

Memory: the embedding step needs about 3 GB. The databases run with small
settings (`shared_buffers` 128 MB and 32 MB) so the lab fits in a 6 GB WSL VM.

## Connecting from VS Code

Install the **PostgreSQL** extension by Microsoft (`ms-ossdata.vscode-pgsql`), then
add connections (Windows reaches the containers on `localhost`):

| Connection | Host | Port | Database | User | Password |
|---|---|---|---|---|---|
| Bank, admin | localhost | 5437 | bank | lab_admin | dbi2026! |
| Bank, as the agent (RLS applies) | localhost | 5437 | bank | app_agent | dbi2026! |
| Bank, masked analyst | localhost | 5437 | bank | analyst_masked | dbi2026! |
| Vault, admin | localhost | 5438 | vault | vault_admin | vault2026! |
| Vault, reidentifier | localhost | 5438 | vault | reidentifier | vault2026! |

As `app_agent` or `analyst_masked`, set a bank first or every query returns zero rows:
`BEGIN; SELECT set_config('app.bank_id', 'bank_a', true); ... COMMIT;`

## What the run shows

- **Raw text** carries 7,171 sensitive values across the corpus (scanner count).
- **Tokenized text** carries none from the labelled columns. What remains is the
  deliberate gap: 154 phone numbers from `clients.contact_phone`, which is not in the
  registry. The scanner catches them by pattern and the egress gate blocks. Closing
  the gap (`1b_close_gap.sql`, then re-tokenize) clears it.
- **The honest limit:** 9 planted documents name a lawyer who exists in no table.
  Dictionary filtering only knows what was labelled, and no pattern describes a
  name, so these pass. Listed in `data/planted.json`.
- **Entities, not strings:** a server's hostname and FQDN get the same token, so
  tokenization does not break the link between the two spellings.
- **Relevance of retrieval** (recall@5, 30 entity questions, rebuilt vectors of 2026-10-02):
  raw hybrid 0.956, redacted 0.192, tokenized hybrid 0.650. With entity-aware search
  (`05_entity_retrieval.sql`, which uses the mention index the labels built), tokenized
  reaches 1.000, equal to raw with the same search.
- **Relevance of answers** (runbook step 19, 36 questions, gpt-6-luna, deterministic
  scoring on cited `[doc N]`, runs of 2026-10-03). Same entity-aware search and same
  search tool, raw text against tokens (`--modes raw_entity,tokenized_search`): cited
  recall on entity questions 0.722 raw against 0.719 tokens, context recall 1.000 for
  both, so no measurable cost; detected sensitive values sent 871 in 73 requests raw,
  0 in 74 tokenized. The naive setup (plain hybrid search across all banks) had context
  recall 0.981 and sent 1,134 in 75 requests.
- **Trace store** (`python/scan_traces.py`, Langfuse self-hosted, runbook steps 18 and 19):
  in the three-setup run the naive traces hold 2,336 detected occurrences, in ClickHouse
  and again in object storage; the governed traces none.
- Full step-by-step with real output: `walkthrough.sql`.

## Trust boundaries: what this lab proves and what it does not

- **The vault protects tokenized output, not the bank from its own administrators.**
  The bank database holds the raw data, and its token columns sit next to the raw
  values, so a bank administrator or the pipeline role (`gateway`) can pair a value with
  its token without the vault. That is acceptable here because they already see the raw
  data. What the vault keeps away is the reversal for everything downstream of the
  filter: the agent role, the model provider, logs, anyone holding tokenized text.
  One route stays open in the lab: `app_agent` can read the raw-text embeddings kept for
  the measurement, and a sparse vector partly exposes the words it was built from.
- **The tenant comes from the application.** `app_agent` scopes each transaction with
  `set_config('app.bank_id', ...)`, and the gateway decides the bank from its own input,
  never from the model. Any code holding `app_agent` credentials could still pick
  another bank. Hardening, not done here: one login role per tenant and a policy that
  derives the bank from `current_user`.
- **Network.** The two servers sit on separate Docker networks (`docker-compose.yml`),
  so the bank container cannot reach the vault container directly (on one Docker host the
  published port stays reachable through the host; production needs separate hosts and a
  firewall). Every statement
  of the `reidentifier` role is written to the vault's server log
  (`docker logs lab16_vault`). Purpose limitation of re-identification (who may reverse
  which token, for which request) is not implemented.
- **Egress accounting.** `gov.egress_log.outcome` says what happened to each request:
  only `sent` left the perimeter; `blocked` and `dry_run` never did; `attempted` means
  the process died before the outcome was known. Evaluation runs tag their requests
  with `run_label`. Count leaks with `outcome = 'sent'` and a label, never with `NOT blocked`.
- **Zero findings is a detector result.** The scanner knows labelled values and a few
  patterns. The nine planted names pass it. "Zero sensitive values sent" means zero
  values this detector knows.
- **The scorecard is a lab control checklist**, not an assessment of an organization's
  AI maturity: it checks the controls this lab implements and cannot see what nobody
  labelled.

## Key rotation and embedding versions

Rotating the vault key changes every token, so the tokenized text and its
embeddings must be rebuilt. That is the price of a rotatable secret, and the
reason embeddings are versioned.

```bash
PGPASSWORD='vault2026!' psql -h localhost -p 5438 -U vault_admin -d vault -c "SELECT vault.rotate_key()"
~/venvs/lab16/bin/python python/tokenize_corpus.py          # new tokens, key version 2
~/venvs/lab16/bin/python python/embed.py --state tokenized  # new embedding version, becomes active
psql -h localhost -p 5437 -U lab_admin -d bank -f sql/demo/7_versioning.sql
```

**Rollback limit.** Re-tokenizing overwrites `body_tokenized`, `body_redacted` and
`bank.document_mentions` in place. An older embedding version is therefore only a
valid rollback while the current text still matches it: `gov.embedding_staleness`
compares each active vector's `text_sha256` with the current text, and the checklist
fails on any stale or unverifiable vector. A complete rollback needs a full corpus
revision (text, mentions and vectors built side by side, one serving reference switched
after validation); that is not implemented. Key rotation has not been run end to end.

## Files

- `docker/docker-compose.yml` - the two servers; the bank image is lab 07's Dockerfile.
- `sql/bank/` - schema, governance layers, agent tools, roles and maturity scorecard.
- `sql/vault/01_vault.sql` - key, mapping, `tokenize()`, `rotate_key()`, roles.
- `sql/demo/` - one script per beat, numbered in demo order.
- `data/generate_bank.py` - the seeded generator; `questions.json`, `planted.json`.
- `python/_common.py` - settings, connections, `tenant_cursor()`, embedding helpers.
- `python/filtering.py` - dictionary, tokenize/redact, scanner patterns.
- `python/tokenize_corpus.py` - structured columns through the vault, free text by dictionary.
- `python/embed.py` - versioned local embeddings per state.
- `python/egress.py` - the egress gate. `python/agent.py` - the agent. `python/reidentify.py` - vault lookup.
- `python/measure.py` - recall and nDCG per state, method (dense, sparse, hybrid, entity) and question set.
- `python/scan_corpus.py` - sensitive values the scanner finds per text state.
- `python/status.py` - embedding versions, stale vectors, the running model job.
- `python/evaluate_answers.py` - the 36 questions answered naive and governed, scored on citations.
- `sql/bank/05_entity_retrieval.sql` - entity-aware `bank.retrieve()` and `bank.search_documents()`.
