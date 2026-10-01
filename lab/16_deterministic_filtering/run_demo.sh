#!/usr/bin/env bash
# =============================================================================
# Lab 16 - walk the governance layers in order, pausing between beats.
#   ./run_demo.sh            full walk (assumes containers up, data built)
#   ./run_demo.sh --build    first: start containers, generate, tokenize, embed (~35 min on CPU)
#   ./run_demo.sh --auto     no pauses
# Needs: docker, psql client, the Python venv (VENV, default ~/venvs/lab16), .env filled.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
VENV=${VENV:-$HOME/venvs/lab16}
PY="$VENV/bin/python"
AUTO=0; BUILD=0
for a in "$@"; do
  case $a in --auto) AUTO=1 ;; --build) BUILD=1 ;; *) echo "unknown flag $a"; exit 1 ;; esac
done
export PGPASSWORD='dbi2026!'
ADMIN="psql -h localhost -p 5437 -U lab_admin -d bank -X"
AGENT="psql -h localhost -p 5437 -U app_agent -d bank -X"
Q_CLIENT=$($PY -c "import json;q=json.load(open('data/questions.json'));print(next(x['question'] for x in q if x['bank_id']=='bank_a' and x['question'].startswith('What did')))")

beat() { echo; echo "=== $1"; [ $AUTO -eq 1 ] || read -rp "--- press Enter ---" _; }

if [ $BUILD -eq 1 ]; then
  beat "Build: containers, synthetic bank, tokens, local embeddings"
  (cd docker && docker compose up -d --build)
  until $ADMIN -Atc "select 1" >/dev/null 2>&1; do sleep 2; done
  $PY data/generate_bank.py
  $PY python/tokenize_corpus.py
  $PY python/embed.py --state all
fi

beat "Layer 1, row-level security (as app_agent)"
$AGENT -f sql/demo/0_rls.sql

beat "Layer 2, labelling and coverage: contact_phone is a gap"
$ADMIN -f sql/demo/1_labels.sql

beat "Layer 3, deterministic tokens"
$ADMIN -f sql/demo/2_tokens.sql

beat "The vault is a separate server"
$ADMIN -f sql/demo/3_vault.sql
PGPASSWORD='vault2026!' psql -h localhost -p 5438 -U tokenizer -d vault -X -c "SELECT * FROM vault.keys" || true

beat "The leak: naive agent, raw text, egress logged but not blocked"
$PY python/agent.py --bank bank_a --filtering off "$Q_CLIENT"

beat "Filtered agent: tokens only, the egress gate blocks on any hit"
$PY python/agent.py --bank bank_a --reviewer "$Q_CLIENT" || true

beat "If the gate blocked on PHONE: close the labelling gap, re-tokenize, run again"
$ADMIN -f sql/demo/1b_close_gap.sql
$PY python/tokenize_corpus.py
$PY python/agent.py --bank bank_a --reviewer "$Q_CLIENT"

beat "Egress log: what left, with which filtering"
$ADMIN -f sql/demo/4_egress.sql

beat "Audit: the agent's tool calls in the pgAudit log"
$ADMIN -f sql/demo/5_audit.sql
docker logs lab16_pg18 2>&1 | grep 'AUDIT' | grep -E 'FUNCTION' | tail -8 || true

beat "Relevance: raw versus redacted versus tokenized"
$PY python/measure.py
$ADMIN -f sql/demo/6_quality.sql

beat "Embedding versions"
$ADMIN -f sql/demo/7_versioning.sql

beat "Maturity scorecard"
$ADMIN -f sql/demo/8_maturity.sql

beat "Query plans for the post"
$AGENT -f sql/demo/9_plans.sql

echo; echo "Done. To replay the gap: $ADMIN -f sql/demo/1c_reopen_gap.sql && $PY python/tokenize_corpus.py"
