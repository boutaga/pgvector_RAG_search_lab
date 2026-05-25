#!/usr/bin/env bash
# Continuous monotonic INSERT workload against the current leader (via HAProxy).
# Each row gets a strictly increasing n and records the serving node's IP, so the
# switchover is visible in the data. Ctrl-C to stop.
#   ./scripts/workload.sh [interval_seconds]   (default 0.5)
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INTERVAL="${1:-0.5}"
echo "[workload] INSERT into ${DEMO_DB}.orders every ${INTERVAL}s via HAProxy :5000 (Ctrl-C to stop)"

ok=0
while true; do
  if dc exec -T -e PGPASSWORD="${POSTGRES_PASSWORD}" patroni1 \
        psql "host=haproxy port=5000 user=postgres dbname=${DEMO_DB}" \
        -q -v ON_ERROR_STOP=0 -f - < "${REPO_ROOT}/sql/10-workload.sql" >/dev/null 2>&1; then
    ok=$((ok+1))
    printf '\r[workload] inserts ok: %d ' "${ok}"
  else
    echo
    echo "[workload] insert failed at $(date +%T) — failover in progress? retrying ..."
  fi
  sleep "${INTERVAL}"
done
