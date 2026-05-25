#!/usr/bin/env bash
# Interactive psql against the current leader (routed through HAProxy :5000).
#   ./scripts/psql-leader.sh [dbname]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DB="${1:-${DEMO_DB}}"
if [ -t 1 ]; then TTYFLAG=-it; else TTYFLAG=-T; fi
dc exec "${TTYFLAG}" -e PGPASSWORD="${POSTGRES_PASSWORD}" patroni1 \
  psql "host=haproxy port=5000 user=postgres dbname=${DB}"
