#!/usr/bin/env bash
# Shared helpers for the worklab-patroni-failover-slots scripts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_DIR="${REPO_ROOT}/docker"
ENV_FILE="${REPO_ROOT}/.env"
[ -f "${ENV_FILE}" ] || ENV_FILE="${REPO_ROOT}/.env.example"

# Load environment (export everything we read).
set -a
# shellcheck disable=SC1090
. "${ENV_FILE}"
set +a

PATRONI_SCOPE="${PATRONI_SCOPE:-cdc-demo}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
DEMO_DB="${DEMO_DB:-cdcdemo}"

# docker compose wrapper bound to this lab's compose file + env.
dc() {
  docker compose -f "${DOCKER_DIR}/docker-compose.yml" --env-file "${ENV_FILE}" "$@"
}

# psql on a specific node over its local socket (pg_hba: local trust) — no password.
#   psql_node patroni1 cdcdemo -c "SELECT 1"
psql_node() {
  local node="$1" db="$2"; shift 2
  dc exec -T "${node}" psql -U postgres -d "${db}" -v ON_ERROR_STOP=1 "$@"
}

# Run a command inside patroni1 (it ships curl + jq) to reach in-network services.
in_net() { dc exec -T patroni1 "$@"; }

# Current leader / standby Patroni member names (via patronictl JSON + the container's jq).
leader_name() {
  dc exec -T patroni1 sh -lc \
    'patronictl -c /etc/patroni/patroni.yml list -f json | jq -r ".[] | select(.Role==\"Leader\").Member"' \
    2>/dev/null | head -n1
}
standby_name() {
  dc exec -T patroni1 sh -lc \
    'patronictl -c /etc/patroni/patroni.yml list -f json | jq -r ".[] | select(.Role!=\"Leader\").Member"' \
    2>/dev/null | head -n1
}

# Patroni names a member's physical slot after the member, lowercasing and
# replacing every non-alphanumeric char with '_'. patroni1/patroni2 map to themselves.
slot_of_member() { echo "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/_*$//'; }
