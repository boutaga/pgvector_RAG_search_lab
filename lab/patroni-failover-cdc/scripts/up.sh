#!/usr/bin/env bash
# Build + start the cluster, wait for a leader, apply the demo schema.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

cd "${REPO_ROOT}"
[ -f .env ] || { cp .env.example .env; echo "[up] created .env from .env.example"; }

IMG="pfs-patroni:pg${PG_MAJOR:-18}"
if docker image inspect "${IMG}" >/dev/null 2>&1; then
  echo "[up] image ${IMG} already present, skipping build"
  echo "     (force a rebuild with: docker compose -f docker/docker-compose.yml build)"
else
  echo "[up] building images ..."
  dc build
fi

echo "[up] starting the stack ..."
dc up -d

echo "[up] waiting for a Patroni leader to be elected ..."
L=""
for _ in $(seq 1 60); do
  L="$(leader_name || true)"
  [ -n "${L}" ] && { echo "[up] leader: ${L}"; break; }
  sleep 3
done
[ -n "${L}" ] || { echo "[up] ERROR: no leader after timeout"; dc ps; exit 1; }

echo "[up] waiting for the leader (${L}) to be read-write ..."
for _ in $(seq 1 30); do
  [ "$(psql_node "${L}" postgres -tAc 'SELECT NOT pg_is_in_recovery();' 2>/dev/null | tr -d '[:space:]')" = "t" ] && break
  sleep 2
done

echo "[up] applying sql/00-setup.sql on the leader (${L}) ..."
psql_node "${L}" postgres -f - < sql/00-setup.sql

echo "[up] waiting for Kafka Connect REST ..."
for _ in $(seq 1 60); do
  in_net curl -fsS http://connect:8083/ >/dev/null 2>&1 && { echo "[up] connect is up"; break; }
  sleep 3
done

cat <<'EOF'

[up] ready. Suggested flow:
  ./scripts/patronictl.sh list                  # see leader + replica
  ./scripts/preflight.sh                         # check the failover-slot prerequisites
  ./scripts/enable-failover-config.sh            # set synchronized_standby_slots on the leader
  ./scripts/workload.sh                          # (own terminal) continuous monotonic INSERTs
  ./scripts/register-debezium.sh 02              # 01 = plain slot, 02 = failover slot
  ./scripts/slots.sh                             # failover / synced / confirmed_flush_lsn on both nodes
  ./scripts/consume.sh                           # (own terminal) watch the CDC topic
  ./scripts/switchover.sh                        # flip primary <-> replica
EOF
