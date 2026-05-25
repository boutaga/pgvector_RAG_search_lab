#!/usr/bin/env bash
# Show replication slots on BOTH nodes with the failover-relevant columns.
# The money shot: before/after a switchover, watch `failover` / `synced` /
# `confirmed_flush_lsn` on each node.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

L="$(leader_name)"; S="$(standby_name)"
echo "leader=${L}  standby=${S}"
echo

SQL="SELECT slot_name, slot_type, active, failover, synced, temporary,
            restart_lsn, confirmed_flush_lsn
       FROM pg_replication_slots
      ORDER BY slot_type, slot_name;"

for node in patroni1 patroni2; do
  role="replica"; [ "${node}" = "${L}" ] && role="LEADER"
  rec="$(psql_node "${node}" postgres -tAc 'SELECT pg_is_in_recovery();' 2>/dev/null || echo '?')"
  echo "===== ${node} (${role}) — pg_is_in_recovery=${rec} ====="
  psql_node "${node}" "${DEMO_DB}" -c "${SQL}" 2>/dev/null || echo "  (unreachable)"
  echo
done
