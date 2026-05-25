#!/usr/bin/env bash
# Trigger a planned Patroni switchover (controlled role swap) to the current
# standby. Syntax verified by running against patronictl 4.1.3:
#   switchover --leader <leader> --candidate <standby> --force <cluster>
# (older Patroni used --master; 4.x renamed it to --leader.)
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

L="$(leader_name)"; S="$(standby_name)"
[ -n "${L}" ] && [ -n "${S}" ] || { echo "[switchover] could not resolve leader/standby"; exit 1; }

echo "[switchover] ${L} (leader)  ->  ${S} (candidate)"
echo "[switchover] before:"
dc exec -T patroni1 patronictl -c /etc/patroni/patroni.yml list

dc exec -T patroni1 patronictl -c /etc/patroni/patroni.yml switchover \
  --leader "${L}" --candidate "${S}" --force "${PATRONI_SCOPE}"

sleep 5
echo "[switchover] after:"
dc exec -T patroni1 patronictl -c /etc/patroni/patroni.yml list

echo
echo "[switchover] NOTE: synchronized_standby_slots must now name the NEW standby (${L})."
echo "             Re-run ./scripts/enable-failover-config.sh to fix it."
