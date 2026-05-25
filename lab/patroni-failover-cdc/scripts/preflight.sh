#!/usr/bin/env bash
# Inspect the running cluster for the PG17+ failover-slot prerequisites and
# surface the two classic traps: a missing dbname in primary_conninfo (silent),
# and an unset/stale synchronized_standby_slots.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

L="$(leader_name)"; S="$(standby_name)"
echo "leader=${L}  standby=${S}"
echo

get() { psql_node "$1" postgres -tAc "SHOW $2;" 2>/dev/null | tr -d '[:space:]'; }
row() { printf '  %-30s %s\n' "$1" "$2"; }

echo "== leader (${L}) =="
row "wal_level"                  "$(get "${L}" wal_level)"
row "synchronized_standby_slots" "$(get "${L}" synchronized_standby_slots)"

echo
echo "== standby (${S}) =="
row "wal_level"              "$(get "${S}" wal_level)"
row "hot_standby_feedback"   "$(get "${S}" hot_standby_feedback)"
row "sync_replication_slots" "$(get "${S}" sync_replication_slots)"
PCI="$(psql_node "${S}" postgres -tAc 'SHOW primary_conninfo;' 2>/dev/null)"
row "primary_conninfo"       "${PCI}"
case "${PCI}" in
  *dbname=*) echo "  [PASS] primary_conninfo carries dbname (the slot-sync worker can connect)";;
  *)         echo "  [WARN] NO dbname in primary_conninfo -> slot sync SILENTLY does nothing";;
esac

echo
EXP="$(slot_of_member "${S}")"
SSS="$(get "${L}" synchronized_standby_slots)"
if   [ -z "${SSS}" ];      then echo "[WARN] synchronized_standby_slots empty -> run ./scripts/enable-failover-config.sh (expect '${EXP}')"
elif [ "${SSS}" = "${EXP}" ]; then echo "[PASS] synchronized_standby_slots=${SSS} matches the current standby slot"
else echo "[WARN] synchronized_standby_slots=${SSS} but current standby slot is '${EXP}' (stale after a switchover — re-run enable-failover-config.sh)"
fi
