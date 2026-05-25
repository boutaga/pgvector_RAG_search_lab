#!/usr/bin/env bash
# Set synchronized_standby_slots on the CURRENT leader to the current standby's
# physical member slot. Patroni does not manage this parameter (issue #3431),
# and its correct value flips after a switchover — so this is role-aware and
# meant to be re-run after every role change.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

L="$(leader_name)"; S="$(standby_name)"
SLOT="$(slot_of_member "${S}")"
echo "[cfg] leader=${L}  standby=${S}  ->  synchronized_standby_slots='${SLOT}'"

# Not in Patroni's managed parameter set, so ALTER SYSTEM (postgresql.auto.conf)
# sticks and a reload is enough (sighup parameter).
psql_node "${L}" postgres -v ON_ERROR_STOP=1 <<SQL
ALTER SYSTEM SET synchronized_standby_slots = '${SLOT}';
SELECT pg_reload_conf();
SQL

# Read it back on a FRESH connection. The same session that calls pg_reload_conf()
# can still report the pre-reload value (it processes the SIGHUP between commands),
# so a new connection is what actually reflects the change.
echo -n "[cfg] synchronized_standby_slots now = "
psql_node "${L}" postgres -tAc "SHOW synchronized_standby_slots;"
echo "[cfg] done. Re-run after a switchover (the correct slot name flips to the new standby)."
