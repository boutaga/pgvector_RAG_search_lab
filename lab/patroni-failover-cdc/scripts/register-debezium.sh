#!/usr/bin/env bash
# Register a Debezium PostgreSQL connector. Two scenarios:
#   01 -> plain slot   (slot.failover=false) : breaks after a Patroni switchover
#   02 -> failover slot (slot.failover=true) : survives a Patroni switchover
# The connector talks to HAProxy :5000, so it always targets the current leader.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SCN="${1:-02}"
case "${SCN}" in
  01) NAME="orders-plain";    SLOT="dbz_slot"; FAILOVER="false";;
  02) NAME="orders-failover"; SLOT="${CDC_SLOT:-cdc_slot}"; FAILOVER="true";;
  *)  echo "usage: $0 [01|02]"; exit 2;;
esac

echo "[dbz] registering '${NAME}'  slot=${SLOT}  slot.failover=${FAILOVER}"

CFG=$(cat <<JSON
{
  "name": "${NAME}",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "tasks.max": "1",
    "database.hostname": "haproxy",
    "database.port": "5000",
    "database.user": "postgres",
    "database.password": "${POSTGRES_PASSWORD}",
    "database.dbname": "${DEMO_DB}",
    "topic.prefix": "pfs",
    "table.include.list": "public.orders",
    "plugin.name": "pgoutput",
    "slot.name": "${SLOT}",
    "slot.drop.on.stop": "false",
    "slot.failover": "${FAILOVER}",
    "publication.name": "${CDC_PUBLICATION:-cdc_pub}",
    "publication.autocreate.mode": "disabled",
    "snapshot.mode": "initial",
    "tombstones.on.delete": "false"
  }
}
JSON
)

printf '%s' "${CFG}" | in_net curl -sS -X POST \
  -H "Content-Type: application/json" --data @- \
  http://connect:8083/connectors -w '\nHTTP %{http_code}\n'

echo
echo "[dbz] check it:  ./scripts/connect-status.sh ${NAME}"
