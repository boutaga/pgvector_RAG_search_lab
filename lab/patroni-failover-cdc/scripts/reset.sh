#!/usr/bin/env bash
# Reset the demo for a fresh scenario run, without a full teardown:
#   - delete Debezium connectors
#   - drop the CDC logical slots (physical member slots untouched)
#   - delete the CDC Kafka topic
#   - recreate the schema and truncate the table (n restarts at 1)
# Keeps the cluster running.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo "[reset] deleting Debezium connectors ..."
for c in $(in_net curl -fsS http://connect:8083/connectors 2>/dev/null | tr -d '[]"' | tr ',' ' '); do
  [ -n "$c" ] || continue
  echo "  - ${c}"
  in_net curl -fsS -X DELETE "http://connect:8083/connectors/${c}" >/dev/null || true
done

# Let the walsender release the slot after the connector is gone.
sleep 2

echo "[reset] dropping CDC replication slots on both nodes (member slots untouched) ..."
for node in patroni1 patroni2; do
  psql_node "${node}" postgres -tAc \
    "SELECT pg_drop_replication_slot(slot_name)
       FROM pg_replication_slots
      WHERE slot_name IN ('cdc_slot','dbz_slot') AND NOT active;" 2>/dev/null || true
done

echo "[reset] deleting CDC Kafka topic ..."
dc exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092 \
  --delete --topic pfs.public.orders 2>/dev/null && echo "  - pfs.public.orders" || echo "  - (no topic to delete)"

L="$(leader_name)"
echo "[reset] recreating demo objects on the leader (${L}) ..."
psql_node "${L}" postgres -f - < "${REPO_ROOT}/sql/00-setup.sql"
psql_node "${L}" "${DEMO_DB}" -tAc "TRUNCATE public.orders; ALTER SEQUENCE public.orders_n_seq RESTART WITH 1;"
echo "[reset] done."
