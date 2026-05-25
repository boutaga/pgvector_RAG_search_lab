#!/usr/bin/env bash
# Reconcile the CDC stream against the source table: did every committed row
# reach Kafka? Drains the topic once and compares its distinct keys (n) with the
# orders table on the current leader.
#   missing = 0  -> no loss (duplicates are EXPECTED; CDC is at-least-once)
#   missing > 0  -> data loss (the silent gap a lost slot produces)
# Stop the workload before running this so both sides are stable.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TOPIC="${1:-pfs.public.orders}"
L="$(leader_name)"
TMPK="$(mktemp)"; TMPT="$(mktemp)"
trap 'rm -f "${TMPK}" "${TMPT}"' EXIT

echo "[gaps] leader=${L}  topic=${TOPIC}  (draining ~10s)"
psql_node "${L}" "${DEMO_DB}" -tAc "SELECT n FROM public.orders ORDER BY n;" \
  | tr -d ' \r' | sort -u > "${TMPT}"

RAW=$(dc exec -T kafka /opt/kafka/bin/kafka-console-consumer.sh \
        --bootstrap-server kafka:9092 --topic "${TOPIC}" \
        --from-beginning --timeout-ms 10000 2>/dev/null \
      | tee >(grep -o '"n":[0-9]\+' | sed 's/"n"://' | sort -u > "${TMPK}") | wc -l)

TBL=$(wc -l < "${TMPT}"); KFK=$(wc -l < "${TMPK}"); MISS=$(comm -23 "${TMPT}" "${TMPK}" | wc -l)
echo "table rows (distinct n)         : ${TBL}"
echo "kafka distinct n                : ${KFK}"
echo "kafka raw messages              : ${RAW}  (duplicates: $(( RAW - KFK )))"
echo "MISSING (in table, not in Kafka): ${MISS}"
if [ "${MISS}" -eq 0 ]; then
  echo "RESULT: no loss — every committed row reached Kafka (duplicates expected; CDC is at-least-once)"
else
  echo "RESULT: DATA LOSS — ${MISS} committed rows never reached Kafka (the silent gap)"
fi
