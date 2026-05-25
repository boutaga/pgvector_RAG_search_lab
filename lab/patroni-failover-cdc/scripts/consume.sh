#!/usr/bin/env bash
# Live-stream the CDC topic (Ctrl-C to stop). Topic = <prefix>.<schema>.<table>.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TOPIC="${1:-pfs.public.orders}"
echo "[consume] topic=${TOPIC} (Ctrl-C to stop)"
dc exec -T kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server kafka:9092 \
  --topic "${TOPIC}" \
  --from-beginning
