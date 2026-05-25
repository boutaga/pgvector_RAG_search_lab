#!/usr/bin/env bash
# Show Kafka Connect / Debezium connector + task state. With no arg, lists
# connectors. After a switchover, scenario 01 shows the task FAILED with a
# "replication slot ... does not exist" trace here.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

NAME="${1:-}"
if [ -z "${NAME}" ]; then
  echo "connectors:"
  in_net sh -lc 'curl -fsS http://connect:8083/connectors | jq .' 2>/dev/null \
    || in_net curl -fsS http://connect:8083/connectors
  echo
  echo "usage: $0 <connector-name>   # detailed status + task trace"
  exit 0
fi

in_net sh -lc "curl -fsS http://connect:8083/connectors/${NAME}/status | jq ." 2>/dev/null \
  || in_net curl -fsS "http://connect:8083/connectors/${NAME}/status"
