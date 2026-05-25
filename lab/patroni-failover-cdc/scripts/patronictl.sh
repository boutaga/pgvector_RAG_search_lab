#!/usr/bin/env bash
# Thin wrapper around patronictl inside the cluster.
#   ./scripts/patronictl.sh list
#   ./scripts/patronictl.sh switchover --candidate patroni2 --force
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# `dc` is a shell function, so it cannot be `exec`-ed. Allocate a TTY when attached
# to a terminal, plain pipe otherwise (so output stays capturable in scripts/CI).
if [ -t 1 ]; then TTYFLAG=-it; else TTYFLAG=-T; fi
dc exec "${TTYFLAG}" patroni1 patronictl -c /etc/patroni/patroni.yml "$@"
