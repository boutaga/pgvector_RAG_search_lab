#!/bin/bash
set -euo pipefail

echo "[entrypoint] node=${PATRONI_NAME} scope=${PATRONI_SCOPE} pg=${PG_MAJOR}"

# Render the Patroni config from the template (all ${VARS} come from the env).
envsubst < /etc/patroni/patroni.yml.template > /etc/patroni/patroni.yml

mkdir -p /var/run/postgresql
chmod 0775 /var/run/postgresql || true

echo "[entrypoint] waiting for etcd ..."
for _ in $(seq 1 30); do
    if curl -fsS http://etcd:2379/health >/dev/null 2>&1; then
        echo "[entrypoint] etcd is reachable"
        break
    fi
    sleep 2
done

# 'patroni' resolves to the venv binary (PATH is venv-first), which has psycopg2.
exec patroni /etc/patroni/patroni.yml
