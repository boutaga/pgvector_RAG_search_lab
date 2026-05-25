# Patroni + PostgreSQL node image for the failover-slots lab.
# Unlike worklab-PG-Patroni-troubleshoot (which omits psycopg2 on purpose to
# reproduce a bug), this image installs psycopg2 in Patroni's venv — a clean,
# production-shaped layout.
FROM ubuntu:22.04

ARG PG_MAJOR=18
ARG PATRONI_VERSION=4.1.3
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg2 lsb-release \
        python3 python3-pip python3-venv \
        gettext-base jq vim less iproute2 \
    && rm -rf /var/lib/apt/lists/*

# PostgreSQL APT repository (PGDG)
RUN curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
        | gpg --dearmor -o /usr/share/keyrings/postgresql-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/postgresql-keyring.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        postgresql-${PG_MAJOR} postgresql-client-${PG_MAJOR} \
    && rm -rf /var/lib/apt/lists/*

# The distro auto-creates a cluster on install; Patroni bootstraps its own.
RUN pg_dropcluster ${PG_MAJOR} main --stop || true

# Patroni in an isolated venv, WITH psycopg2 (the common production layout).
RUN python3 -m venv /opt/patroni-venv \
    && /opt/patroni-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/patroni-venv/bin/pip install --no-cache-dir \
        "patroni[etcd3]==${PATRONI_VERSION}" psycopg2-binary

ENV PATH="/opt/patroni-venv/bin:/usr/lib/postgresql/${PG_MAJOR}/bin:${PATH}"
ENV PGDATA=/var/lib/postgresql/data
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PG_MAJOR=${PG_MAJOR}

RUN mkdir -p /var/lib/postgresql/data /etc/patroni /var/run/postgresql \
    && chown -R postgres:postgres /var/lib/postgresql /etc/patroni /var/run/postgresql \
    && chmod 700 /var/lib/postgresql/data

COPY --chown=postgres:postgres patroni.yml.template /etc/patroni/patroni.yml.template
COPY --chown=root:root entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5432 8008
USER postgres
WORKDIR /var/lib/postgresql
ENTRYPOINT ["/entrypoint.sh"]
