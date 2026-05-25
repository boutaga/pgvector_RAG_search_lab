-- Run against the 'postgres' database on the current leader (scripts/up.sh does this).
-- Creates the demo database, the CDC table, and the publication Debezium consumes.

-- 1) Create the demo database if it does not already exist (idempotent).
SELECT 'CREATE DATABASE cdcdemo'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'cdcdemo')
\gexec

\connect cdcdemo

-- 2) CDC table. `n` is a strictly increasing counter used to detect gaps or
--    duplicates in the Kafka stream across a failover. `source_node` records
--    which primary produced the row (its container IP), so the switchover is
--    visible in the data itself.
CREATE TABLE IF NOT EXISTS public.orders (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    n           bigint      NOT NULL,
    source_node text        NOT NULL DEFAULT '',
    note        text,
    ts          timestamptz NOT NULL DEFAULT clock_timestamp()
);

-- FULL replica identity so UPDATE/DELETE carry complete before-images (the demo
-- is insert-heavy, but this keeps the lab honest if you extend it).
ALTER TABLE public.orders REPLICA IDENTITY FULL;

-- The monotonic counter behind `n`.
CREATE SEQUENCE IF NOT EXISTS public.orders_n_seq;

-- 3) Publication consumed by the Debezium pgoutput connector.
--    Pre-created here so the connector runs with publication.autocreate.mode=disabled.
DROP PUBLICATION IF EXISTS cdc_pub;
CREATE PUBLICATION cdc_pub FOR TABLE public.orders;

\echo 'setup complete: database cdcdemo, table public.orders, publication cdc_pub'
