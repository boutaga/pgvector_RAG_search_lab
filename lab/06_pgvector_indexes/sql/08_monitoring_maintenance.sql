-- =============================================================================
-- Lab 06 — 08_monitoring_maintenance.sql
-- DBA monitoring: index stats, build progress, maintenance, GUC audit
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Index inventory and sizes
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS size,
    indexdef
FROM pg_indexes
WHERE tablename = 'articles'
ORDER BY pg_relation_size(indexname::regclass) DESC;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Index usage statistics
-- ─────────────────────────────────────────────────────────────────────────────
-- Identify unused indexes (candidates for removal).
-- Note: stats accumulate since last pg_stat_reset().

SELECT
    schemaname,
    relname AS table_name,
    indexrelname AS index_name,
    idx_scan AS times_used,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE relname = 'articles'
ORDER BY idx_scan DESC;

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Table statistics
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    relname,
    n_live_tup,
    n_dead_tup,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE relname = 'articles';

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. Index build progress monitoring (PG 12+)
-- ─────────────────────────────────────────────────────────────────────────────
-- Run this in a second psql session WHILE an index is being built.
-- It shows real-time progress for CREATE INDEX operations.

-- Query to run in a separate session during a long index build:
-- SELECT
--     a.pid,
--     a.query,
--     p.phase,
--     p.tuples_done,
--     p.tuples_total,
--     CASE WHEN p.tuples_total > 0
--          THEN ROUND(100.0 * p.tuples_done / p.tuples_total, 1)
--          ELSE 0
--     END AS pct_done
-- FROM pg_stat_progress_create_index p
-- JOIN pg_stat_activity a ON a.pid = p.pid;

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. REINDEX CONCURRENTLY
-- ─────────────────────────────────────────────────────────────────────────────
-- Rebuilds an index without blocking reads/writes.
-- Use after significant data changes or index bloat.

-- NOTE: Must be run outside a transaction block.
-- REINDEX INDEX CONCURRENTLY idx_content_hnsw_halfvec;

-- ─────────────────────────────────────────────────────────────────────────────
-- 6. VACUUM ANALYZE
-- ─────────────────────────────────────────────────────────────────────────────
-- After bulk data loading or updates, always run VACUUM ANALYZE.
-- This updates planner statistics and reclaims dead tuple space.

VACUUM ANALYZE articles;

-- Verify stats are fresh
SELECT
    relname,
    last_vacuum,
    last_analyze,
    n_live_tup
FROM pg_stat_user_tables
WHERE relname = 'articles';

-- ─────────────────────────────────────────────────────────────────────────────
-- 7. GUC audit — all pgvector/pgvectorscale settings
-- ─────────────────────────────────────────────────────────────────────────────

SELECT name, setting, unit, short_desc
FROM pg_settings
WHERE name LIKE 'hnsw.%'
   OR name LIKE 'ivfflat.%'
   OR name LIKE 'diskann.%'
ORDER BY name;

-- ─────────────────────────────────────────────────────────────────────────────
-- 8. Table and index bloat check
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    relname,
    pg_size_pretty(pg_total_relation_size(oid)) AS total_size,
    pg_size_pretty(pg_relation_size(oid)) AS table_size,
    pg_size_pretty(pg_indexes_size(oid)) AS indexes_size,
    pg_size_pretty(pg_total_relation_size(oid) - pg_relation_size(oid) - pg_indexes_size(oid)) AS toast_size
FROM pg_class
WHERE relname = 'articles';

-- ─────────────────────────────────────────────────────────────────────────────
-- 9. Access method capabilities
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    am.amname AS access_method,
    opc.opcname AS operator_class,
    t.typname AS data_type
FROM pg_opclass opc
JOIN pg_am am ON am.oid = opc.opcmethod
JOIN pg_type t ON t.oid = opc.opcintype
WHERE am.amname IN ('hnsw', 'ivfflat', 'diskann')
ORDER BY am.amname, t.typname, opc.opcname;
