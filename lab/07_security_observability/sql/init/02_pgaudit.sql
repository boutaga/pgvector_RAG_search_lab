-- 02_pgaudit.sql — audit configuration (fuller pgaudit angle)
-- =============================================================================
-- shared_preload_libraries=pgaudit is set on the server command line; session
-- logging classes are set there too. Here we add database-scoped settings and
-- the object-audit pattern, so the talk can show:
--   1. Session auditing: the leaking SELECT is logged as authorized access.
--   2. Object auditing: every read of the sensitive tables is logged via the
--      `auditor` role (pgaudit.role), independent of who runs the query.
-- The point on stage: pgaudit's trail stays green while the data leaks, because
-- the leak is an authorization gap, not an unauthorized access.
-- =============================================================================

ALTER DATABASE secobs SET pgaudit.log = 'read, write, ddl, role';
ALTER DATABASE secobs SET pgaudit.log_catalog = off;
ALTER DATABASE secobs SET pgaudit.log_parameter = on;
ALTER DATABASE secobs SET pgaudit.log_relation = on;

-- Object-level auditing: grant the audited privilege to a dedicated role.
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'auditor') THEN
    CREATE ROLE auditor NOLOGIN;
  END IF;
END$$;
GRANT SELECT ON app.documents, app.embeddings TO auditor;
ALTER DATABASE secobs SET pgaudit.role = 'auditor';
