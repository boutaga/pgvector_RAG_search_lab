-- 5_audit.sql - layer 6, audit: every tool call the agent made
-- pgAudit writes to the server log; read it from the host:
--   docker logs lab16_pg18 2>&1 | grep 'AUDIT' | grep -E 'FUNCTION|READ' | tail -20
-- The setting that makes it happen:
\set ECHO queries
SELECT r.rolname, s.setconfig FROM pg_db_role_setting s JOIN pg_roles r ON r.oid = s.setrole
WHERE r.rolname = 'app_agent';
