-- 3_vault.sql - layer 5, the vault is a separate server
-- Part 1, on the bank (port 5437), as lab_admin: no path to the vault exists.
\set ECHO queries
SELECT count(*) AS foreign_servers FROM pg_foreign_server;
SELECT extname FROM pg_extension WHERE extname IN ('postgres_fdw', 'dblink');
-- The agent's role cannot read raw columns, even here.
SELECT has_column_privilege('app_agent', 'bank.clients', 'client_name', 'SELECT') AS agent_reads_client_name,
       has_column_privilege('app_agent', 'bank.clients', 'client_token', 'SELECT') AS agent_reads_client_token;

-- Part 2, on the vault (port 5438): run these by hand, they are expected to fail where noted.
--   psql -h localhost -p 5438 -U app_agent -d vault           -> fails: no such role on the vault
--   psql -h localhost -p 5438 -U tokenizer -d vault -c "SELECT * FROM vault.keys"
--                                                             -> permission denied (nobody reads the key)
--   psql -h localhost -p 5438 -U tokenizer -d vault -c "SELECT * FROM vault.tokenize('client', ARRAY['Test AG'])"
--                                                             -> works: tokens computed inside the vault
--   psql -h localhost -p 5438 -U reidentifier -d vault -c "SELECT token, value FROM vault.mapping LIMIT 3"
--                                                             -> works: the only role that reverses tokens
