-- 03_tools.sql - the only doors the agent has into the database
-- =============================================================================
-- The agent never writes SQL. It calls tools, and each tool is one of these
-- functions. They are SECURITY INVOKER: they run as app_agent, so row-level
-- security and column privileges apply inside them. app_agent has no privilege
-- on any raw sensitive column, so a tool cannot return one even by mistake.
-- =============================================================================

-- Retrieval by state and method, ids and scores only (used by the measurement
-- for all three states, and by the agent's search tool for the tokenized state).
-- hnsw.iterative_scan: with row-level security and a version filter, a plain
-- HNSW scan would stop after ef_search candidates and return too few rows
-- (the filtered-search cliff shown in the Swiss PGDay lab).
CREATE FUNCTION bank.retrieve(p_dense vector, p_sparse sparsevec, p_state text,
                              p_method text DEFAULT 'hybrid', p_k int DEFAULT 10)
RETURNS TABLE (doc_id int, score double precision)
LANGUAGE sql STABLE
SET hnsw.iterative_scan = 'strict_order'
SET hnsw.ef_search = 100
AS $$
WITH v AS (
    SELECT version_id FROM bank.embedding_versions WHERE state = p_state AND is_active
), d AS (
    SELECT e.doc_id, row_number() OVER (ORDER BY e.dense <=> p_dense) AS r
    FROM bank.embeddings e JOIN v USING (version_id)
    ORDER BY e.dense <=> p_dense
    LIMIT 50
), s AS (
    SELECT e.doc_id, row_number() OVER (ORDER BY e.sparse <#> p_sparse) AS r
    FROM bank.embeddings e JOIN v USING (version_id)
    ORDER BY e.sparse <#> p_sparse
    LIMIT 50
), fused AS (
    -- reciprocal rank fusion, k = 60
    SELECT coalesce(d.doc_id, s.doc_id) AS doc_id,
           CASE p_method
             WHEN 'dense'  THEN CASE WHEN d.r IS NULL THEN NULL ELSE 1.0 / (60 + d.r) END
             WHEN 'sparse' THEN CASE WHEN s.r IS NULL THEN NULL ELSE 1.0 / (60 + s.r) END
             ELSE coalesce(1.0 / (60 + d.r), 0) + coalesce(1.0 / (60 + s.r), 0)
           END AS score
    FROM d FULL JOIN s ON d.doc_id = s.doc_id
)
SELECT doc_id, score FROM fused WHERE score IS NOT NULL
ORDER BY score DESC, doc_id
LIMIT p_k
$$;

-- Tool: search the documents, tokenized text only.
CREATE FUNCTION bank.search_documents(p_dense vector, p_sparse sparsevec, p_k int DEFAULT 5)
RETURNS TABLE (doc_id int, doc_type text, created_at date, title text, body text, score double precision)
LANGUAGE sql STABLE
AS $$
SELECT d.doc_id, d.doc_type, d.created_at::date, d.title_tokenized, d.body_tokenized, r.score
FROM bank.retrieve(p_dense, p_sparse, 'tokenized', 'hybrid', p_k) r
JOIN bank.documents d USING (doc_id)
ORDER BY r.score DESC
$$;

-- Tool: profile of one client, by token.
CREATE FUNCTION bank.client_profile(p_client_token text)
RETURNS jsonb
LANGUAGE sql STABLE
AS $$
SELECT jsonb_build_object(
    'client',          c.client_token,
    'client_type',     c.client_type,
    'domicile',        c.domicile,
    'relationship_manager', rm.rm_token,
    'accounts', (SELECT coalesce(jsonb_agg(jsonb_build_object(
                    'iban', a.iban_token, 'currency', a.currency, 'balance', a.balance)
                    ORDER BY a.account_id), '[]')
                 FROM bank.accounts a WHERE a.client_id = c.client_id),
    'documents_mentioning', (SELECT count(*) FROM bank.document_mentions m
                             WHERE m.token = c.client_token))
FROM bank.clients c
JOIN bank.relationship_managers rm ON rm.rm_id = c.rm_id
WHERE c.client_token = p_client_token
$$;

-- Tool: documents that mention one server, by host, fqdn or IP token.
CREATE FUNCTION bank.host_documents(p_host_token text)
RETURNS TABLE (doc_id int, doc_type text, created_at date, title text, server_role text, environment text)
LANGUAGE sql STABLE
AS $$
SELECT DISTINCT d.doc_id, d.doc_type, d.created_at::date, d.title_tokenized, s.role, s.environment
FROM bank.servers s
JOIN bank.document_mentions m ON m.token IN (s.host_token, s.fqdn_token, s.ip_token)
JOIN bank.documents d ON d.doc_id = m.doc_id
WHERE p_host_token IN (s.host_token, s.fqdn_token, s.ip_token)
ORDER BY 3 DESC, 1
$$;
