-- 9_plans.sql - query plans for the retrieval paths (for the post)
-- Run as app_agent. Uses a stored document vector as the query vector.
\set ECHO queries
BEGIN;
SELECT set_config('app.bank_id', 'bank_a', true);
SELECT dense AS qd, sparse AS qs FROM bank.embeddings e
JOIN bank.embedding_versions v USING (version_id)
WHERE v.state = 'tokenized' AND v.is_active ORDER BY doc_id LIMIT 1 \gset

SET LOCAL hnsw.iterative_scan = 'strict_order';
SET LOCAL hnsw.ef_search = 100;

-- dense path
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
SELECT e.doc_id FROM bank.embeddings e
JOIN bank.embedding_versions v USING (version_id)
WHERE v.state = 'tokenized' AND v.is_active
ORDER BY e.dense <=> :'qd'::vector LIMIT 10;

-- sparse path
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
SELECT e.doc_id FROM bank.embeddings e
JOIN bank.embedding_versions v USING (version_id)
WHERE v.state = 'tokenized' AND v.is_active
ORDER BY e.sparse <#> :'qs'::sparsevec LIMIT 10;

-- hybrid (the function the agent calls)
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
SELECT * FROM bank.retrieve(:'qd'::vector, :'qs'::sparsevec, 'tokenized', 'hybrid', 10);
COMMIT;
