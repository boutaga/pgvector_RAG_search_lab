-- 05_entity_retrieval.sql - labelling as the relevance lever
-- =============================================================================
-- The labels already produce bank.document_mentions: which document mentions
-- which entity token. When a question names an entity, method 'entity' ranks
-- the documents that mention ALL the named entities first (hybrid fusion among
-- them), then the rest by the usual hybrid score. Exact, deterministic, and it
-- works for any text state, because the mention index comes from the labels,
-- not from the text the model sees. Row-level security applies to the index too.
-- =============================================================================

DROP FUNCTION IF EXISTS bank.search_documents(vector, sparsevec, int);
DROP FUNCTION IF EXISTS bank.retrieve(vector, sparsevec, text, text, int);

CREATE FUNCTION bank.retrieve(p_dense vector, p_sparse sparsevec, p_state text,
                              p_method text DEFAULT 'hybrid', p_k int DEFAULT 10,
                              p_tokens text[] DEFAULT NULL)
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
), ent AS (
    -- documents mentioning every entity the question names (empty if none named)
    SELECT m.doc_id FROM bank.document_mentions m
    WHERE p_method = 'entity' AND cardinality(p_tokens) > 0 AND m.token = ANY(p_tokens)
    GROUP BY m.doc_id HAVING count(DISTINCT m.token) = cardinality(p_tokens)
), ent_scored AS (
    SELECT e.doc_id,
           1.0 + 1.0 / (60 + row_number() OVER (ORDER BY e.dense <=> p_dense))
               + 1.0 / (60 + row_number() OVER (ORDER BY e.sparse <#> p_sparse)) AS score
    FROM bank.embeddings e JOIN v USING (version_id)
    WHERE e.doc_id IN (SELECT doc_id FROM ent)
), ranked AS (
    SELECT doc_id, score FROM ent_scored
    UNION ALL
    SELECT f.doc_id, f.score FROM fused f
    WHERE f.score IS NOT NULL AND f.doc_id NOT IN (SELECT doc_id FROM ent_scored)
)
SELECT doc_id, score FROM ranked
ORDER BY score DESC, doc_id
LIMIT p_k
$$;

-- Tool: search the documents, tokenized text only, entity-aware when tokens are named.
CREATE FUNCTION bank.search_documents(p_dense vector, p_sparse sparsevec, p_k int DEFAULT 5,
                                      p_tokens text[] DEFAULT NULL)
RETURNS TABLE (doc_id int, doc_type text, created_at date, title text, body text, score double precision)
LANGUAGE sql STABLE
AS $$
SELECT d.doc_id, d.doc_type, d.created_at::date, d.title_tokenized, d.body_tokenized, r.score
FROM bank.retrieve(p_dense, p_sparse, 'tokenized',
                   CASE WHEN cardinality(p_tokens) > 0 THEN 'entity' ELSE 'hybrid' END,
                   p_k, p_tokens) r
JOIN bank.documents d USING (doc_id)
ORDER BY r.score DESC
$$;

REVOKE ALL ON FUNCTION bank.retrieve(vector, sparsevec, text, text, int, text[]),
                       bank.search_documents(vector, sparsevec, int, text[]) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION bank.retrieve(vector, sparsevec, text, text, int, text[]) TO gateway, app_agent;
GRANT EXECUTE ON FUNCTION bank.search_documents(vector, sparsevec, int, text[]) TO app_agent;
