-- ============================================================================
-- Evaluation Schema for nDCG Tracking
-- Based on: https://www.dbi-services.com/blog/rag-series-adaptive-rag-understanding-confidence-precision-ndcg/
-- ============================================================================
--
-- This schema supports:
-- - Test query management
-- - Multi-level relevance grading (0-2 scale)
-- - Retrieval logging for analysis
-- - Evaluation results tracking
-- - Temporal trend analysis
--
-- Usage:
--   psql -U postgres -d your_database -f evaluation_schema.sql
--
-- ============================================================================

-- Drop existing objects if they exist (for clean reinstall)
DROP VIEW IF EXISTS latest_evaluation_summary CASCADE;
DROP VIEW IF EXISTS relevance_grade_distribution CASCADE;
DROP FUNCTION IF EXISTS get_ndcg_trend(VARCHAR, INTEGER) CASCADE;
DROP TABLE IF EXISTS evaluation_results CASCADE;
DROP TABLE IF EXISTS retrieval_log CASCADE;
DROP TABLE IF EXISTS relevance_grades CASCADE;
DROP TABLE IF EXISTS test_queries CASCADE;

-- ============================================================================
-- Table: test_queries
-- Stores test queries for evaluation
-- ============================================================================

CREATE TABLE test_queries (
    query_id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_type VARCHAR(50),          -- factual, conceptual, exploratory, other
    category VARCHAR(100),            -- e.g., technical, general, specific domain
    created_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100),          -- human name or 'llm'
    notes TEXT,
    CONSTRAINT query_text_not_empty CHECK (length(trim(query_text)) > 0)
);

COMMENT ON TABLE test_queries IS 'Test queries for RAG evaluation';
COMMENT ON COLUMN test_queries.query_type IS 'Type of query: factual, conceptual, exploratory, other';
COMMENT ON COLUMN test_queries.category IS 'Optional category for organizing queries';
COMMENT ON COLUMN test_queries.created_by IS 'Creator identifier (human name or ''llm'')';

-- Indexes for test_queries
CREATE INDEX idx_test_queries_type ON test_queries(query_type);
CREATE INDEX idx_test_queries_category ON test_queries(category);
CREATE INDEX idx_test_queries_created_at ON test_queries(created_at);

-- ============================================================================
-- Table: relevance_grades
-- Stores human or LLM-assigned relevance grades
-- ============================================================================

CREATE TABLE relevance_grades (
    query_id INTEGER REFERENCES test_queries(query_id) ON DELETE CASCADE,
    doc_id INTEGER NOT NULL,
    rel_grade INTEGER NOT NULL CHECK (rel_grade IN (0, 1, 2)),
    labeler VARCHAR(100),             -- human name or 'llm'
    label_method VARCHAR(50),         -- 'human', 'llm', 'hybrid'
    label_date TIMESTAMP DEFAULT NOW(),
    notes TEXT,
    PRIMARY KEY (query_id, doc_id)
);

COMMENT ON TABLE relevance_grades IS 'Relevance grades for query-document pairs';
COMMENT ON COLUMN relevance_grades.rel_grade IS '0=irrelevant, 1=relevant, 2=highly relevant';
COMMENT ON COLUMN relevance_grades.labeler IS 'Who assigned the grade (human or LLM)';
COMMENT ON COLUMN relevance_grades.label_method IS 'Method used for labeling';

-- Indexes for relevance_grades
CREATE INDEX idx_relevance_query ON relevance_grades(query_id);
CREATE INDEX idx_relevance_doc ON relevance_grades(doc_id);
CREATE INDEX idx_relevance_grade ON relevance_grades(rel_grade);
CREATE INDEX idx_relevance_labeler ON relevance_grades(labeler);

-- ============================================================================
-- Table: retrieval_log
-- Logs retrieval attempts for analysis
-- ============================================================================

CREATE TABLE retrieval_log (
    log_id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES test_queries(query_id) ON DELETE CASCADE,
    doc_id INTEGER NOT NULL,
    rank INTEGER NOT NULL,            -- Position in results (1-based)
    score FLOAT,                      -- Similarity/relevance score
    retrieval_method VARCHAR(50),     -- 'vector', 'hybrid', 'adaptive', etc.
    timestamp TIMESTAMP DEFAULT NOW(),
    session_id VARCHAR(100),          -- Optional session grouping
    CONSTRAINT rank_positive CHECK (rank > 0)
);

COMMENT ON TABLE retrieval_log IS 'Log of all retrieval attempts for analysis';
COMMENT ON COLUMN retrieval_log.rank IS 'Position in search results (1-based)';
COMMENT ON COLUMN retrieval_log.retrieval_method IS 'Search method used (vector, hybrid, adaptive)';
COMMENT ON COLUMN retrieval_log.session_id IS 'Optional identifier for grouping related retrievals';

-- Indexes for retrieval_log
CREATE INDEX idx_retrieval_log_query ON retrieval_log(query_id);
CREATE INDEX idx_retrieval_log_doc ON retrieval_log(doc_id);
CREATE INDEX idx_retrieval_log_time ON retrieval_log(timestamp);
CREATE INDEX idx_retrieval_log_method ON retrieval_log(retrieval_method);
CREATE INDEX idx_retrieval_log_session ON retrieval_log(session_id);

-- ============================================================================
-- Table: evaluation_results
-- Stores computed metrics for each evaluation run
-- ============================================================================

CREATE TABLE evaluation_results (
    eval_id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES test_queries(query_id) ON DELETE CASCADE,
    retrieval_method VARCHAR(50) NOT NULL,
    k_value INTEGER NOT NULL,        -- Number of results evaluated (top-k)
    ndcg_score FLOAT,                 -- Normalized DCG score (0-1)
    dcg_score FLOAT,                  -- DCG score (unnormalized)
    idcg_score FLOAT,                 -- Ideal DCG score
    precision_at_k FLOAT,             -- Precision@k metric
    recall_at_k FLOAT,                -- Recall@k metric
    f1_at_k FLOAT,                    -- F1 score@k
    mrr FLOAT,                        -- Mean Reciprocal Rank
    timestamp TIMESTAMP DEFAULT NOW(),
    session_id VARCHAR(100),
    CONSTRAINT k_value_positive CHECK (k_value > 0),
    CONSTRAINT ndcg_range CHECK (ndcg_score IS NULL OR (ndcg_score >= 0 AND ndcg_score <= 1)),
    CONSTRAINT precision_range CHECK (precision_at_k IS NULL OR (precision_at_k >= 0 AND precision_at_k <= 1)),
    CONSTRAINT recall_range CHECK (recall_at_k IS NULL OR (recall_at_k >= 0 AND recall_at_k <= 1))
);

COMMENT ON TABLE evaluation_results IS 'Pre-computed evaluation metrics for each test';
COMMENT ON COLUMN evaluation_results.k_value IS 'Number of top results evaluated';
COMMENT ON COLUMN evaluation_results.ndcg_score IS 'nDCG score (0-1, higher is better)';
COMMENT ON COLUMN evaluation_results.session_id IS 'Groups related evaluations together';

-- Indexes for evaluation_results
CREATE INDEX idx_eval_results_query ON evaluation_results(query_id);
CREATE INDEX idx_eval_results_method ON evaluation_results(retrieval_method);
CREATE INDEX idx_eval_results_k ON evaluation_results(k_value);
CREATE INDEX idx_eval_results_time ON evaluation_results(timestamp);
CREATE INDEX idx_eval_results_session ON evaluation_results(session_id);

-- ============================================================================
-- View: latest_evaluation_summary
-- Quick summary of latest evaluation for each method
-- ============================================================================

CREATE VIEW latest_evaluation_summary AS
SELECT
    retrieval_method,
    k_value,
    COUNT(*) as num_queries,
    ROUND(AVG(ndcg_score)::numeric, 4) as avg_ndcg,
    ROUND(STDDEV(ndcg_score)::numeric, 4) as stddev_ndcg,
    ROUND(MIN(ndcg_score)::numeric, 4) as min_ndcg,
    ROUND(MAX(ndcg_score)::numeric, 4) as max_ndcg,
    ROUND(AVG(precision_at_k)::numeric, 4) as avg_precision,
    ROUND(AVG(recall_at_k)::numeric, 4) as avg_recall,
    ROUND(AVG(f1_at_k)::numeric, 4) as avg_f1,
    ROUND(AVG(mrr)::numeric, 4) as avg_mrr,
    MAX(timestamp) as last_eval_time
FROM evaluation_results
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY retrieval_method, k_value
ORDER BY retrieval_method, k_value;

COMMENT ON VIEW latest_evaluation_summary IS 'Summary of evaluations from the last 7 days';

-- ============================================================================
-- View: relevance_grade_distribution
-- Shows distribution of relevance grades per query
-- ============================================================================

CREATE VIEW relevance_grade_distribution AS
SELECT
    tq.query_id,
    tq.query_text,
    tq.query_type,
    COUNT(*) as total_labeled,
    SUM(CASE WHEN rg.rel_grade = 0 THEN 1 ELSE 0 END) as irrelevant,
    SUM(CASE WHEN rg.rel_grade = 1 THEN 1 ELSE 0 END) as relevant,
    SUM(CASE WHEN rg.rel_grade = 2 THEN 1 ELSE 0 END) as highly_relevant,
    ROUND(AVG(rg.rel_grade)::numeric, 2) as avg_grade,
    STRING_AGG(DISTINCT rg.labeler, ', ' ORDER BY rg.labeler) as labelers,
    MAX(rg.label_date) as last_labeled_date
FROM test_queries tq
JOIN relevance_grades rg ON tq.query_id = rg.query_id
GROUP BY tq.query_id, tq.query_text, tq.query_type
ORDER BY tq.query_id;

COMMENT ON VIEW relevance_grade_distribution IS 'Relevance grade statistics per query';

-- ============================================================================
-- Function: get_ndcg_trend
-- Get nDCG trend over time for a specific method
-- ============================================================================

CREATE OR REPLACE FUNCTION get_ndcg_trend(
    method_name VARCHAR,
    days_back INTEGER DEFAULT 30
)
RETURNS TABLE (
    date DATE,
    avg_ndcg FLOAT,
    num_queries INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        DATE(timestamp) as date,
        AVG(ndcg_score)::FLOAT as avg_ndcg,
        COUNT(DISTINCT query_id)::INTEGER as num_queries
    FROM evaluation_results
    WHERE retrieval_method = method_name
      AND timestamp > NOW() - (days_back || ' days')::INTERVAL
    GROUP BY DATE(timestamp)
    ORDER BY date;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_ndcg_trend IS 'Returns nDCG trend over time for a retrieval method';

-- ============================================================================
-- Function: compare_methods
-- Compare performance of different retrieval methods
-- ============================================================================

CREATE OR REPLACE FUNCTION compare_methods(
    k_val INTEGER DEFAULT 10,
    days_back INTEGER DEFAULT 7
)
RETURNS TABLE (
    method VARCHAR,
    num_queries INTEGER,
    avg_ndcg FLOAT,
    avg_precision FLOAT,
    avg_recall FLOAT,
    avg_f1 FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        retrieval_method as method,
        COUNT(DISTINCT query_id)::INTEGER as num_queries,
        AVG(ndcg_score)::FLOAT as avg_ndcg,
        AVG(precision_at_k)::FLOAT as avg_precision,
        AVG(recall_at_k)::FLOAT as avg_recall,
        AVG(f1_at_k)::FLOAT as avg_f1
    FROM evaluation_results
    WHERE k_value = k_val
      AND timestamp > NOW() - (days_back || ' days')::INTERVAL
    GROUP BY retrieval_method
    ORDER BY avg_ndcg DESC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION compare_methods IS 'Compare retrieval methods by average metrics';

-- ============================================================================
-- Sample Data (Optional - for testing)
-- ============================================================================

-- Uncomment to insert sample test queries
/*
INSERT INTO test_queries (query_text, query_type, category, created_by, notes)
VALUES
    ('What is machine learning?', 'conceptual', 'AI', 'system', 'Basic ML concept'),
    ('Who invented Python?', 'factual', 'programming', 'system', 'Creator of Python'),
    ('Explain quantum computing', 'exploratory', 'physics', 'system', 'Advanced topic');

-- Sample relevance grades (assuming doc IDs 1-10 exist)
INSERT INTO relevance_grades (query_id, doc_id, rel_grade, labeler, label_method)
VALUES
    (1, 1, 2, 'admin', 'human'),
    (1, 2, 1, 'admin', 'human'),
    (1, 3, 0, 'admin', 'human'),
    (2, 4, 2, 'admin', 'human'),
    (2, 5, 1, 'admin', 'human'),
    (3, 6, 2, 'admin', 'human'),
    (3, 7, 1, 'admin', 'human'),
    (3, 8, 1, 'admin', 'human');
*/

-- ============================================================================
-- Grant Permissions (Adjust as needed for your environment)
-- ============================================================================

-- Uncomment and modify as needed:
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rag_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rag_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO rag_user;

-- ============================================================================
-- Verification Queries
-- ============================================================================

-- Check schema creation
DO $$
BEGIN
    RAISE NOTICE 'Evaluation schema created successfully!';
    RAISE NOTICE 'Tables: test_queries, relevance_grades, retrieval_log, evaluation_results';
    RAISE NOTICE 'Views: latest_evaluation_summary, relevance_grade_distribution';
    RAISE NOTICE 'Functions: get_ndcg_trend, compare_methods';
END $$;

-- Display table counts
SELECT
    'test_queries' as table_name,
    COUNT(*) as row_count
FROM test_queries
UNION ALL
SELECT 'relevance_grades', COUNT(*) FROM relevance_grades
UNION ALL
SELECT 'retrieval_log', COUNT(*) FROM retrieval_log
UNION ALL
SELECT 'evaluation_results', COUNT(*) FROM evaluation_results;

-- ============================================================================
-- End of evaluation_schema.sql
-- ============================================================================
