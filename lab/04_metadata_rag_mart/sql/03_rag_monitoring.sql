-- 03_rag_monitoring.sql — RAG Quality Monitoring
--
-- Measures and tracks the quality of the metadata RAG search over time.
-- Enables continuous improvement of embeddings, similarity thresholds,
-- and metadata enrichment (detail_bi / detail_agent).
--
-- Metrics: Precision@K, Recall@K, nDCG@K, MRR, MAP, latency

-- =========================================================================
-- SEARCH LOG — every search query + results
-- =========================================================================
CREATE TABLE rag_monitor.search_log (
    search_id           SERIAL PRIMARY KEY,
    query_text          TEXT NOT NULL,
    query_embedding     vector(1024),        -- stored for offline analysis
    -- Results returned
    result_tables       JSONB,               -- [{table_name, similarity, rank}, ...]
    result_columns      JSONB,               -- [{table.column, similarity, rank}, ...]
    result_kpis         JSONB,               -- [{kpi_name, similarity, rank}, ...]
    total_results       INTEGER,
    -- Classification of results
    max_classification  VARCHAR(32),
    pii_fields_found    TEXT[],
    -- Performance
    embedding_time_ms   INTEGER,             -- time to embed the query
    search_time_ms      INTEGER,             -- time for vector search
    reasoning_time_ms   INTEGER,             -- time for LLM reasoning
    total_time_ms       INTEGER,
    -- Context
    requester           VARCHAR(128),
    requester_role      VARCHAR(64),
    embedding_model     VARCHAR(128) DEFAULT 'voyage-finance-2',
    similarity_threshold REAL,
    top_k               INTEGER,
    --
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_search_log_time ON rag_monitor.search_log (created_at DESC);
CREATE INDEX idx_search_log_model ON rag_monitor.search_log (embedding_model);

-- =========================================================================
-- RELEVANCE JUDGMENTS — ground truth for evaluation
-- =========================================================================
-- Each judgment says: "for query X, catalog item Y has relevance grade Z"
-- Grades follow TREC conventions:
--   0 = irrelevant
--   1 = marginally relevant
--   2 = relevant
--   3 = highly relevant (perfect match)

CREATE TABLE rag_monitor.relevance_judgments (
    judgment_id         SERIAL PRIMARY KEY,
    query_text          TEXT NOT NULL,
    -- The catalog item being judged
    catalog_type        VARCHAR(32) NOT NULL
                        CHECK (catalog_type IN ('table','column','relationship','kpi')),
    catalog_item        TEXT NOT NULL,        -- e.g. "positions" or "positions.market_value"
    -- Relevance grade
    relevance_grade     INTEGER NOT NULL CHECK (relevance_grade BETWEEN 0 AND 3),
    -- Provenance
    judged_by           VARCHAR(128) NOT NULL DEFAULT 'golden_set',
    judgment_method     VARCHAR(32) DEFAULT 'manual'
                        CHECK (judgment_method IN ('manual','llm_auto','user_feedback')),
    notes               TEXT,
    --
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (query_text, catalog_type, catalog_item, judged_by)
);

CREATE INDEX idx_judgments_query ON rag_monitor.relevance_judgments (query_text);

-- =========================================================================
-- EVALUATION RUNS — batch metric computation results
-- =========================================================================
CREATE TABLE rag_monitor.evaluation_runs (
    run_id              SERIAL PRIMARY KEY,
    run_name            VARCHAR(256),
    run_date            TIMESTAMPTZ DEFAULT NOW(),
    -- Configuration
    embedding_model     VARCHAR(128),
    similarity_threshold REAL,
    top_k               INTEGER,
    num_queries         INTEGER,
    num_judgments        INTEGER,
    -- Retrieval quality metrics
    precision_at_5      REAL,
    precision_at_10     REAL,
    recall_at_5         REAL,
    recall_at_10        REAL,
    ndcg_at_5           REAL,
    ndcg_at_10          REAL,
    mrr                 REAL,                -- Mean Reciprocal Rank
    map                 REAL,                -- Mean Average Precision
    -- Latency metrics
    avg_latency_ms      REAL,
    p50_latency_ms      REAL,
    p95_latency_ms      REAL,
    p99_latency_ms      REAL,
    -- Per-query detail (for drill-down)
    per_query_metrics   JSONB,               -- [{query, precision, recall, ndcg, latency}, ...]
    --
    notes               TEXT
);

-- =========================================================================
-- USER FEEDBACK LOG — thumbs up/down on search results
-- =========================================================================
CREATE TABLE rag_monitor.feedback_log (
    feedback_id         SERIAL PRIMARY KEY,
    search_id           INTEGER REFERENCES rag_monitor.search_log(search_id),
    result_rank         INTEGER,             -- which result was rated
    catalog_type        VARCHAR(32),
    catalog_item        TEXT,
    is_relevant         BOOLEAN NOT NULL,    -- thumbs up (true) / down (false)
    feedback_text       TEXT,
    created_by          VARCHAR(128),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_feedback_search ON rag_monitor.feedback_log (search_id);

-- =========================================================================
-- HELPER VIEWS
-- =========================================================================

-- Search quality over time
CREATE OR REPLACE VIEW rag_monitor.v_search_quality_trend AS
SELECT
    date_trunc('day', run_date) AS eval_date,
    embedding_model,
    AVG(precision_at_5) AS avg_p5,
    AVG(recall_at_5) AS avg_r5,
    AVG(ndcg_at_5) AS avg_ndcg5,
    AVG(mrr) AS avg_mrr,
    AVG(avg_latency_ms) AS avg_latency
FROM rag_monitor.evaluation_runs
GROUP BY 1, 2
ORDER BY 1 DESC;

-- Feedback summary per query pattern
CREATE OR REPLACE VIEW rag_monitor.v_feedback_summary AS
SELECT
    sl.query_text,
    COUNT(*) AS total_feedback,
    SUM(CASE WHEN fl.is_relevant THEN 1 ELSE 0 END) AS positive,
    SUM(CASE WHEN NOT fl.is_relevant THEN 1 ELSE 0 END) AS negative,
    ROUND(AVG(CASE WHEN fl.is_relevant THEN 1.0 ELSE 0.0 END)::numeric, 3) AS relevance_rate
FROM rag_monitor.feedback_log fl
JOIN rag_monitor.search_log sl ON fl.search_id = sl.search_id
GROUP BY sl.query_text
ORDER BY total_feedback DESC;

-- Grant monitoring access
GRANT USAGE ON SCHEMA rag_monitor TO pipeline_agent, bi_analyst, risk_manager, compliance_officer;
GRANT SELECT ON ALL TABLES IN SCHEMA rag_monitor TO bi_analyst, risk_manager, compliance_officer;
GRANT INSERT ON rag_monitor.search_log TO pipeline_agent;
GRANT INSERT ON rag_monitor.feedback_log TO pipeline_agent;
GRANT ALL ON ALL SEQUENCES IN SCHEMA rag_monitor TO pipeline_agent;
