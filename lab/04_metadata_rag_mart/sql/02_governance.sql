-- 02_governance.sql — Governance framework
--
-- Key principle: Agent 2 (Pipeline) ENFORCES governance before creating any mart.
-- The audit trail captures every request end-to-end for regulatory compliance.

-- =========================================================================
-- PROVISIONING AUDIT (immutable log)
-- =========================================================================
CREATE TABLE governance.provisioning_audit (
    audit_id            SERIAL PRIMARY KEY,
    request_id          UUID NOT NULL DEFAULT uuid_generate_v4(),
    requested_at        TIMESTAMPTZ DEFAULT NOW(),
    requested_by        VARCHAR(128) NOT NULL,
    requester_role      VARCHAR(64) NOT NULL,
    -- Request
    request_text        TEXT NOT NULL,                -- original natural language
    -- Agent 1 output
    rag_results         JSONB,                       -- search results summary
    rag_search_time_ms  INTEGER,
    -- Agent 2 output
    sql_generated       TEXT,                         -- DDL/DML
    target_object       VARCHAR(256),                 -- data_mart.dm_xxx
    source_tables       TEXT[],                       -- Parquet files used
    -- Governance applied
    classification      VARCHAR(32),
    pii_columns_found   TEXT[],
    masking_applied     BOOLEAN DEFAULT FALSE,
    rls_applied         BOOLEAN DEFAULT FALSE,
    grants_applied      TEXT[],                       -- roles that got SELECT
    -- Execution
    status              VARCHAR(32) DEFAULT 'pending'
                        CHECK (status IN ('pending','validated','executed','rejected','error')),
    row_count           BIGINT,
    execution_time_ms   INTEGER,
    executed_at         TIMESTAMPTZ,
    error_message       TEXT
);

CREATE INDEX idx_audit_status ON governance.provisioning_audit (status);
CREATE INDEX idx_audit_time ON governance.provisioning_audit (requested_at DESC);
CREATE INDEX idx_audit_requester ON governance.provisioning_audit (requested_by);

-- =========================================================================
-- DATA MART REGISTRY
-- =========================================================================
CREATE TABLE governance.data_mart_registry (
    registry_id         SERIAL PRIMARY KEY,
    mart_name           VARCHAR(256) NOT NULL UNIQUE,
    schema_name         VARCHAR(128) DEFAULT 'data_mart',
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    created_by          VARCHAR(128),
    request_id          UUID,
    -- Lineage
    source_tables       TEXT[],                       -- Parquet sources
    source_columns      TEXT[],
    join_conditions     TEXT[],
    -- Governance
    classification      VARCHAR(32),
    has_pii_masking     BOOLEAN DEFAULT FALSE,
    has_rls             BOOLEAN DEFAULT FALSE,
    allowed_roles       TEXT[],
    -- State
    refresh_policy      VARCHAR(32) DEFAULT 'snapshot'
                        CHECK (refresh_policy IN ('snapshot','manual','daily','cdc')),
    last_refreshed      TIMESTAMPTZ,
    row_count           BIGINT,
    description         TEXT
);

-- =========================================================================
-- ROLES (consumers of data marts)
-- =========================================================================
DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'bi_analyst') THEN
        CREATE ROLE bi_analyst LOGIN PASSWORD 'bi_2026!';
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'risk_manager') THEN
        CREATE ROLE risk_manager LOGIN PASSWORD 'risk_2026!';
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'compliance_officer') THEN
        CREATE ROLE compliance_officer LOGIN PASSWORD 'compl_2026!';
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'pipeline_agent') THEN
        CREATE ROLE pipeline_agent LOGIN PASSWORD 'pipe_2026!';
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'portfolio_manager') THEN
        CREATE ROLE portfolio_manager LOGIN PASSWORD 'pm_2026!';
    END IF;
END $$;

-- Permissions
GRANT USAGE ON SCHEMA data_mart TO bi_analyst, risk_manager, compliance_officer, portfolio_manager;
GRANT USAGE ON SCHEMA catalog TO pipeline_agent, portfolio_manager;
GRANT USAGE ON SCHEMA lake TO pipeline_agent;
GRANT CREATE ON SCHEMA data_mart TO pipeline_agent;
GRANT CREATE ON SCHEMA lake TO pipeline_agent;
GRANT SELECT ON ALL TABLES IN SCHEMA catalog TO pipeline_agent, portfolio_manager;
GRANT USAGE ON SCHEMA governance TO pipeline_agent, bi_analyst, risk_manager, compliance_officer, portfolio_manager;
GRANT INSERT ON governance.provisioning_audit TO pipeline_agent;
GRANT INSERT, UPDATE ON governance.data_mart_registry TO pipeline_agent;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA governance TO pipeline_agent;

-- Default privileges: new tables in data_mart get SELECT for pipeline_agent
ALTER DEFAULT PRIVILEGES IN SCHEMA data_mart GRANT SELECT ON TABLES TO pipeline_agent;
ALTER DEFAULT PRIVILEGES IN SCHEMA data_mart GRANT SELECT ON TABLES TO portfolio_manager;

-- =========================================================================
-- CLASSIFICATION → ROLE ACCESS MATRIX
-- =========================================================================
--   public       → bi_analyst, risk_manager, compliance_officer, portfolio_manager
--   internal     → bi_analyst, risk_manager, compliance_officer, portfolio_manager
--   confidential → risk_manager, compliance_officer, portfolio_manager
--   restricted   → compliance_officer only

CREATE OR REPLACE FUNCTION governance.get_allowed_roles(data_classification VARCHAR)
RETURNS TEXT[] LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN
    RETURN CASE data_classification
        WHEN 'public'       THEN ARRAY['bi_analyst','risk_manager','compliance_officer','portfolio_manager']
        WHEN 'internal'     THEN ARRAY['bi_analyst','risk_manager','compliance_officer','portfolio_manager']
        WHEN 'confidential' THEN ARRAY['risk_manager','compliance_officer','portfolio_manager']
        WHEN 'restricted'   THEN ARRAY['compliance_officer']
        ELSE                     ARRAY['compliance_officer']
    END;
END $$;

-- =========================================================================
-- PII MASKING FUNCTIONS
-- =========================================================================

-- Mask names: "Dr. Elena Brunner" → "D** E**** B******"
CREATE OR REPLACE FUNCTION governance.mask_name(val TEXT)
RETURNS TEXT LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    parts TEXT[];
    masked TEXT := '';
    p TEXT;
BEGIN
    IF val IS NULL THEN RETURN NULL; END IF;
    parts := string_to_array(val, ' ');
    FOREACH p IN ARRAY parts LOOP
        IF masked != '' THEN masked := masked || ' '; END IF;
        IF length(p) <= 1 THEN
            masked := masked || p;
        ELSE
            masked := masked || left(p, 1) || repeat('*', length(p) - 1);
        END IF;
    END LOOP;
    RETURN masked;
END $$;

-- Mask account numbers: "CH9300010001" → "CH93********"
CREATE OR REPLACE FUNCTION governance.mask_account(val TEXT)
RETURNS TEXT LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN
    IF val IS NULL THEN RETURN NULL; END IF;
    IF length(val) <= 4 THEN RETURN repeat('*', length(val)); END IF;
    RETURN left(val, 4) || repeat('*', length(val) - 4);
END $$;

-- Mask ISIN: "CH0012032048" → "CH00*****048"
CREATE OR REPLACE FUNCTION governance.mask_isin(val TEXT)
RETURNS TEXT LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN
    IF val IS NULL THEN RETURN NULL; END IF;
    IF length(val) <= 7 THEN RETURN repeat('*', length(val)); END IF;
    RETURN left(val, 4) || repeat('*', length(val) - 7) || right(val, 3);
END $$;

GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA governance TO bi_analyst, risk_manager, compliance_officer, pipeline_agent, portfolio_manager;
