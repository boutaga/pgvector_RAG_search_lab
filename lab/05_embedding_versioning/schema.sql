-- =============================================================================
-- Lab 5: Embedding Versioning & Event-Driven Refresh
-- Schema for Wikipedia database (25K articles)
-- =============================================================================
-- Target: PostgreSQL with pgvector + pgvectorscale (DiskANN)
-- Table: articles (id INTEGER PK, url, title, content, ...)
-- =============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Extend the articles table with versioning support columns
-- ---------------------------------------------------------------------------

ALTER TABLE articles
    ADD COLUMN IF NOT EXISTS content_hash TEXT,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();

-- Trigger: auto-compute content_hash and updated_at on content changes
CREATE OR REPLACE FUNCTION fn_articles_content_hash()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' OR NEW.content IS DISTINCT FROM OLD.content THEN
        NEW.content_hash := md5(NEW.content);
        NEW.updated_at   := now();
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_content_hash ON articles;
CREATE TRIGGER trg_content_hash
    BEFORE INSERT OR UPDATE OF content ON articles
    FOR EACH ROW
    EXECUTE FUNCTION fn_articles_content_hash();

-- Backfill existing rows
UPDATE articles
SET content_hash = md5(content),
    updated_at   = now()
WHERE content_hash IS NULL;

-- ---------------------------------------------------------------------------
-- 2. Versioned embeddings table
-- ---------------------------------------------------------------------------

-- Note: vector(1536) matches text-embedding-3-small (default model).
-- If upgrading to text-embedding-3-large (3072 dims), you must ALTER
-- the column dimension first:
--   ALTER TABLE article_embeddings_versioned
--       ALTER COLUMN embedding TYPE vector(3072);
CREATE TABLE IF NOT EXISTS article_embeddings_versioned (
    id              BIGSERIAL PRIMARY KEY,
    article_id      INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL DEFAULT 0,
    chunk_text      TEXT,
    embedding       vector(1536),
    model_version   TEXT NOT NULL DEFAULT 'text-embedding-3-small',
    source_hash     TEXT,                          -- content_hash at embed time
    is_current      BOOLEAN NOT NULL DEFAULT true,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    replaced_at     TIMESTAMPTZ
);

-- Partial unique: only one current embedding per (article, chunk, model)
CREATE UNIQUE INDEX IF NOT EXISTS uix_current_embedding
    ON article_embeddings_versioned (article_id, chunk_index, model_version)
    WHERE is_current = true;

-- Fast lookup by article
CREATE INDEX IF NOT EXISTS ix_embed_article
    ON article_embeddings_versioned (article_id)
    WHERE is_current = true;

-- DiskANN vector index (pgvectorscale) for similarity search
CREATE INDEX IF NOT EXISTS ix_embed_diskann
    ON article_embeddings_versioned
    USING diskann (embedding)
    WHERE is_current = true;

-- ---------------------------------------------------------------------------
-- 3. Embedding queue (SKIP LOCKED pattern)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS embedding_queue (
    id              BIGSERIAL PRIMARY KEY,
    article_id      INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    status          TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'skipped')),
    priority        INTEGER NOT NULL DEFAULT 5,     -- 1 = highest
    content_hash    TEXT,                            -- hash when queued
    change_type     TEXT DEFAULT 'content_update',   -- content_update | new | manual | model_upgrade
    queued_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error_message   TEXT,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    worker_id       TEXT
);

CREATE INDEX IF NOT EXISTS ix_queue_pending
    ON embedding_queue (priority, queued_at)
    WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS ix_queue_article
    ON embedding_queue (article_id, queued_at DESC);

-- ---------------------------------------------------------------------------
-- 4. Change log (tracks embed/skip decisions)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS embedding_change_log (
    id              BIGSERIAL PRIMARY KEY,
    article_id      INTEGER NOT NULL,
    old_hash        TEXT,
    new_hash        TEXT,
    change_type     TEXT,          -- content_update | new | manual
    decision        TEXT,          -- EMBED | SKIP
    similarity      NUMERIC(5,4), -- text similarity ratio
    details         JSONB,        -- structural analysis breakdown
    decided_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- 5. Retrieval quality log (for feedback loop)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS retrieval_quality_log (
    id              BIGSERIAL PRIMARY KEY,
    query_text      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    result_ids      INTEGER[],
    relevance_scores NUMERIC[],
    ndcg_score      NUMERIC(5,4),
    user_feedback   TEXT,          -- positive | negative | null
    logged_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- 6. Trigger: auto-queue embedding on article content change
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION fn_queue_embedding_update()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    -- On INSERT: always queue
    IF TG_OP = 'INSERT' THEN
        INSERT INTO embedding_queue (article_id, content_hash, change_type, priority)
        VALUES (NEW.id, NEW.content_hash, 'new', 3);

        PERFORM pg_notify('embedding_updates', json_build_object(
            'article_id', NEW.id,
            'change_type', 'new'
        )::text);

        RETURN NEW;
    END IF;

    -- On UPDATE: only queue if content actually changed
    IF TG_OP = 'UPDATE' AND NEW.content_hash IS DISTINCT FROM OLD.content_hash THEN
        INSERT INTO embedding_queue (article_id, content_hash, change_type, priority)
        VALUES (NEW.id, NEW.content_hash, 'content_update', 5);

        PERFORM pg_notify('embedding_updates', json_build_object(
            'article_id', NEW.id,
            'change_type', 'content_update'
        )::text);
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_queue_embedding ON articles;
CREATE TRIGGER trg_queue_embedding
    AFTER INSERT OR UPDATE OF content ON articles
    FOR EACH ROW
    EXECUTE FUNCTION fn_queue_embedding_update();

-- ---------------------------------------------------------------------------
-- 7. Reaper: clean up old completed/failed queue entries (run periodically)
-- ---------------------------------------------------------------------------
-- If pg_cron is available, uncomment:
-- SELECT cron.schedule('embedding-queue-reaper', '0 3 * * *',
--     $$DELETE FROM embedding_queue
--       WHERE status IN ('completed', 'skipped')
--         AND completed_at < now() - interval '7 days'$$
-- );

-- Manual reaper query:
-- DELETE FROM embedding_queue
-- WHERE status IN ('completed', 'skipped')
--   AND completed_at < now() - interval '7 days';

COMMIT;

-- ---------------------------------------------------------------------------
-- Verification queries (run after applying schema)
-- ---------------------------------------------------------------------------
-- \d articles                           -- should show content_hash, updated_at
-- \d article_embeddings_versioned       -- check FK, partial unique index
-- \d embedding_queue                    -- check FK, status constraint
-- SELECT count(*) FROM articles WHERE content_hash IS NOT NULL;  -- should = total rows
