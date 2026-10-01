-- 01_vault.sql - the vault server: the secret key and the token mapping
-- =============================================================================
-- This database runs on its own PostgreSQL server (container lab16_vault).
-- The bank database has no foreign server pointing here, so nothing in the
-- bank, its superuser included, can read the mapping.
--
-- Nobody reads the key, not even the tokenizer: tokens are computed INSIDE this
-- database by vault.tokenize(), a SECURITY DEFINER function. The tokenizer role
-- may only execute it. The reidentifier role may only read the mapping.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA vault;
REVOKE ALL ON SCHEMA public FROM PUBLIC;

-- One row per key version. Rotating the key = insert a new version, set it active.
CREATE TABLE vault.keys (
    key_version int PRIMARY KEY,
    secret      bytea NOT NULL,
    active      boolean NOT NULL DEFAULT false,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX keys_one_active ON vault.keys (active) WHERE active;

INSERT INTO vault.keys (key_version, secret, active)
VALUES (1, gen_random_bytes(32), true);

-- The mapping: token -> real value, per key version.
CREATE TABLE vault.mapping (
    token       text NOT NULL,
    key_version int  NOT NULL REFERENCES vault.keys,
    category    text NOT NULL,
    value       text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (token, key_version),
    UNIQUE (key_version, category, value)
);

-- token = CATEGORY_ + first 6 bytes of HMAC-SHA256(key, category:normalized value)
-- 12 hex characters: collisions are caught by the primary key, never silent.
CREATE FUNCTION vault.tokenize(p_category text, p_values text[])
RETURNS TABLE (value text, token text)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = vault, public, pg_temp
AS $$
#variable_conflict use_column
DECLARE
    k  bytea;
    kv int;
BEGIN
    SELECT secret, key_version INTO k, kv FROM vault.keys WHERE active;
    IF k IS NULL THEN
        RAISE EXCEPTION 'no active key';
    END IF;
    RETURN QUERY
    WITH src AS (
        SELECT DISTINCT v AS value FROM unnest(p_values) AS v WHERE v IS NOT NULL
    ), tok AS (
        SELECT s.value,
               upper(p_category) || '_' ||
               substr(encode(hmac(convert_to(upper(p_category) || ':' || lower(btrim(s.value)), 'UTF8'),
                                  k, 'sha256'), 'hex'), 1, 12) AS token
        FROM src s
    ), ins AS (
        INSERT INTO vault.mapping (token, key_version, category, value)
        -- two spellings of one value normalize to the same token: keep the first
        SELECT DISTINCT ON (t.token) t.token, kv, upper(p_category), t.value
        FROM tok t ORDER BY t.token, t.value COLLATE "C"
        ON CONFLICT DO NOTHING
    )
    SELECT t.value, t.token FROM tok t;
END;
$$;

CREATE FUNCTION vault.active_key_version()
RETURNS int LANGUAGE sql SECURITY DEFINER SET search_path = vault, pg_temp
AS $$ SELECT key_version FROM vault.keys WHERE active $$;

-- Key rotation: new random key becomes active. Old mappings stay for rollback.
CREATE FUNCTION vault.rotate_key()
RETURNS int LANGUAGE plpgsql SECURITY DEFINER SET search_path = vault, public, pg_temp
AS $$
DECLARE nv int;
BEGIN
    SELECT coalesce(max(key_version), 0) + 1 INTO nv FROM vault.keys;
    UPDATE vault.keys SET active = false WHERE active;
    INSERT INTO vault.keys (key_version, secret, active) VALUES (nv, gen_random_bytes(32), true);
    RETURN nv;
END;
$$;

-- Roles ------------------------------------------------------------------------
CREATE ROLE tokenizer    LOGIN PASSWORD 'vault2026!';
CREATE ROLE reidentifier LOGIN PASSWORD 'vault2026!';

REVOKE ALL ON ALL TABLES IN SCHEMA vault FROM PUBLIC;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA vault FROM PUBLIC;
GRANT USAGE ON SCHEMA vault TO tokenizer, reidentifier;

-- tokenizer: compute tokens, nothing else (no SELECT on keys or mapping)
GRANT EXECUTE ON FUNCTION vault.tokenize(text, text[]), vault.active_key_version() TO tokenizer;

-- reidentifier: read the mapping, never the key
GRANT SELECT ON vault.mapping TO reidentifier;
GRANT EXECUTE ON FUNCTION vault.active_key_version() TO reidentifier;

-- rotate_key stays with vault_admin only.

-- Every statement the reidentifier runs goes to the vault's server log: who reversed
-- which tokens, and when. (This image has no pgAudit; log_statement is the plain-core audit.)
ALTER ROLE reidentifier SET log_statement = 'all';
