-- 01_schema.sql - the synthetic bank (three tenants, same as the Swiss PGDay lab)
-- =============================================================================
-- Raw business data lives in schema bank. The *_token columns are filled by the
-- tokenization pipeline from the vault; they are not secret on their own (a
-- token without the vault means nothing) but the agent role cannot read them
-- either: it only reaches data through the tool functions in 04_tools.sql.
-- =============================================================================

CREATE SCHEMA bank;
CREATE SCHEMA gov;

CREATE TABLE bank.banks (
    bank_id text PRIMARY KEY,
    name    text NOT NULL
);

CREATE TABLE bank.relationship_managers (
    rm_id     int PRIMARY KEY,
    bank_id   text NOT NULL REFERENCES bank.banks,
    full_name text NOT NULL,
    email     text NOT NULL,
    rm_token    text,
    email_token text
);

CREATE TABLE bank.clients (
    client_id    int PRIMARY KEY,
    bank_id      text NOT NULL REFERENCES bank.banks,
    client_name  text NOT NULL,
    client_type  text NOT NULL CHECK (client_type IN ('company', 'private')),
    domicile     text NOT NULL,
    rm_id        int NOT NULL REFERENCES bank.relationship_managers,
    contact_phone text NOT NULL,   -- sensitive, deliberately left out of the labelling registry
    client_token text,
    phone_token  text
);

CREATE TABLE bank.accounts (
    account_id int PRIMARY KEY,
    client_id  int NOT NULL REFERENCES bank.clients,
    bank_id    text NOT NULL REFERENCES bank.banks,
    iban       text NOT NULL,
    currency   text NOT NULL,
    balance    numeric(14,2) NOT NULL,
    iban_token text
);

CREATE TABLE bank.servers (
    server_id   int PRIMARY KEY,
    bank_id     text NOT NULL REFERENCES bank.banks,
    hostname    text NOT NULL,
    fqdn        text NOT NULL,
    ip_address  text NOT NULL,
    role        text NOT NULL,
    environment text NOT NULL,
    host_token  text,
    fqdn_token  text,
    ip_token    text
);

-- Free text: advisor notes, incident tickets, emails.
-- body = raw, body_redacted = every sensitive value replaced by [REDACTED],
-- body_tokenized = every sensitive value replaced by its deterministic token.
CREATE TABLE bank.documents (
    doc_id          int PRIMARY KEY,
    bank_id         text NOT NULL REFERENCES bank.banks,
    doc_type        text NOT NULL CHECK (doc_type IN ('advisor_note', 'incident', 'email')),
    title           text NOT NULL,
    body            text NOT NULL,
    created_at      timestamptz NOT NULL,
    title_redacted  text,
    body_redacted   text,
    title_tokenized text,
    body_tokenized  text,
    key_version     int
);

-- Which tokens a document mentions, filled by the tokenizer from dictionary hits.
CREATE TABLE bank.document_mentions (
    doc_id   int  NOT NULL REFERENCES bank.documents,
    bank_id  text NOT NULL,
    token    text NOT NULL,
    category text NOT NULL,
    PRIMARY KEY (doc_id, token)
);
CREATE INDEX ON bank.document_mentions (token);

-- Embedding versions: one row per (model, text state, key version).
-- Exactly one active version per state; queries read the active one.
CREATE TABLE bank.embedding_versions (
    version_id  int GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    state       text NOT NULL CHECK (state IN ('raw', 'redacted', 'tokenized')),
    dense_model text NOT NULL,
    sparse_model text NOT NULL,
    dims        int  NOT NULL,
    key_version int,
    is_active   boolean NOT NULL DEFAULT false,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX embedding_versions_one_active ON bank.embedding_versions (state) WHERE is_active;

CREATE TABLE bank.embeddings (
    version_id int  NOT NULL REFERENCES bank.embedding_versions,
    doc_id     int  NOT NULL REFERENCES bank.documents,
    bank_id    text NOT NULL,
    dense      vector(1024) NOT NULL,
    sparse     sparsevec(30522) NOT NULL,
    text_sha256 text,   -- hash of the exact text embedded: detects vectors left stale by a re-tokenization
    PRIMARY KEY (version_id, doc_id)
);
CREATE INDEX embeddings_dense_hnsw ON bank.embeddings USING hnsw (dense vector_cosine_ops);
CREATE INDEX ON bank.embeddings (version_id, bank_id);
