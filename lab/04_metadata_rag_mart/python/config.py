#!/usr/bin/env python3
"""Shared configuration for Lab 04."""
import os

# ---- PostgreSQL (metadata catalog + data marts) ----
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5434")
DB_NAME     = os.getenv("DB_NAME", "metadata_catalog")
DB_USER     = os.getenv("DB_USER", "dba_admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "dbi2026!")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ---- MinIO (S3-compatible data lake) ----
S3_ENDPOINT   = os.getenv("S3_ENDPOINT", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minio_admin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minio_2026!")
S3_BUCKET     = os.getenv("S3_BUCKET", "trading-lake")

# ---- API Keys ----
VOYAGE_API_KEY    = os.getenv("VOYAGE_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ---- Ollama (self-hosted models) ----
OLLAMA_ENDPOINT    = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

# ---- Embedding Models Registry ----
# All models output 1024 dimensions (OpenAI via `dimensions` param, Ollama via model default)
EMBEDDING_DIM = 1024

EMBEDDING_MODELS = {
    "voyage-finance-2": {
        "provider": "voyage",
        "dimension": 1024,
        "description": "Finance-optimized, +7% nDCG@10 vs OpenAI on financial benchmarks",
    },
    "text-embedding-3-small": {
        "provider": "openai",
        "dimension": 1024,
        "description": "OpenAI small embedding model, cost-effective, supports dimension reduction",
    },
    "text-embedding-3-large": {
        "provider": "openai",
        "dimension": 1024,
        "description": "OpenAI large embedding model, highest quality, supports dimension reduction",
    },
    "mxbai-embed-large": {
        "provider": "ollama",
        "dimension": 1024,
        "description": "Open-source mxbai-embed-large via Ollama, no API key needed",
    },
    "fake": {
        "provider": "fake",
        "dimension": 1024,
        "description": "Deterministic random vectors for API-free testing",
    },
}

# ---- LLM Models Registry ----
LLM_MODELS = {
    "gpt-5.2": {
        "provider": "openai",
        "tier": "flagship",
        "description": "OpenAI flagship model, best for complex DDL generation",
    },
    "gpt-5-mini": {
        "provider": "openai",
        "tier": "fast",
        "description": "OpenAI fast model, good for reasoning summaries",
    },
    "claude-opus-4-6": {
        "provider": "anthropic",
        "tier": "flagship",
        "description": "Anthropic flagship model, strong reasoning and code generation",
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "tier": "balanced",
        "description": "Anthropic balanced model, good quality/speed trade-off",
    },
    "claude-haiku-4-5": {
        "provider": "anthropic",
        "tier": "fast",
        "description": "Anthropic fast model, lowest latency",
    },
    "fake": {
        "provider": "fake",
        "tier": "test",
        "description": "Deterministic string responses for API-free testing",
    },
}

# ---- Default Providers (backward compatible) ----
EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "voyage-finance-2")
CHAT_MODEL      = os.getenv("CHAT_MODEL", "gpt-5.2")
CHAT_MODEL_FAST = os.getenv("CHAT_MODEL_FAST", "gpt-5-mini")

# ---- RAG parameters ----
SIMILARITY_THRESHOLD = 0.40
TOP_K = 15
