#!/usr/bin/env python3
"""Shared configuration for Lab 04."""
import os

# ---- PostgreSQL (metadata catalog + data marts) ----
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5433")
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

# ---- Embeddings: Voyage AI (Anthropic recommended) ----
# voyage-finance-2: finance-optimized, 1024 dims, 32K context
# +7% NDCG@10 vs OpenAI on financial benchmarks
# Supports input_type="query" vs "document" for asymmetric search
VOYAGE_API_KEY  = os.getenv("VOYAGE_API_KEY", "")
EMBEDDING_MODEL = "voyage-finance-2"
EMBEDDING_DIM   = 1024

# ---- LLM: OpenAI (Agent reasoning + DDL generation) ----
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL      = os.getenv("CHAT_MODEL", "gpt-5.2")
CHAT_MODEL_FAST = os.getenv("CHAT_MODEL_FAST", "gpt-5-mini")

# ---- RAG parameters ----
SIMILARITY_THRESHOLD = 0.40
TOP_K = 15
