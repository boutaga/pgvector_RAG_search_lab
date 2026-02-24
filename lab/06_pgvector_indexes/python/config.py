"""
Lab 06 — Shared configuration
Loads settings from DEV/.env via python-dotenv, with sensible defaults.
"""

import os
from pathlib import Path

# Load .env from DEV/ folder (two levels up from python/)
_env_path = Path(__file__).resolve().parent.parent / "DEV" / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(_env_path, override=True)
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables

# Database
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = os.getenv("PGPORT", "5435")
DB_NAME = os.getenv("PGDATABASE", "wikipedia")
DB_USER = os.getenv("PGUSER", "dba_admin")
DB_PASSWORD = os.getenv("PGPASSWORD", "dbi2026!")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "3072"))

# Batch sizes
BATCH_SIZE_DENSE = int(os.getenv("BATCH_SIZE_DENSE", "30"))
BATCH_SIZE_SPARSE = int(os.getenv("BATCH_SIZE_SPARSE", "5"))

# SPLADE
SPLADE_MODEL = os.getenv("SPLADE_MODEL", "naver/splade-cocondenser-ensembledistil")
SPLADE_VOCAB_SIZE = 30522

# Data source (repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WIKIPEDIA_ZIP = REPO_ROOT / "vector_database_wikipedia_articles_embedded.zip"
WIKIPEDIA_CSV = "vector_database_wikipedia_articles_embedded.csv"
