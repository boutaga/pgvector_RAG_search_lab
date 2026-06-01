"""Shared helpers for Lab 07 (security observability demo)."""
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

LAB_DIR = Path(__file__).resolve().parents[1]          # lab/07_security_observability
REPO_ROOT = Path(__file__).resolve().parents[3]        # Movies_pgvector_lab
# override=True so the lab's .env wins over any stale OPENAI_API_KEY already in
# the ambient shell environment.
load_dotenv(LAB_DIR / ".env", override=True)


def _env(key, default=None):
    return os.environ.get(key, default)


EMBED_MODEL = _env("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = _env("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_DIM = 1536  # text-embedding-3-small


def admin_conn():
    """Superuser connection (seeding, DDL). Bypasses RLS."""
    c = psycopg2.connect(
        host=_env("PGHOST", "localhost"), port=_env("PGPORT", "5436"),
        dbname=_env("PGDATABASE", "secobs"),
        user=_env("PGUSER", "dba_admin"), password=_env("PGPASSWORD", "dbi2026!"),
    )
    c.autocommit = True
    return c


def app_conn(tenant_id):
    """Non-superuser app connection with app.tenant_id set (RLS applies)."""
    c = psycopg2.connect(
        host=_env("PGHOST", "localhost"), port=_env("PGPORT", "5436"),
        dbname=_env("PGDATABASE", "secobs"),
        user=_env("APP_USER", "app_user"), password=_env("APP_PASSWORD", "dbi2026!"),
    )
    c.autocommit = True
    with c.cursor() as cur:
        cur.execute("SET app.tenant_id = %s", (tenant_id,))
    return c


def openai_client():
    from openai import OpenAI
    key = (_env("OPENAI_API_KEY", "") or "").strip()
    if not key:
        sys.exit(
            "OPENAI_API_KEY is empty.\n"
            "Paste your OpenAI key into lab/07_security_observability/.env "
            "(OPENAI_API_KEY=...) and re-run."
        )
    return OpenAI(api_key=key)


def embed_texts(client, texts, batch=128):
    """Return a list of embedding vectors for the given texts."""
    out = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch])
        out.extend(d.embedding for d in resp.data)
    return out


def vec_literal(v):
    """Format a python list as a pgvector literal."""
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"
