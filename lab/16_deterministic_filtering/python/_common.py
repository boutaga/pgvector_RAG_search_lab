"""Shared helpers for lab 16: settings, connections, tenant scoping, embeddings."""
import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
from dotenv import load_dotenv

LAB_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = LAB_DIR.parent.parent
DATA_DIR = LAB_DIR / "data"
RESULTS_DIR = LAB_DIR / "results"
load_dotenv(LAB_DIR / ".env", override=True)  # the lab .env wins over a stale exported key

DENSE_MODEL = os.getenv("EMBED_MODEL", "voyageai/voyage-4-nano")
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
DIMS = 1024
SPARSE_DIMS = 30522
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-6-luna")


def _conn(host, port, db, user, password, autocommit=True):
    return psycopg.connect(host=os.getenv(host, "localhost"), port=os.getenv(port),
                           dbname=os.getenv(db), user=os.getenv(user),
                           password=os.getenv(password), autocommit=autocommit)


def admin_conn():
    return _conn("BANK_HOST", "BANK_PORT", "BANK_DB", "BANK_ADMIN_USER", "BANK_ADMIN_PASSWORD")


def gateway_conn():
    return _conn("BANK_HOST", "BANK_PORT", "BANK_DB", "GATEWAY_USER", "GATEWAY_PASSWORD")


def agent_conn():
    # autocommit off: the tenant is set per transaction (pooler-safe)
    return _conn("BANK_HOST", "BANK_PORT", "BANK_DB", "AGENT_USER", "AGENT_PASSWORD", autocommit=False)


def tokenizer_conn():
    return _conn("VAULT_HOST", "VAULT_PORT", "VAULT_DB", "TOKENIZER_USER", "TOKENIZER_PASSWORD")


def reidentifier_conn():
    return _conn("VAULT_HOST", "VAULT_PORT", "VAULT_DB", "REIDENTIFIER_USER", "REIDENTIFIER_PASSWORD")


@contextmanager
def tenant_cursor(conn, bank_id):
    """Cursor scoped to one bank for one transaction.

    set_config(..., true) is transaction-local, so a connection pooler cannot
    hand this bank's setting to the next request (the Swiss PGDay lab pattern).
    """
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("SELECT set_config('app.bank_id', %s, true)", (bank_id,))
            yield cur


_dense = None
_sparse = None
_lock = None
LOCK_FILE = "/tmp/lab16-models.lock"


def _model_lock():
    """One model job at a time. Each job loads about 3 GB of models; two of them do not
    fit in a 6 GB WSL VM, and the kernel kills one (seen in the 2026-10-01 hand run).
    The lock is held until the process exits."""
    global _lock
    if _lock is not None:
        return
    f = open(LOCK_FILE, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        holder = f.read().strip() or "unknown"
        raise SystemExit(f"Another lab job is using the embedding models ({holder}). "
                         f"Wait for it to finish: python python/status.py")
    f.seek(0)
    f.truncate()
    f.write(f"pid {os.getpid()}: {' '.join(os.sys.argv)}")
    f.flush()
    _lock = f


def dense_model():
    global _dense
    if _dense is None:
        _model_lock()
        from sentence_transformers import SentenceTransformer
        _dense = SentenceTransformer(DENSE_MODEL, trust_remote_code=True, truncate_dim=DIMS,
                                     device="cpu")
    return _dense


def sparse_model():
    global _sparse
    if _sparse is None:
        _model_lock()
        from fastembed import SparseTextEmbedding
        _sparse = SparseTextEmbedding(SPARSE_MODEL)
    return _sparse


def embed_documents(texts, batch_size=8):
    return dense_model().encode_document(texts, batch_size=batch_size, show_progress_bar=False)


def embed_queries(texts):
    return dense_model().encode_query(texts, show_progress_bar=False)


def sparse_embed(texts):
    return list(sparse_model().embed(texts, batch_size=16))


def vec_literal(v):
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


def sparse_literal(se):
    # pgvector sparsevec indices are 1-based
    pairs = sorted(zip(se.indices.tolist(), se.values.tolist()))
    return "{" + ",".join(f"{i + 1}:{w:.5f}" for i, w in pairs) + "}/" + str(SPARSE_DIMS)
