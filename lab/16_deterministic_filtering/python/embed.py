"""Embed the documents locally, one versioned embedding set per text state.

Dense: voyage-4-nano (1024 dimensions), sparse: SPLADE through fastembed. Both
run on this machine, so no text leaves the perimeter at the embedding step.

  --state raw|redacted|tokenized|all   which text(s) to embed, several allowed (default all)

Progress is printed as one line per 160 documents when the output is a file, so
`tail -f captures/07_embed.txt` or `python python/status.py` can follow a background run.

Each run creates a new row in bank.embedding_versions and makes it the active
version for its state; the previous version stays for rollback (see 10_versioning.sql).

    python python/embed.py --state all
"""
import argparse
import hashlib
import sys
import time

from _common import (DENSE_MODEL, DIMS, SPARSE_MODEL, embed_documents, gateway_conn,
                     sparse_embed, sparse_literal, vec_literal)

TEXT_COLUMNS = {
    "raw": ("title", "body"),
    "redacted": ("title_redacted", "body_redacted"),
    "tokenized": ("title_tokenized", "body_tokenized"),
}
BATCH = 16


def embed_state(cur, state):
    title_col, body_col = TEXT_COLUMNS[state]
    cur.execute(f"SELECT doc_id, bank_id, {title_col} || '. ' || {body_col}, key_version "
                f"FROM bank.documents ORDER BY doc_id")
    rows = cur.fetchall()
    if any(r[2] is None for r in rows):
        raise SystemExit(f"{state}: some documents have no {body_col}. Run tokenize_corpus.py first.")
    key_version = rows[0][3] if state == "tokenized" else None
    cur.execute("INSERT INTO bank.embedding_versions (state, dense_model, sparse_model, dims, key_version) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING version_id",
                (state, DENSE_MODEL, SPARSE_MODEL, DIMS, key_version))
    version_id = cur.fetchone()[0]
    t0 = time.time()
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        texts = [r[2] for r in chunk]
        dense = embed_documents(texts)
        sparse = sparse_embed(texts)
        cur.executemany("INSERT INTO bank.embeddings (version_id, doc_id, bank_id, dense, sparse, text_sha256) "
                        "VALUES (%s, %s, %s, %s::vector, %s::sparsevec, %s)",
                        [(version_id, r[0], r[1], vec_literal(d), sparse_literal(s),
                          hashlib.sha256(r[2].encode()).hexdigest())
                         for r, d, s in zip(chunk, dense, sparse)])
        done = min(i + BATCH, len(rows))
        if sys.stdout.isatty():
            print(f"\r  {state}: {done}/{len(rows)}", end="", flush=True)
        elif done % 160 == 0 or done == len(rows):
            print(f"  {state}: {done}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    cur.execute("UPDATE bank.embedding_versions SET is_active = false WHERE state = %s AND is_active", (state,))
    cur.execute("UPDATE bank.embedding_versions SET is_active = true WHERE version_id = %s", (version_id,))
    print(f"\r  {state}: {len(rows)} documents, version {version_id} active, {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", nargs="+", choices=["raw", "redacted", "tokenized", "all"], default=["all"])
    args = ap.parse_args()
    # tokenized first: it is the one the agent needs
    states = ["tokenized", "raw", "redacted"] if "all" in args.state else args.state
    conn = gateway_conn()
    with conn.cursor() as cur:
        for state in states:
            embed_state(cur, state)


if __name__ == "__main__":
    main()
