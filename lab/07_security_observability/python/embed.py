"""Generate embeddings for the corpus.

  --mode baseline : embed each document's real text (creates embeddings rows)
  --mode masked   : re-embed with client names replaced by anon.pseudo_company()
                    (deterministic), populating embedding_masked

Needs OPENAI_API_KEY in lab/07_security_observability/.env.
"""
import argparse

from _common import admin_conn, openai_client, embed_texts, vec_literal, EMBED_DIM


def run_baseline():
    conn = admin_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, tenant_id, title, body FROM app.documents ORDER BY id")
    docs = cur.fetchall()
    if not docs:
        raise SystemExit("No documents found. Run data/seed_documents.py first.")
    texts = [f"{title}. {body}" for (_id, _t, title, body) in docs]

    vecs = embed_texts(openai_client(), texts)
    if vecs and len(vecs[0]) != EMBED_DIM:
        raise SystemExit(f"Embedding dim {len(vecs[0])} != schema {EMBED_DIM}. "
                         f"Check OPENAI_EMBED_MODEL.")

    cur.execute("DELETE FROM app.embeddings")
    for (_id, tenant, _title, _body), vec, txt in zip(docs, vecs, texts):
        cur.execute(
            "INSERT INTO app.embeddings(tenant_id, document_id, chunk_text, embedding) "
            "VALUES (%s, %s, %s, %s::vector)",
            (tenant, _id, txt, vec_literal(vec)),
        )
    print(f"baseline: embedded {len(docs)} documents")


def run_masked():
    conn = admin_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT d.id, d.client_name, e.chunk_text "
        "FROM app.documents d JOIN app.embeddings e ON e.document_id = d.id "
        "ORDER BY d.id"
    )
    rows = cur.fetchall()
    if not rows:
        raise SystemExit("No embeddings found. Run embed.py --mode baseline first.")

    masked = []
    for _id, client, chunk in rows:
        cur.execute("SELECT anon.pseudo_company(%s::text)", (client,))
        pseudo = cur.fetchone()[0]
        masked.append((_id, chunk.replace(client, pseudo)))

    vecs = embed_texts(openai_client(), [m for (_id, m) in masked])
    for (_id, mtxt), vec in zip(masked, vecs):
        cur.execute(
            "UPDATE app.embeddings SET chunk_text_masked = %s, embedding_masked = %s::vector "
            "WHERE document_id = %s",
            (mtxt, vec_literal(vec), _id),
        )
    print(f"masked: re-embedded {len(rows)} documents (client names pseudonymized)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "masked"], required=True)
    args = ap.parse_args()
    run_baseline() if args.mode == "baseline" else run_masked()


if __name__ == "__main__":
    main()
