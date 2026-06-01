"""Tenant-scoped RAG query — the Act 1 leak demo.

Connects as the non-superuser app role, sets app.tenant_id, runs vector search
against the embeddings table and answers with the retrieved context. Retrieved
rows from another tenant are flagged as LEAK.

  python ask.py --tenant bank_a --query "Summarize the Q3 trading positions"
  python ask.py --tenant bank_a --query "..." --masked   # search masked embeddings
"""
import argparse

from _common import app_conn, openai_client, embed_texts, vec_literal, CHAT_MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--masked", action="store_true", help="search embedding_masked")
    args = ap.parse_args()

    oc = openai_client()
    qvec = vec_literal(embed_texts(oc, [args.query])[0])
    col = "embedding_masked" if args.masked else "embedding"

    conn = app_conn(args.tenant)
    cur = conn.cursor()
    cur.execute(
        f"SELECT tenant_id, document_id, chunk_text, {col} <=> %s::vector AS dist "
        f"FROM app.embeddings ORDER BY {col} <=> %s::vector LIMIT %s",
        (qvec, qvec, args.k),
    )
    rows = cur.fetchall()

    context = "\n".join(f"- {r[2]}" for r in rows)
    try:
        resp = oc.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a bank analyst assistant. Answer the "
                 "question using ONLY the provided context snippets."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {args.query}"},
            ],
            max_completion_tokens=400,
        )
        answer = (resp.choices[0].message.content or "").strip() or "(model returned no text)"
    except Exception as e:
        # The leak is demonstrated by the retrieved-rows table below regardless.
        answer = f"(LLM answer skipped: {e})"

    print(f"\nQuery as {args.tenant}: {args.query}")
    print(f"Embeddings column: {col}\n")
    print("ANSWER:")
    print(answer)
    print("\nRETRIEVED CHUNKS (rank, tenant, doc, distance):")
    leaks = 0
    for rank, (tenant_id, doc_id, _chunk, dist) in enumerate(rows, 1):
        flag = ""
        if tenant_id != args.tenant:
            flag = "   <-- LEAK (other tenant)"
            leaks += 1
        print(f"  {rank}. {tenant_id:8s} doc#{doc_id:<4d} dist={dist:.4f}{flag}")
    print(f"\nCross-tenant rows retrieved: {leaks}/{len(rows)}"
          + ("   *** DATA LEAK ***" if leaks else "   (clean)"))


if __name__ == "__main__":
    main()
