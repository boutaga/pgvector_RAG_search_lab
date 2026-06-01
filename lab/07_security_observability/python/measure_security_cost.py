"""Measure the cost of each security control across three states, at several k.

States:
  baseline : embeddings has NO RLS  -> leaks other tenants
  rls      : embeddings RLS applied -> no leak
  masked   : RLS applied, search the masked embeddings -> recall cost

For each (state, query) it retrieves the top-10 once, then derives Recall / Precision /
nDCG and the leak rate at k in {1,3,5,10}, plus a per-query detail of the retrieved rows
(relevant / leak flags + snippet). Writes mock/frozen_run.json (the dashboard's offline
source). Reuses the lab's nDCG implementation from lab/evaluation/metrics.py.
"""
import json
from datetime import datetime
from pathlib import Path

from _common import (admin_conn, app_conn, openai_client, embed_texts,
                     vec_literal, LAB_DIR, REPO_ROOT)

# Load the lab's nDCG implementation directly from its file, bypassing
# lab/evaluation/__init__.py (which imports core.database -> pgvector, not needed here).
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "lab07_metrics", REPO_ROOT / "lab" / "evaluation" / "metrics.py")
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
ndcg_at_k_binary = _metrics.ndcg_at_k_binary

K_VALUES = [1, 3, 5, 10]
MAX_K = max(K_VALUES)
DEFAULT_K = 5
DEMO_SQL = LAB_DIR / "sql" / "demo"


def apply_sql(filename):
    admin_conn().cursor().execute((DEMO_SQL / filename).read_text())


def _mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def retrieve(tenant, qv, vec_col, snip_col, k):
    cur = app_conn(tenant).cursor()
    cur.execute(
        f"SELECT document_id, tenant_id, {snip_col}, {vec_col} <=> %s::vector AS dist "
        f"FROM app.embeddings ORDER BY {vec_col} <=> %s::vector LIMIT %s",
        (qv, qv, k),
    )
    return cur.fetchall()  # (doc_id, tenant, snippet, dist)


def score_state(vec_col, snip_col, cases, qvecs):
    per_query = []
    agg = {k: {"recall": [], "precision": [], "ndcg": [], "leak_q": 0, "leak_docs": 0}
           for k in K_VALUES}
    for case, qv in zip(cases, qvecs):
        rows = retrieve(case["tenant_id"], qv, vec_col, snip_col, MAX_K)
        rel = set(case["expected_doc_ids"])
        ids = [r[0] for r in rows]
        retrieved = [{"rank": i + 1, "doc_id": r[0], "tenant": r[1],
                      "dist": round(float(r[3]), 4), "relevant": r[0] in rel,
                      "leak": r[1] != case["tenant_id"], "snippet": (r[2] or "")[:90]}
                     for i, r in enumerate(rows)]
        byk = {}
        for k in K_VALUES:
            topk = ids[:k]
            hits = sum(1 for d in topk if d in rel)
            byk[str(k)] = {"recall": round(hits / len(rel), 4) if rel else 0.0,
                           "precision": round(hits / k, 4),
                           "ndcg": round(ndcg_at_k_binary(topk, list(rel), k), 4)}
            leak_in_k = sum(1 for r in retrieved[:k] if r["leak"])
            agg[k]["recall"].append(byk[str(k)]["recall"])
            agg[k]["precision"].append(byk[str(k)]["precision"])
            agg[k]["ndcg"].append(byk[str(k)]["ndcg"])
            agg[k]["leak_docs"] += leak_in_k
            agg[k]["leak_q"] += 1 if leak_in_k else 0
        per_query.append({"retrieved": retrieved, "by_k": byk})
    n = len(cases)
    state_by_k = {str(k): {"recall": _mean(agg[k]["recall"]),
                           "precision": _mean(agg[k]["precision"]),
                           "ndcg": _mean(agg[k]["ndcg"]),
                           "leak_rate": round(agg[k]["leak_q"] / n, 4),
                           "leak_docs": agg[k]["leak_docs"]} for k in K_VALUES}
    return state_by_k, per_query


def main():
    cases = json.loads((LAB_DIR / "data" / "test_cases.json").read_text())

    cur = admin_conn().cursor()
    cur.execute("SELECT count(*) FROM app.embeddings WHERE embedding_masked IS NULL")
    if cur.fetchone()[0] > 0:
        raise SystemExit("Some embedding_masked are NULL. Run embed.py --mode masked first.")

    qvecs = [vec_literal(v) for v in embed_texts(openai_client(), [c["query"] for c in cases])]

    queries_out = [{"query": c["query"], "tenant_id": c["tenant_id"],
                    "query_type": c.get("query_type", ""),
                    "expected_doc_ids": c["expected_doc_ids"], "states": {}}
                   for c in cases]

    state_defs = [
        ("baseline", "embedding", "chunk_text", "Baseline (no RLS on embeddings)", None),
        ("rls", "embedding", "chunk_text", "+ RLS on embeddings", "10_rls_embeddings.sql"),
        ("masked", "embedding_masked", "chunk_text_masked", "+ Anonymization (client names)", None),
    ]

    apply_sql("99_rollback.sql")  # baseline: no RLS on embeddings, no masking
    states_out = []
    for key, vcol, scol, label, setup in state_defs:
        if setup:
            apply_sql(setup)
        state_by_k, per_query = score_state(vcol, scol, cases, qvecs)
        states_out.append({"key": key, "label": label, "by_k": state_by_k})
        for i, rec in enumerate(per_query):
            queries_out[i]["states"][key] = rec
    apply_sql("99_rollback.sql")  # leave DB in baseline for rehearsal

    out = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "k_values": K_VALUES, "default_k": DEFAULT_K, "n_queries": len(cases),
           "states": states_out, "queries": queries_out}
    (LAB_DIR / "mock").mkdir(exist_ok=True)
    dest = LAB_DIR / "mock" / "frozen_run.json"
    dest.write_text(json.dumps(out, indent=2))

    dk = str(DEFAULT_K)
    print(f"\nSecurity cost over {len(cases)} queries (k={DEFAULT_K}):\n")
    print(f"  {'state':36s} {'Recall':>7s} {'Prec':>7s} {'nDCG':>7s} {'Leak%':>7s}")
    for s in states_out:
        m = s["by_k"][dk]
        print(f"  {s['label']:36s} {m['recall']:7.3f} {m['precision']:7.3f} "
              f"{m['ndcg']:7.3f} {m['leak_rate']*100:6.1f}%")
    print(f"\nWrote {dest}  (k_values={K_VALUES})")


if __name__ == "__main__":
    main()
