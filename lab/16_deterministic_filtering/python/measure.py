"""Measure retrieval quality for the three text states: raw, redacted, tokenized.

Each question is transformed the same way its state's documents were (redacted
or tokenized with the same dictionary), embedded locally, and run through
bank.retrieve() as app_agent with its bank as tenant, so row-level security
applies exactly as it does for the agent. Recall and nDCG at k = 1, 3, 5, 10,
per method (dense, sparse, hybrid, entity) and question set (entity, generic, all).

Results go to gov.quality_runs (read by the maturity scorecard) and to
results/measure.json.

    python python/measure.py
"""
import importlib.util
import json

from _common import (DATA_DIR, REPO_ROOT, RESULTS_DIR, agent_conn, embed_queries, gateway_conn,
                     sparse_embed, sparse_literal, tenant_cursor, vec_literal)
from filtering import Dictionary

# Reuse the corrected nDCG from lab/evaluation without importing the lab package.
_spec = importlib.util.spec_from_file_location("metrics", REPO_ROOT / "lab" / "evaluation" / "metrics.py")
metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrics)

STATES = ["raw", "redacted", "tokenized"]
METHODS = ["dense", "sparse", "hybrid", "entity"]
KS = [1, 3, 5, 10]


def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def main():
    questions = json.loads((DATA_DIR / "questions.json").read_text())
    gw_conn = gateway_conn()
    with gw_conn.cursor() as gw:
        dictionary = Dictionary.load(gw)
        gw.execute("SELECT state, version_id FROM bank.embedding_versions WHERE is_active")
        versions = dict(gw.fetchall())
    missing = [s for s in STATES if s not in versions]
    if missing:
        raise SystemExit(f"No active embedding version for: {', '.join(missing)}. "
                         f"Let python/embed.py finish (see bank.embedding_versions), then run again.")
    transform = {"raw": lambda q: q, "redacted": dictionary.redact, "tokenized": dictionary.tokenize}
    # entities the question names, from the label-derived dictionary; a redacted question names none
    named = {"raw": lambda q: sorted(dictionary.mentions(q)), "redacted": lambda q: [],
             "tokenized": lambda q: sorted(dictionary.mentions(q))}

    agent = agent_conn()
    per_question = []
    for state in STATES:
        texts = [transform[state](q["question"]) for q in questions]
        dense = embed_queries(texts)
        sparse = sparse_embed(texts)
        for q, text, d, s in zip(questions, texts, dense, sparse):
            row = dict(state=state, bank_id=q["bank_id"], type=q["type"], question=text,
                       expected=q["expected"], retrieved={})
            tokens = named[state](q["question"])
            for method in METHODS:
                with tenant_cursor(agent, q["bank_id"]) as cur:
                    cur.execute("SELECT doc_id FROM bank.retrieve(%s::vector, %s::sparsevec, %s, %s, 10, %s)",
                                (vec_literal(d), sparse_literal(s), state, method, tokens))
                    row["retrieved"][method] = [r[0] for r in cur.fetchall()]
            per_question.append(row)

    summary = []
    for state in STATES:
        for method in METHODS:
            for qset in ["entity", "generic", "all"]:
                rows = [r for r in per_question if r["state"] == state and (qset == "all" or r["type"] == qset)]
                for k in KS:
                    rec = sum(recall_at_k(r["retrieved"][method], r["expected"], k) for r in rows) / len(rows)
                    ndcg = sum(metrics.ndcg_at_k_binary(r["retrieved"][method], r["expected"], k)
                               for r in rows) / len(rows)
                    summary.append(dict(state=state, method=method, question_set=qset, k=k,
                                        recall=round(rec, 3), ndcg=round(ndcg, 3), n=len(rows)))

    with gw_conn.cursor() as gw:
        gw.executemany("INSERT INTO gov.quality_runs (version_id, state, method, question_set, k, recall, ndcg, "
                       "n_questions) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                       [(versions[s["state"]], s["state"], s["method"], s["question_set"], s["k"],
                         s["recall"], s["ndcg"], s["n"]) for s in summary])
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "measure.json").write_text(json.dumps(dict(summary=summary, questions=per_question), indent=1))

    print(f"{'state':<10} {'method':<7} {'set':<8} {'recall@5':>9} {'nDCG@5':>7} {'recall@10':>10} {'nDCG@10':>8}")
    for s5 in (s for s in summary if s["k"] == 5):
        s10 = next(s for s in summary if s["k"] == 10 and all(s[x] == s5[x] for x in ("state", "method", "question_set")))
        print(f"{s5['state']:<10} {s5['method']:<7} {s5['question_set']:<8} {s5['recall']:>9.3f} {s5['ndcg']:>7.3f} "
              f"{s10['recall']:>10.3f} {s10['ndcg']:>8.3f}")


if __name__ == "__main__":
    main()
