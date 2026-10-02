"""Answer-level relevance: the same 36 questions answered naive and governed.

For each question and mode, the agent answers with document citations ([doc N]).
Scores, all deterministic (no LLM judge):
  cited recall     expected documents cited / expected documents
  cited precision  expected documents cited / documents cited
  context recall   expected documents the tools returned / expected documents
  explicit entity  the answer explicitly names the entity the question asked about
                   (governed: the token); "the client" does not count
  leaked values    scanner hits in requests whose outcome is 'sent' (gov.egress_log,
                   rows tagged with this run's label)

    python python/evaluate_answers.py [--label baseline] [--limit N]
Writes results/answers_<label>.json and rows in gov.answer_runs.
"""
import argparse
import json
import re
import time

from _common import DATA_DIR, RESULTS_DIR, gateway_conn
from agent import run
from filtering import Dictionary
from tracing import langfuse

CITE = re.compile(r"\bdoc\s*#?\s*(\d+)", re.IGNORECASE)
MODES = ["off", "tokenized"]


def score(q, result, question_tokens, dictionary, mode):
    expected = set(q["expected"])
    if result is None:
        return dict(blocked=True, cited_recall=0.0, cited_precision=0.0, context_recall=0.0,
                    explicit_entity=False, retrieved=[])
    cited = {int(n) for n in CITE.findall(result["answer"])}
    hit = cited & expected
    if mode == "tokenized":
        answer_tokens = set(re.findall(r"\b[A-Z]+_[0-9a-f]{12}\b", result["answer"]))
    else:
        answer_tokens = set(dictionary.mentions(result["answer"]))
    return dict(blocked=False,
                cited_recall=len(hit) / len(expected),
                cited_precision=len(hit) / len(cited) if cited else 0.0,
                context_recall=len(set(result["seen_docs"]) & expected) / len(expected),
                explicit_entity=question_tokens <= answer_tokens if question_tokens else None,
                retrieved=result["seen_docs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    questions = json.loads((DATA_DIR / "questions.json").read_text())[:args.limit]
    gw_conn = gateway_conn()
    with gw_conn.cursor() as cur:
        dictionary = Dictionary.load(cur)

    rows, usage = [], {"prompt": 0, "completion": 0}
    t0 = time.time()
    for i, q in enumerate(questions, 1):
        question_tokens = set(dictionary.mentions(q["question"]))
        for mode in MODES:
            result = run(q["question"], q["bank_id"], mode, reviewer=False, quiet=True, run_label=args.label)
            s = score(q, result, question_tokens, dictionary, mode)
            if result:  # the deterministic scores, attached to the question's trace
                for name in ("cited_recall", "cited_precision", "context_recall"):
                    langfuse.create_score(trace_id=result["trace_id"], name=name, value=s[name])
                langfuse.create_score(trace_id=result["trace_id"], name="sensitive_values_sent",
                                      value=result["values_sent"])
            if result:
                usage["prompt"] += result["usage"]["prompt"]
                usage["completion"] += result["usage"]["completion"]
            rows.append(dict(q, mode=mode, **s,
                             answer=result["answer"] if result else None,
                             readable=result["readable"] if result else None))
        print(f"\r  {i}/{len(questions)} questions, {time.time() - t0:.0f}s", end="", flush=True)
    print()
    langfuse.flush()

    with gw_conn.cursor() as cur:
        cur.execute("SELECT filtering, count(*) FILTER (WHERE outcome = 'sent'), "
                    "coalesce(sum(hits) FILTER (WHERE outcome = 'sent'), 0), "
                    "count(*) FILTER (WHERE outcome = 'blocked'), count(*) FILTER (WHERE outcome IN ('failed', 'attempted')) "
                    "FROM gov.egress_log WHERE run_label = %s GROUP BY filtering", (args.label,))
        egress = {f: dict(sent=n, values_sent=leaked, blocked=b, failed=fl) for f, n, leaked, b, fl in cur.fetchall()}

    summary = []
    for mode in MODES:
        for qset in ["entity", "generic", "all"]:
            sel = [r for r in rows if r["mode"] == mode and (qset == "all" or r["type"] == qset)]
            if not sel:
                continue
            ent = [r for r in sel if r["explicit_entity"] is not None]
            summary.append(dict(
                mode=mode, question_set=qset, n=len(sel),
                cited_recall=round(sum(r["cited_recall"] for r in sel) / len(sel), 3),
                cited_precision=round(sum(r["cited_precision"] for r in sel) / len(sel), 3),
                context_recall=round(sum(r["context_recall"] for r in sel) / len(sel), 3),
                explicit_entity=round(sum(r["explicit_entity"] for r in ent) / len(ent), 3) if ent else None,
                blocked=sum(r["blocked"] for r in sel)))

    with gw_conn.cursor() as cur:
        cur.executemany("INSERT INTO gov.answer_runs (label, mode, question_set, n_questions, cited_recall, "
                        "cited_precision, context_recall, explicit_entity, blocked) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        [(args.label, s["mode"], s["question_set"], s["n"], s["cited_recall"],
                          s["cited_precision"], s["context_recall"], s["explicit_entity"], s["blocked"])
                         for s in summary])
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"answers_{args.label}.json").write_text(
        json.dumps(dict(summary=summary, egress=egress, usage=usage, rows=rows), indent=1, default=str))

    print(f"{'mode':<10} {'set':<8} {'cited rec':>9} {'cited prec':>10} {'context rec':>11} {'expl. entity':>12} {'blocked':>7}")
    for s in summary:
        print(f"{s['mode']:<10} {s['question_set']:<8} {s['cited_recall']:>9.3f} {s['cited_precision']:>10.3f} "
              f"{s['context_recall']:>11.3f} {str(s['explicit_entity']):>12} {s['blocked']:>7}")
    print(f"egress: {egress}")
    print(f"tokens: {usage}")


if __name__ == "__main__":
    main()
