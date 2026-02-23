#!/usr/bin/env python3
"""
50_evaluate_rag.py — RAG Quality Evaluation

Runs golden benchmark queries through Agent 1, compares results against
expected tables/columns/KPIs, and computes IR metrics:

  - Precision@K: fraction of retrieved items that are relevant
  - Recall@K:    fraction of relevant items that are retrieved
  - nDCG@K:      normalized Discounted Cumulative Gain (position-aware)
  - MRR:         Mean Reciprocal Rank of first relevant result
  - MAP:         Mean Average Precision

Results stored in rag_monitor.evaluation_runs + rag_monitor.relevance_judgments.

Usage:
    python python/50_evaluate_rag.py [--k 5] [--verbose]
"""

import sys, os, json, math, argparse, time
import psycopg2, psycopg2.extras
from typing import List, Dict, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))
import config
from importlib import import_module
agent_rag = import_module("20_agent_rag_search")

# ---------------------------------------------------------------------------
# IR METRICS
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """Fraction of top-k retrieved items that are relevant."""
    top = retrieved[:k]
    if not top: return 0.0
    return sum(1 for r in top if r in relevant) / len(top)

def recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """Fraction of relevant items found in top-k."""
    if not relevant: return 1.0
    top = set(retrieved[:k])
    return sum(1 for r in relevant if r in top) / len(relevant)

def dcg_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """Discounted Cumulative Gain."""
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        rel = 1.0 if item in relevant else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg

def ndcg_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """Normalized DCG: DCG / ideal DCG."""
    dcg = dcg_at_k(retrieved, relevant, k)
    # Ideal: all relevant items at top positions
    ideal = sorted(retrieved[:k], key=lambda x: x in relevant, reverse=True)
    ideal_dcg = dcg_at_k(ideal, relevant, k)
    # Also account for relevant items NOT in retrieved
    n_relevant_in_k = min(len(relevant), k)
    ideal_dcg_full = sum(1.0 / math.log2(i + 2) for i in range(n_relevant_in_k))
    if ideal_dcg_full == 0: return 1.0
    return dcg / ideal_dcg_full

def reciprocal_rank(retrieved: List[str], relevant: set) -> float:
    """Reciprocal rank of first relevant result."""
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0

def average_precision(retrieved: List[str], relevant: set) -> float:
    """Average precision for a single query."""
    if not relevant: return 1.0
    hits = 0
    sum_prec = 0.0
    for i, item in enumerate(retrieved):
        if item in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / len(relevant) if relevant else 0.0

# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query: str
    precision_5: float
    precision_10: float
    recall_5: float
    recall_10: float
    ndcg_5: float
    ndcg_10: float
    rr: float
    ap: float
    latency_ms: int
    retrieved_tables: List[str]
    expected_tables: List[str]
    retrieved_kpi: str
    expected_kpi: str


def evaluate_query(query_data: Dict, k_values: List[int] = [5, 10], verbose: bool = False) -> QueryResult:
    """Run one query through Agent 1 and evaluate."""
    query = query_data["query"]
    expected = query_data["expected"]
    expected_tables = set(expected.get("tables", []))
    expected_columns = set(expected.get("columns", []))
    expected_kpi = expected.get("kpi")

    # Run Agent 1
    t0 = time.time()
    rec = agent_rag.search(query)
    latency = int((time.time() - t0) * 1000)

    # Extract retrieved items
    retrieved_tables = [t["table_name"] for t in rec.tables]
    retrieved_columns = [f"{c['table_name']}.{c['column_name']}" for c in rec.columns]
    retrieved_kpis = [k["kpi_name"] for k in rec.kpi_patterns]

    # Combine tables + columns for overall metric
    all_relevant = expected_tables | expected_columns
    all_retrieved = retrieved_tables + retrieved_columns

    result = QueryResult(
        query=query,
        precision_5=precision_at_k(all_retrieved, all_relevant, 5),
        precision_10=precision_at_k(all_retrieved, all_relevant, 10),
        recall_5=recall_at_k(all_retrieved, all_relevant, 5),
        recall_10=recall_at_k(all_retrieved, all_relevant, 10),
        ndcg_5=ndcg_at_k(all_retrieved, all_relevant, 5),
        ndcg_10=ndcg_at_k(all_retrieved, all_relevant, 10),
        rr=reciprocal_rank(all_retrieved, all_relevant),
        ap=average_precision(all_retrieved, all_relevant),
        latency_ms=latency,
        retrieved_tables=retrieved_tables,
        expected_tables=list(expected_tables),
        retrieved_kpi=retrieved_kpis[0] if retrieved_kpis else "",
        expected_kpi=expected_kpi or "",
    )

    if verbose:
        kpi_match = "✓" if expected_kpi and expected_kpi in retrieved_kpis else ("✗" if expected_kpi else "—")
        tbl_found = expected_tables & set(retrieved_tables)
        tbl_miss = expected_tables - set(retrieved_tables)
        print(f"\n   Q: {query[:70]}")
        print(f"   P@5={result.precision_5:.2f} R@5={result.recall_5:.2f} nDCG@5={result.ndcg_5:.2f} RR={result.rr:.2f} | {latency}ms")
        print(f"   Tables found: {tbl_found or '∅'} | missed: {tbl_miss or '∅'}")
        print(f"   KPI: {kpi_match} expected={expected_kpi} got={result.retrieved_kpi or '—'}")

    return result


def run_evaluation(golden_path: str, k: int = 5, verbose: bool = False):
    """Run full evaluation suite."""
    with open(golden_path) as f:
        queries = json.load(f)

    print("="*70)
    print(f"  RAG Quality Evaluation — {len(queries)} benchmark queries")
    print(f"  Model: {config.EMBEDDING_MODEL} | Threshold: {config.SIMILARITY_THRESHOLD} | Top-K: {config.TOP_K}")
    print("="*70)

    results = []
    for qd in queries:
        r = evaluate_query(qd, verbose=verbose)
        results.append(r)

    # Aggregate metrics
    n = len(results)
    sorted_latencies = sorted(r.latency_ms for r in results)
    metrics = {
        "precision_at_5":  sum(r.precision_5 for r in results) / n,
        "precision_at_10": sum(r.precision_10 for r in results) / n,
        "recall_at_5":     sum(r.recall_5 for r in results) / n,
        "recall_at_10":    sum(r.recall_10 for r in results) / n,
        "ndcg_at_5":       sum(r.ndcg_5 for r in results) / n,
        "ndcg_at_10":      sum(r.ndcg_10 for r in results) / n,
        "mrr":             sum(r.rr for r in results) / n,
        "map":             sum(r.ap for r in results) / n,
        "avg_latency_ms":  sum(r.latency_ms for r in results) / n,
        "p50_latency_ms":  sorted_latencies[n // 2],
        "p95_latency_ms":  sorted_latencies[min(int(n * 0.95), n - 1)],
        "p99_latency_ms":  sorted_latencies[min(int(n * 0.99), n - 1)],
    }

    # KPI match rate
    kpi_queries = [r for r in results if r.expected_kpi]
    kpi_match = sum(1 for r in kpi_queries if r.expected_kpi == r.retrieved_kpi) / len(kpi_queries) if kpi_queries else 0

    # Print summary
    print(f"\n{'='*70}")
    print(f"  RESULTS ({n} queries)")
    print(f"{'='*70}")
    print(f"  Precision@5:    {metrics['precision_at_5']:.3f}")
    print(f"  Precision@10:   {metrics['precision_at_10']:.3f}")
    print(f"  Recall@5:       {metrics['recall_at_5']:.3f}")
    print(f"  Recall@10:      {metrics['recall_at_10']:.3f}")
    print(f"  nDCG@5:         {metrics['ndcg_at_5']:.3f}")
    print(f"  nDCG@10:        {metrics['ndcg_at_10']:.3f}")
    print(f"  MRR:            {metrics['mrr']:.3f}")
    print(f"  MAP:            {metrics['map']:.3f}")
    print(f"  KPI match rate: {kpi_match:.1%} ({sum(1 for r in kpi_queries if r.expected_kpi == r.retrieved_kpi)}/{len(kpi_queries)})")
    print(f"\n  Latency (ms):   avg={metrics['avg_latency_ms']:.0f}  p50={metrics['p50_latency_ms']}  p95={metrics['p95_latency_ms']}  p99={metrics['p99_latency_ms']}")
    print(f"{'='*70}")

    # Store in PG
    try:
        conn = psycopg2.connect(config.DATABASE_URL)
        cur = conn.cursor()

        # Store relevance judgments
        for qd, r in zip(queries, results):
            for tbl in qd["expected"].get("tables", []):
                grade = 3 if tbl in r.retrieved_tables else 0
                cur.execute("""
                    INSERT INTO rag_monitor.relevance_judgments
                    (query_text, catalog_type, catalog_item, relevance_grade, judged_by, judgment_method)
                    VALUES (%s, 'table', %s, %s, 'golden_set', 'manual')
                    ON CONFLICT (query_text, catalog_type, catalog_item, judged_by) DO UPDATE
                    SET relevance_grade = EXCLUDED.relevance_grade
                """, (qd["query"], tbl, grade))

        # Store evaluation run
        per_query = [{"query": r.query[:80], "p5": round(r.precision_5,3),
                      "r5": round(r.recall_5,3), "ndcg5": round(r.ndcg_5,3),
                      "latency": r.latency_ms} for r in results]
        cur.execute("""
            INSERT INTO rag_monitor.evaluation_runs
            (run_name, embedding_model, similarity_threshold, top_k,
             num_queries, num_judgments,
             precision_at_5, precision_at_10, recall_at_5, recall_at_10,
             ndcg_at_5, ndcg_at_10, mrr, map,
             avg_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms,
             per_query_metrics, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (f"eval_{config.EMBEDDING_MODEL}", config.EMBEDDING_MODEL,
              config.SIMILARITY_THRESHOLD, config.TOP_K,
              n, n * 3,  # approximate judgments
              metrics["precision_at_5"], metrics["precision_at_10"],
              metrics["recall_at_5"], metrics["recall_at_10"],
              metrics["ndcg_at_5"], metrics["ndcg_at_10"],
              metrics["mrr"], metrics["map"],
              metrics["avg_latency_ms"], metrics["p50_latency_ms"],
              metrics["p95_latency_ms"], metrics["p99_latency_ms"],
              json.dumps(per_query),
              f"Embedding: {config.EMBEDDING_MODEL}, threshold: {config.SIMILARITY_THRESHOLD}"))

        conn.commit()
        conn.close()
        print(f"\n✅ Results stored in rag_monitor.evaluation_runs")
    except Exception as e:
        print(f"\n⚠ Could not store results: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--golden", default=os.path.join(os.path.dirname(__file__), "..", "benchmarks", "golden_queries.json"))
    args = parser.parse_args()

    if not config.VOYAGE_API_KEY:
        print("❌ Set VOYAGE_API_KEY"); sys.exit(1)

    run_evaluation(args.golden, args.k, args.verbose)

if __name__ == "__main__":
    main()
