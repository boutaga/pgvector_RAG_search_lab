#!/usr/bin/env python3
"""
20_agent_rag_search.py — Agent 1: RAG Metadata Search

Searches the pgvector metadata catalog to find relevant tables, columns,
relationships, and KPI patterns for a business question.

CRITICAL: This agent NEVER touches raw data. Only catalog.* (metadata + embeddings).
The detail_bi and detail_agent fields provide rich semantic context for better matching.

Output: Structured recommendation consumed by Agent 2 (Pipeline).
"""

import sys
import json
import time
import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
import config
from embedding_provider import get_embedding_provider
from llm_provider import get_llm_provider


@dataclass
class SearchRecommendation:
    question: str
    tables: List[Dict]
    columns: List[Dict]
    join_paths: List[Dict]
    kpi_patterns: List[Dict]
    max_classification: str
    pii_fields: List[str]
    reasoning: str
    embedding_time_ms: int
    search_time_ms: int
    total_time_ms: int


def embed_query(text: str, embedding_model: str = None) -> List[float]:
    """Embed with input_type='query' for asymmetric retrieval."""
    provider = get_embedding_provider(embedding_model)
    return provider.embed_query(text)


def vector_search(conn, query_emb: List[float]) -> Dict[str, List[Dict]]:
    """Vector similarity search across all catalog tables."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    emb_str = '[' + ','.join(str(x) for x in query_emb) + ']'
    results = {}

    cur.execute("""
        SELECT id, table_name, description, detail_bi, detail_agent,
               classification, contains_pii, row_count, column_count, lake_path,
               1 - (embedding <=> %s::vector) AS similarity
        FROM catalog.table_metadata
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (emb_str, emb_str, config.TOP_K))
    results["tables"] = [dict(r) for r in cur.fetchall()
                         if r["similarity"] >= config.SIMILARITY_THRESHOLD]

    cur.execute("""
        SELECT id, table_name, column_name, data_type,
               is_primary_key, is_foreign_key, referenced_table, referenced_column,
               is_pii, classification, masking_rule, detail_bi, detail_agent,
               sample_values, n_distinct,
               1 - (embedding <=> %s::vector) AS similarity
        FROM catalog.column_metadata
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (emb_str, emb_str, config.TOP_K * 2))
    results["columns"] = [dict(r) for r in cur.fetchall()
                          if r["similarity"] >= config.SIMILARITY_THRESHOLD]

    cur.execute("""
        SELECT id, source_table, source_column, target_table, target_column,
               relationship_type, join_condition,
               1 - (embedding <=> %s::vector) AS similarity
        FROM catalog.relationship_metadata
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT 10
    """, (emb_str, emb_str))
    results["joins"] = [dict(r) for r in cur.fetchall()
                        if r["similarity"] >= config.SIMILARITY_THRESHOLD * 0.8]

    cur.execute("""
        SELECT id, kpi_name, kpi_description, domain, required_tables,
               required_columns, sql_template, classification,
               1 - (embedding <=> %s::vector) AS similarity
        FROM catalog.kpi_patterns
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT 5
    """, (emb_str, emb_str))
    results["kpis"] = [dict(r) for r in cur.fetchall()
                       if r["similarity"] >= config.SIMILARITY_THRESHOLD]

    cur.close()
    return results


def llm_reason(question: str, results: Dict, llm_model: str = None) -> str:
    """LLM reasoning over search results."""
    context = json.dumps({
        "tables": [{"name": t["table_name"], "sim": round(t["similarity"], 3),
                     "class": t["classification"], "bi": (t.get("detail_bi") or "")[:100]}
                    for t in results.get("tables", [])],
        "columns": [{"col": f"{c['table_name']}.{c['column_name']}",
                      "type": c["data_type"], "pii": c.get("is_pii"),
                      "sim": round(c["similarity"], 3)}
                     for c in results.get("columns", [])[:15]],
        "kpis": [{"name": k["kpi_name"], "domain": k["domain"],
                   "sim": round(k["similarity"], 3)}
                  for k in results.get("kpis", [])],
    }, indent=2)

    provider = get_llm_provider(llm_model or config.CHAT_MODEL_FAST)
    return provider.chat(
        system="You are a metadata analyst for a Swiss private bank. "
               "Given search results from the metadata catalog, explain which tables, columns, and joins "
               "are needed. Note governance concerns (PII, classification). Be concise (3-5 sentences).",
        user=f"Question: {question}\n\nResults:\n{context}",
        max_tokens=400, temperature=0.1,
    )


def search(question: str, requester: str = None, role: str = None,
           embedding_model: str = None, llm_model: str = None) -> SearchRecommendation:
    """Main entry point for Agent 1."""
    t0 = time.time()
    conn = psycopg2.connect(config.DATABASE_URL)

    try:
        # Embed
        t_emb = time.time()
        query_emb = embed_query(question, embedding_model=embedding_model)
        embedding_ms = int((time.time() - t_emb) * 1000)

        # Search
        t_search = time.time()
        results = vector_search(conn, query_emb)
        search_ms = int((time.time() - t_search) * 1000)

        # Governance
        classifications = [t.get("classification", "internal") for t in results.get("tables", [])]
        classifications += [c.get("classification", "internal") for c in results.get("columns", [])]
        pii_fields = [f"{c['table_name']}.{c['column_name']}" for c in results.get("columns", []) if c.get("is_pii")]
        class_order = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
        max_class = max(classifications, key=lambda x: class_order.get(x, 3)) if classifications else "internal"

        # Reason
        t_reason = time.time()
        reasoning = llm_reason(question, results, llm_model=llm_model)
        reasoning_ms = int((time.time() - t_reason) * 1000)

        # Log to monitoring
        total_ms = int((time.time() - t0) * 1000)
        try:
            cur = conn.cursor()
            # Resolve actual model names for logging
            emb_model_used = embedding_model or config.EMBEDDING_MODEL
            llm_model_used = llm_model or config.CHAT_MODEL_FAST
            cur.execute("""
                INSERT INTO rag_monitor.search_log
                (query_text, query_embedding, result_tables, result_columns, result_kpis,
                 total_results, max_classification, pii_fields_found,
                 embedding_time_ms, search_time_ms, reasoning_time_ms, total_time_ms,
                 requester, requester_role, similarity_threshold, top_k,
                 embedding_model, llm_model)
                VALUES (%s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (question, '[' + ','.join(str(x) for x in query_emb) + ']',
                  json.dumps([{"table": t["table_name"], "sim": round(t["similarity"],3)} for t in results.get("tables",[])]),
                  json.dumps([{"col": f"{c['table_name']}.{c['column_name']}", "sim": round(c["similarity"],3)} for c in results.get("columns",[])[:20]]),
                  json.dumps([{"kpi": k["kpi_name"], "sim": round(k["similarity"],3)} for k in results.get("kpis",[])]),
                  sum(len(v) for v in results.values()),
                  max_class, pii_fields,
                  embedding_ms, search_ms, reasoning_ms, total_ms,
                  requester, role, config.SIMILARITY_THRESHOLD, config.TOP_K,
                  emb_model_used, llm_model_used))
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"   ⚠ Could not log search: {e}")

        return SearchRecommendation(
            question=question,
            tables=results.get("tables", []),
            columns=results.get("columns", []),
            join_paths=results.get("joins", []),
            kpi_patterns=results.get("kpis", []),
            max_classification=max_class,
            pii_fields=pii_fields,
            reasoning=reasoning,
            embedding_time_ms=embedding_ms,
            search_time_ms=search_ms,
            total_time_ms=total_ms,
        )
    finally:
        conn.close()


def print_recommendation(rec: SearchRecommendation):
    cls_icon = {"public":"🟢","internal":"🔵","confidential":"🟠","restricted":"🔴"}
    print("="*70)
    print(f"🔍 AGENT 1: RAG Metadata Search")
    print(f"   Question: {rec.question}")
    print(f"   Embed: {rec.embedding_time_ms}ms | Search: {rec.search_time_ms}ms | Total: {rec.total_time_ms}ms")
    print("="*70)
    print(f"\n📋 TABLES ({len(rec.tables)}):")
    for t in rec.tables:
        ic = cls_icon.get(t["classification"],"⚪")
        print(f"   {ic} {t['table_name']} (sim={t['similarity']:.3f}) [{t['classification']}]")
        if t.get("detail_bi"):
            print(f"      └─ {t['detail_bi'][:90]}...")
    print(f"\n📊 COLUMNS ({len(rec.columns)}):")
    for c in rec.columns[:12]:
        pii = " 🔒PII" if c.get("is_pii") else ""
        print(f"   {c['table_name']}.{c['column_name']} ({c['data_type']}) sim={c['similarity']:.3f}{pii}")
    if rec.kpi_patterns:
        print(f"\n📐 KPI PATTERNS:")
        for k in rec.kpi_patterns:
            print(f"   ✦ {k['kpi_name']} [{k['domain']}] sim={k['similarity']:.3f}")
    print(f"\n🔐 GOVERNANCE: {cls_icon.get(rec.max_classification,'⚪')} {rec.max_classification}")
    if rec.pii_fields:
        print(f"   PII: {', '.join(rec.pii_fields)}")
    print(f"\n🤖 REASONING: {rec.reasoning}")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 20_agent_rag_search.py \"<question>\"")
        sys.exit(1)
    rec = search(sys.argv[1])
    print_recommendation(rec)
