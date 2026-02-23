#!/usr/bin/env python3
"""
40_demo.py — End-to-End Demo: NL → RAG Search → Governed Data Mart

Runs 3 scenarios showing different governance outcomes:
  1. BI Analyst     → exposure by sector     → internal     → all roles, PII masked
  2. Risk Manager   → clients near limits    → confidential → risk+compliance only
  3. Compliance Off  → AML investigation     → restricted   → compliance only, full PII

Usage:
    python python/40_demo.py [--dry-run] [--scenario 1|2|3|all]
"""

import sys, os, argparse, json
import psycopg2, psycopg2.extras
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(__file__))
import config
from importlib import import_module
agent_rag = import_module("20_agent_rag_search")
agent_pipe = import_module("30_agent_pipeline")

SCENARIOS = [
    {"id":1, "title":"Portfolio Exposure Dashboard",
     "requester":"maria.chen", "role":"bi_analyst",
     "question":"I need a dashboard showing portfolio market value exposure by asset class, sector, and client segment"},
    {"id":2, "title":"Risk Limit Monitoring",
     "requester":"thomas.weber", "role":"risk_manager",
     "question":"Show me clients above 70% risk limit utilization with VaR metrics and exposure breakdown"},
    {"id":3, "title":"AML Investigation View",
     "requester":"sophie.martin", "role":"compliance_officer",
     "question":"All open and escalated AML alerts with full client details, PEP status, and trading activity for SAR investigation"},
]

def run_scenario(s, dry_run=False):
    print(f"\n{'█'*70}")
    print(f"  SCENARIO {s['id']}: {s['title']}")
    print(f"  Requester: {s['requester']} ({s['role']})")
    print(f"  Question: {s['question']}")
    print(f"{'█'*70}")

    # Agent 1
    rec = agent_rag.search(s["question"], s["requester"], s["role"])
    agent_rag.print_recommendation(rec)

    # Agent 2
    result = agent_pipe.provision(
        question=s["question"], recommendation=asdict(rec),
        requester=s["requester"], requester_role=s["role"], dry_run=dry_run)
    agent_pipe.print_result(result)

    # Sample data
    if result.status == "executed":
        conn = psycopg2.connect(config.DATABASE_URL)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute(f"SELECT * FROM {result.mart_name} LIMIT 5")
            rows = cur.fetchall()
            if rows:
                print(f"\n📊 SAMPLE from {result.mart_name}:")
                cols = list(rows[0].keys())[:6]
                print(f"   {' | '.join(f'{c:>20}' for c in cols)}")
                print(f"   {'─'*130}")
                for r in rows:
                    print(f"   {' | '.join(f'{str(r[c])[:20]:>20}' for c in cols)}")
        except Exception as e:
            print(f"   ⚠ {e}")
        finally:
            cur.close(); conn.close()
    return result

def show_governance_summary():
    conn = psycopg2.connect(config.DATABASE_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cls_icon = {"public":"🟢","internal":"🔵","confidential":"🟠","restricted":"🔴"}

    print(f"\n{'█'*70}")
    print(f"  GOVERNANCE: Audit Trail")
    print(f"{'█'*70}")
    cur.execute("""SELECT target_object, requested_by, requester_role, classification,
                          masking_applied, rls_applied, grants_applied, row_count, execution_time_ms
                   FROM governance.provisioning_audit WHERE status='executed'
                   ORDER BY requested_at DESC LIMIT 10""")
    for r in cur.fetchall():
        ic = cls_icon.get(r["classification"],"⚪")
        print(f"\n   {ic} {r['target_object']}")
        print(f"      By: {r['requested_by']} ({r['requester_role']})")
        print(f"      Mask: {'✓' if r['masking_applied'] else '✗'} | RLS: {'✓' if r['rls_applied'] else '✗'} | Grants: {r['grants_applied']}")

    print(f"\n{'█'*70}")
    print(f"  GOVERNANCE: Data Mart Registry")
    print(f"{'█'*70}")
    cur.execute("""SELECT mart_name, classification, has_pii_masking, allowed_roles, row_count
                   FROM governance.data_mart_registry ORDER BY created_at DESC""")
    for r in cur.fetchall():
        ic = cls_icon.get(r["classification"],"⚪")
        print(f"\n   📦 {r['mart_name']} {ic}")
        print(f"      {r['row_count']:,} rows | Roles: {r['allowed_roles']}")

    cur.close(); conn.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scenario", default="all")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Lab 04 — Metadata RAG + Governed Data Mart Provisioning           ║")
    print("║                                                                      ║")
    print("║  S3 (Parquet) ─→ pgvector (metadata) ─→ PostgreSQL (data mart)     ║")
    print("║    storage         RAG search              compute + governance      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    if not config.VOYAGE_API_KEY or not config.OPENAI_API_KEY:
        print("❌ Set VOYAGE_API_KEY and OPENAI_API_KEY"); sys.exit(1)

    scenarios = SCENARIOS
    if args.scenario != "all":
        sid = int(args.scenario)
        scenarios = [s for s in SCENARIOS if s["id"] == sid]

    results = []
    for s in scenarios:
        results.append(run_scenario(s, args.dry_run))

    if not args.dry_run:
        show_governance_summary()

    print(f"\n{'═'*70}")
    print("  SUMMARY")
    print(f"{'═'*70}")
    for s, r in zip(scenarios, results):
        ic = {"executed":"✅","dry_run":"🔍","error":"❌"}.get(r.status,"❓")
        print(f"  {ic} S{s['id']}: {s['title']}")
        print(f"     {r.mart_name} | {r.row_count:,} rows | {r.total_time_ms}ms | Grants: {r.grants}")

if __name__ == "__main__":
    main()
