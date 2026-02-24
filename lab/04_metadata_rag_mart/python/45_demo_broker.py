#!/usr/bin/env python3
"""
45_demo_broker.py — Demo: 3-Agent Broker Intelligence Workflow

Demonstrates the full broker agent pipeline:
  1. Agent 3 (Broker) scans financial intelligence → generates alerts
  2. Human reviews alerts and approves data mart requests
  3. Agent 1 (RAG) + Agent 2 (Pipeline) provision targeted data marts

Usage:
    python python/45_demo_broker.py                  # interactive
    python python/45_demo_broker.py --approve-all    # auto-approve all
    python python/45_demo_broker.py --approve 1      # approve alert #1
    python python/45_demo_broker.py --dry-run        # dry-run (no mart creation)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from importlib import import_module
import config
broker = import_module("35_agent_broker")


def main():
    parser = argparse.ArgumentParser(description="Broker Intelligence Demo")
    parser.add_argument("--approve-all", action="store_true",
                        help="Auto-approve all data mart requests")
    parser.add_argument("--approve", type=int, nargs="*", default=[],
                        help="Approve specific alert numbers (1-based)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry-run mode (show SQL but don't execute)")
    parser.add_argument("--pm", default="Senior PM",
                        help="Portfolio manager name")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Agent 3: Broker Intelligence — Senior Portfolio Manager            ║")
    print("║                                                                      ║")
    print("║  Scans: News → Filings → Analyst Recs → Earnings Reports           ║")
    print("║  Cross-refs with position metadata → Generates actionable alerts    ║")
    print("║  Human approves → Agent 1 (RAG) + Agent 2 (Pipeline) provision     ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    if not config.VOYAGE_API_KEY or not config.OPENAI_API_KEY:
        print("\n❌ Set VOYAGE_API_KEY and OPENAI_API_KEY before running.")
        sys.exit(1)

    # Phase 1: Intelligence Analysis
    print(f"\n{'█'*70}")
    print(f"  PHASE 1: Intelligence Scan & Analysis")
    print(f"{'█'*70}")

    brief = broker.run_analysis(portfolio_manager=args.pm)
    broker.print_brief(brief)

    # Phase 2: Human-in-the-Loop Approval
    mart_alerts = [a for a in brief.alerts if a.recommended_action == "request_mart"]

    if not mart_alerts:
        print("\n   No alerts require data mart provisioning.")
        print_summary(brief, [])
        return

    print(f"\n{'█'*70}")
    print(f"  PHASE 2: Human-in-the-Loop — Data Mart Approval")
    print(f"{'█'*70}")

    provisioned = []

    for i, alert in enumerate(mart_alerts, 1):
        severity_icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
        sev = severity_icon.get(alert.severity, "⚪")

        print(f"\n   {sev} Alert #{i}: {alert.signal.headline[:70]}")
        print(f"      Suggested mart: \"{alert.suggested_mart_question[:90]}\"")

        # Determine approval
        approved = False
        if args.approve_all:
            approved = True
            print(f"      → Auto-approved (--approve-all)")
        elif i in args.approve:
            approved = True
            print(f"      → Approved (--approve {i})")
        else:
            if not args.approve_all and not args.approve:
                # Interactive mode
                try:
                    response = input(f"      Approve data mart? [y/N]: ").strip().lower()
                    approved = response in ("y", "yes")
                except (EOFError, KeyboardInterrupt):
                    print("\n      → Skipped")
                    approved = False

        if approved:
            print(f"\n   ⚡ Provisioning data mart for Alert #{i}...")
            print(f"   {'─'*60}")
            result = broker.request_data_mart(
                alert, approved=True,
                requester=f"broker.{args.pm.lower().replace(' ', '_')}",
                dry_run=args.dry_run,
            )
            if result:
                provisioned.append({"alert": alert.alert_id, "result": result})
                print(f"   ✅ Mart provisioned: {result.get('mart_name', 'N/A')}")
            else:
                print(f"   ⚠ Provisioning returned no result")
        else:
            print(f"      → Skipped (not approved)")

    # Phase 3: Summary
    print_summary(brief, provisioned)


def print_summary(brief: broker.BrokerBrief, provisioned: list):
    """Print final summary."""
    print(f"\n{'═'*70}")
    print(f"  SUMMARY — Broker Intelligence Workflow")
    print(f"{'═'*70}")
    print(f"   Portfolio Manager: {brief.portfolio_manager}")
    print(f"   Generated at:     {brief.generated_at}")
    print(f"\n   📊 Signals & Alerts:")
    print(f"      Total alerts:    {len(brief.alerts)}")
    print(f"      🔴 Critical:     {brief.critical_count}")
    print(f"      🟠 Warning:      {brief.warning_count}")
    print(f"      🔵 Info:         {brief.info_count}")
    print(f"\n   ⚡ Data Marts Provisioned: {len(provisioned)}")
    for p in provisioned:
        result = p["result"]
        status_icon = {"executed": "✅", "dry_run": "🔍", "error": "❌"}.get(result.get("status", ""), "❓")
        print(f"      {status_icon} {result.get('mart_name', 'N/A')} ({result.get('row_count', 0):,} rows)")
    if not provisioned:
        print(f"      (none)")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
