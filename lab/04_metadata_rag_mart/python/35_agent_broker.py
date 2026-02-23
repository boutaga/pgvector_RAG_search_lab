#!/usr/bin/env python3
"""
35_agent_broker.py — Agent 3: Broker (Senior Portfolio Manager)

Architecture:
    1. Scans financial intelligence datasets for signals (news, filings, reports, analyst recs)
    2. Cross-references signals with client positions (via metadata, NOT raw data)
    3. Identifies concerns: overexposure to downgraded/negative sectors, concentration risk
    4. Presents findings with severity + recommended action
    5. On approval, calls Agent 1 (RAG) → Agent 2 (Pipeline) for targeted data mart
    6. Also checks existing KPI patterns for matching analytics

Key principle: Human-in-the-loop — Broker Agent presents analysis, human confirms
before any data mart provisioning.

Usage:
    from importlib import import_module
    broker = import_module("35_agent_broker")
    brief = broker.run_analysis(conn, s3)
"""

import io
import json
import time
import psycopg2
import psycopg2.extras
import pandas as pd
import boto3
from botocore.client import Config
from openai import OpenAI
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import config

llm = OpenAI(api_key=config.OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    signal_id: str
    source_type: str       # news, filing, recommendation, report
    source_table: str
    headline: str
    sentiment: str
    impact_level: str
    related_sectors: List[str]
    related_isins: List[str]
    signal_date: str


@dataclass
class BrokerAlert:
    alert_id: str
    severity: str          # info, warning, critical
    signal: Signal
    affected_positions: List[Dict]   # from catalog metadata
    exposure_summary: str
    recommended_action: str          # "monitor", "review_positions", "request_mart"
    suggested_mart_question: str     # NL question for Agent 1


@dataclass
class BrokerBrief:
    generated_at: str
    portfolio_manager: str
    alerts: List[BrokerAlert]
    summary: str
    critical_count: int
    warning_count: int
    info_count: int


# ---------------------------------------------------------------------------
# S3 CLIENT
# ---------------------------------------------------------------------------

def get_s3():
    return boto3.client("s3", endpoint_url=config.S3_ENDPOINT,
                        aws_access_key_id=config.S3_ACCESS_KEY,
                        aws_secret_access_key=config.S3_SECRET_KEY,
                        config=Config(signature_version="s3v4"),
                        region_name="us-east-1")


def read_parquet_df(s3, table_name: str) -> pd.DataFrame:
    """Read a Parquet file from S3 into DataFrame."""
    key = f"{table_name}.parquet"
    obj = s3.get_object(Bucket=config.S3_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


# ---------------------------------------------------------------------------
# SIGNAL SCANNING
# ---------------------------------------------------------------------------

def scan_signals(s3) -> List[Signal]:
    """Scan financial intelligence datasets for actionable signals.

    Reads from S3 Parquet files directly (like Agent 2 reads from S3).
    Filters for high-impact, negative/critical items that need attention.
    """
    signals = []
    sid = 0

    # 1. Financial News — high/critical impact with negative sentiment
    try:
        news_df = read_parquet_df(s3, "financial_news")
        critical_news = news_df[
            (news_df["impact_level"].isin(["high", "critical"])) &
            (news_df["sentiment"].isin(["negative"]))
        ]
        for _, row in critical_news.iterrows():
            sid += 1
            related_isins = json.loads(row["related_isins"]) if isinstance(row["related_isins"], str) else row.get("related_isins", [])
            related_sectors = json.loads(row["related_sectors"]) if isinstance(row["related_sectors"], str) else row.get("related_sectors", [])
            signals.append(Signal(
                signal_id=f"NEWS-{sid:03d}",
                source_type="news",
                source_table="financial_news",
                headline=row["headline"],
                sentiment=row["sentiment"],
                impact_level=row["impact_level"],
                related_sectors=related_sectors,
                related_isins=related_isins,
                signal_date=str(row["published_at"])[:10],
            ))
    except Exception as e:
        print(f"   ⚠ Could not scan financial_news: {e}")

    # 2. Analyst Recommendations — downgrades (recommendation worse than prev)
    downgrade_map = {"strong_buy": 5, "buy": 4, "hold": 3, "sell": 2, "strong_sell": 1}
    try:
        recs_df = read_parquet_df(s3, "analyst_recommendations")
        for _, row in recs_df.iterrows():
            curr_score = downgrade_map.get(row["recommendation"], 3)
            prev_score = downgrade_map.get(row["prev_recommendation"], 3)
            if curr_score < prev_score:  # downgrade
                sid += 1
                signals.append(Signal(
                    signal_id=f"REC-{sid:03d}",
                    source_type="recommendation",
                    source_table="analyst_recommendations",
                    headline=f"{row['analyst_firm']} downgrades {row['instrument_name']} from {row['prev_recommendation']} to {row['recommendation']}",
                    sentiment="negative",
                    impact_level="high" if curr_score <= 2 else "medium",
                    related_sectors=[row["sector"]] if row.get("sector") else [],
                    related_isins=[row["instrument_isin"]] if row.get("instrument_isin") else [],
                    signal_date=str(row["published_date"]),
                ))
    except Exception as e:
        print(f"   ⚠ Could not scan analyst_recommendations: {e}")

    # 3. Financial Reports — revenue/EPS misses or lowered guidance
    try:
        reports_df = read_parquet_df(s3, "financial_reports")
        negative_reports = reports_df[
            (reports_df["revenue_surprise_pct"] < -1.5) |
            (reports_df["guidance"].isin(["lowered", "withdrawn"]))
        ]
        for _, row in negative_reports.iterrows():
            sid += 1
            signals.append(Signal(
                signal_id=f"RPT-{sid:03d}",
                source_type="report",
                source_table="financial_reports",
                headline=f"{row['company']} {row['period']}: revenue surprise {row['revenue_surprise_pct']:+.1f}%, guidance {row['guidance']}",
                sentiment="negative",
                impact_level="critical" if row["revenue_surprise_pct"] < -2.0 else "high",
                related_sectors=[row["sector"]] if row.get("sector") else [],
                related_isins=[row["company_isin"]] if row.get("company_isin") else [],
                signal_date=str(row["report_date"]),
            ))
    except Exception as e:
        print(f"   ⚠ Could not scan financial_reports: {e}")

    # 4. Public Filings — material or negative impact
    try:
        filings_df = read_parquet_df(s3, "public_filings")
        material_filings = filings_df[
            filings_df["financial_impact"].isin(["material", "negative"])
        ]
        for _, row in material_filings.iterrows():
            sid += 1
            signals.append(Signal(
                signal_id=f"FIL-{sid:03d}",
                source_type="filing",
                source_table="public_filings",
                headline=row["title"],
                sentiment="negative",
                impact_level="critical" if row["financial_impact"] == "material" else "high",
                related_sectors=[row["sector"]] if row.get("sector") else [],
                related_isins=[row["issuer_isin"]] if row.get("issuer_isin") else [],
                signal_date=str(row["filing_date"]),
            ))
    except Exception as e:
        print(f"   ⚠ Could not scan public_filings: {e}")

    return signals


# ---------------------------------------------------------------------------
# CROSS-REFERENCE WITH POSITIONS (via catalog metadata)
# ---------------------------------------------------------------------------

def cross_reference_positions(conn, signals: List[Signal]) -> List[BrokerAlert]:
    """Cross-reference signals with position metadata from the catalog.

    Uses catalog metadata (table_metadata, column_metadata, relationship_metadata)
    to understand which positions/clients may be affected.
    Does NOT read raw position data — only metadata about the data lake.
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Gather catalog context: which sectors/ISINs exist in positions
    cur.execute("""
        SELECT cm.table_name, cm.column_name, cm.sample_values, cm.n_distinct,
               cm.detail_bi, cm.detail_agent
        FROM catalog.column_metadata cm
        WHERE cm.table_name IN ('positions', 'instruments', 'clients', 'accounts')
        AND cm.column_name IN ('sector', 'isin', 'instrument_id', 'market_value',
                               'weight_pct', 'client_id', 'segment', 'risk_profile')
    """)
    catalog_context = cur.fetchall()

    # Get table-level stats
    cur.execute("""
        SELECT table_name, row_count, detail_bi
        FROM catalog.table_metadata
        WHERE table_name IN ('positions', 'instruments', 'clients', 'accounts')
    """)
    table_stats = {r["table_name"]: r for r in cur.fetchall()}

    cur.close()

    alerts = []
    aid = 0
    position_count = table_stats.get("positions", {}).get("row_count", 0)
    instrument_count = table_stats.get("instruments", {}).get("row_count", 0)

    for signal in signals:
        aid += 1
        # Determine severity based on signal characteristics
        if signal.impact_level == "critical":
            severity = "critical"
        elif signal.impact_level == "high" and signal.source_type in ("news", "recommendation"):
            severity = "warning"
        elif signal.impact_level == "high":
            severity = "warning"
        else:
            severity = "info"

        # Build exposure summary from metadata
        sector_str = ", ".join(signal.related_sectors) if signal.related_sectors else "multiple sectors"
        isin_str = ", ".join(signal.related_isins[:3]) if signal.related_isins else "unknown instruments"

        exposure_summary = (
            f"Signal affects {sector_str}. "
            f"Positions table has ~{position_count} position records across "
            f"~{instrument_count} instruments. "
            f"ISINs involved: {isin_str}. "
            f"Cross-reference with client portfolios recommended."
        )

        # Determine recommended action
        if severity == "critical":
            action = "request_mart"
        elif severity == "warning":
            action = "review_positions"
        else:
            action = "monitor"

        # Build suggested mart question
        if signal.source_type == "news" and "Nestlé" in signal.headline:
            question = "Show me all client positions in Consumer Staples sector with exposure to Nestlé, including market value, weight percentage, and client segment breakdown"
        elif signal.source_type == "filing" and "FINMA" in signal.headline:
            question = "Show me client exposure to Financials sector, especially UBS and Swiss banks, with concentration risk analysis and VaR impact"
        elif signal.source_type == "recommendation" and signal.related_sectors and "Information Technology" in signal.related_sectors:
            question = "Show me Technology sector positions with recent analyst downgrades, including client exposure and unrealized P&L impact"
        elif signal.related_sectors:
            question = f"Show me all client positions in {signal.related_sectors[0]} sector with market value, exposure weight, and risk metrics"
        else:
            question = f"Show me positions affected by {signal.headline[:80]}, with client exposure and sector breakdown"

        affected_positions = [
            {"context": "catalog_metadata", "tables": ["positions", "instruments"],
             "sectors": signal.related_sectors, "isins": signal.related_isins}
        ]

        alerts.append(BrokerAlert(
            alert_id=f"ALERT-{aid:03d}",
            severity=severity,
            signal=signal,
            affected_positions=affected_positions,
            exposure_summary=exposure_summary,
            recommended_action=action,
            suggested_mart_question=question,
        ))

    # Sort by severity (critical first)
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: severity_order.get(a.severity, 9))

    return alerts


# ---------------------------------------------------------------------------
# BROKER BRIEF GENERATION
# ---------------------------------------------------------------------------

def generate_broker_brief(alerts: List[BrokerAlert],
                          portfolio_manager: str = "Senior PM") -> BrokerBrief:
    """Generate a structured broker brief from alerts.

    Uses LLM to create a concise executive summary of the intelligence findings.
    """
    critical = [a for a in alerts if a.severity == "critical"]
    warnings = [a for a in alerts if a.severity == "warning"]
    info = [a for a in alerts if a.severity == "info"]

    # Build context for LLM
    alert_summaries = []
    for a in alerts[:15]:  # cap for token budget
        alert_summaries.append({
            "id": a.alert_id,
            "severity": a.severity,
            "source": a.signal.source_type,
            "headline": a.signal.headline[:120],
            "sectors": a.signal.related_sectors,
            "action": a.recommended_action,
        })

    prompt = f"""You are a Senior Portfolio Manager at a Swiss private bank.
Summarize these {len(alerts)} intelligence alerts into a 3-5 sentence executive brief.
Focus on: (1) most urgent items, (2) sector exposure concerns, (3) recommended actions.

Alerts:
{json.dumps(alert_summaries, indent=2)}

Write a concise, actionable brief in professional financial language."""

    try:
        resp = llm.chat.completions.create(
            model=config.CHAT_MODEL_FAST,
            messages=[
                {"role": "system", "content": "You are a Senior Portfolio Manager at a Swiss private bank. Write concise, professional intelligence briefs."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400, temperature=0.2
        )
        summary = resp.choices[0].message.content
    except Exception:
        summary = (
            f"Intelligence scan detected {len(critical)} critical and {len(warnings)} warning signals. "
            f"Key concerns: {', '.join(set(s for a in critical for s in a.signal.related_sectors))}. "
            f"Immediate review recommended for {len([a for a in alerts if a.recommended_action == 'request_mart'])} positions."
        )

    return BrokerBrief(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        portfolio_manager=portfolio_manager,
        alerts=alerts,
        summary=summary,
        critical_count=len(critical),
        warning_count=len(warnings),
        info_count=len(info),
    )


# ---------------------------------------------------------------------------
# MART REQUEST (delegates to Agent 1 + Agent 2)
# ---------------------------------------------------------------------------

def request_data_mart(alert: BrokerAlert, approved: bool = False,
                      requester: str = "broker.agent",
                      dry_run: bool = False) -> Optional[Dict]:
    """Request a targeted data mart via Agent 1 → Agent 2.

    Human-in-the-loop gate: only proceeds if approved=True.
    """
    if not approved:
        return None

    from importlib import import_module
    agent_rag = import_module("20_agent_rag_search")
    agent_pipe = import_module("30_agent_pipeline")
    from dataclasses import asdict

    # Agent 1: RAG search
    rec = agent_rag.search(
        alert.suggested_mart_question,
        requester=requester,
        role="portfolio_manager"
    )
    agent_rag.print_recommendation(rec)

    # Agent 2: Pipeline
    rec_dict = asdict(rec)
    result = agent_pipe.provision(
        question=alert.suggested_mart_question,
        recommendation=rec_dict,
        requester=requester,
        requester_role="portfolio_manager",
        dry_run=dry_run,
    )
    agent_pipe.print_result(result)

    return asdict(result)


# ---------------------------------------------------------------------------
# MAIN ANALYSIS ENTRY POINT
# ---------------------------------------------------------------------------

def run_analysis(portfolio_manager: str = "Senior PM") -> BrokerBrief:
    """Run the full broker intelligence analysis.

    Returns a BrokerBrief ready for human review and approval.
    """
    s3 = get_s3()
    conn = psycopg2.connect(config.DATABASE_URL)

    try:
        # Step 1: Scan signals
        print("\n   Scanning financial intelligence datasets...")
        signals = scan_signals(s3)
        print(f"   Found {len(signals)} signals")

        # Step 2: Cross-reference with positions
        print("   Cross-referencing with position metadata...")
        alerts = cross_reference_positions(conn, signals)
        print(f"   Generated {len(alerts)} alerts")

        # Step 3: Generate brief
        print("   Generating broker brief...")
        brief = generate_broker_brief(alerts, portfolio_manager)

        return brief
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------------------

def print_brief(brief: BrokerBrief):
    """Print formatted broker brief."""
    severity_icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
    source_icon = {"news": "📰", "recommendation": "📊", "report": "📈", "filing": "📋"}
    action_icon = {"request_mart": "⚡", "review_positions": "👁️", "monitor": "📌"}

    print(f"\n{'='*70}")
    print(f"🏦 AGENT 3: Broker Intelligence — {brief.portfolio_manager}")
    print(f"   Generated: {brief.generated_at}")
    print(f"{'='*70}")

    print(f"\n📊 SIGNAL SUMMARY:")
    print(f"   🔴 Critical: {brief.critical_count}")
    print(f"   🟠 Warning:  {brief.warning_count}")
    print(f"   🔵 Info:     {brief.info_count}")
    print(f"   Total alerts: {len(brief.alerts)}")

    print(f"\n📝 EXECUTIVE BRIEF:")
    print(f"   {brief.summary}")

    print(f"\n{'─'*70}")
    print(f"   ALERTS (top 15):")
    print(f"{'─'*70}")

    for i, alert in enumerate(brief.alerts[:15]):
        sev = severity_icon.get(alert.severity, "⚪")
        src = source_icon.get(alert.signal.source_type, "📄")
        act = action_icon.get(alert.recommended_action, "❓")

        print(f"\n   {sev} [{alert.alert_id}] {alert.signal.headline[:80]}")
        print(f"      {src} Source: {alert.signal.source_type} | Date: {alert.signal.signal_date}")
        print(f"      Sectors: {', '.join(alert.signal.related_sectors) if alert.signal.related_sectors else 'N/A'}")
        print(f"      {act} Action: {alert.recommended_action}")
        if alert.recommended_action == "request_mart":
            print(f"      💡 Suggested mart: \"{alert.suggested_mart_question[:90]}...\"")

    # Count actionable
    mart_requests = [a for a in brief.alerts if a.recommended_action == "request_mart"]
    if mart_requests:
        print(f"\n{'─'*70}")
        print(f"   ⚡ {len(mart_requests)} alerts recommend data mart provisioning")
        print(f"   Awaiting human approval before proceeding...")

    print(f"{'='*70}")
