#!/usr/bin/env python3
"""
10_scan_and_embed.py — Scan Parquet schemas from S3 + Embed into pgvector

Reads Parquet file schemas from MinIO, builds semantic metadata text
enriched with BI team annotations and LLM-generated insights,
generates Voyage finance-2 embeddings, and stores in catalog.*.

The pgvector database stores ZERO raw data — only metadata about
the structure and meaning of the data lake files.

Usage:
    python python/10_scan_and_embed.py
"""

import io
import json
import psycopg2
import psycopg2.extras
import pyarrow.parquet as pq
import boto3
import voyageai
from botocore.client import Config
from typing import List, Dict
import config

vo = voyageai.Client(api_key=config.VOYAGE_API_KEY)

# ---------------------------------------------------------------------------
# S3 / MINIO
# ---------------------------------------------------------------------------

def get_s3():
    return boto3.client("s3", endpoint_url=config.S3_ENDPOINT,
                        aws_access_key_id=config.S3_ACCESS_KEY,
                        aws_secret_access_key=config.S3_SECRET_KEY,
                        config=Config(signature_version="s3v4"),
                        region_name="us-east-1")

def read_parquet_schema(s3, bucket: str, key: str) -> Dict:
    """Read Parquet file metadata without downloading the full file."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    buf = io.BytesIO(obj["Body"].read())
    pf = pq.ParquetFile(buf)
    schema = pf.schema_arrow
    meta = pf.metadata
    return {
        "schema": schema,
        "num_rows": meta.num_rows,
        "num_columns": meta.num_columns,
        "file_size": obj["ContentLength"],
        "row_groups": meta.num_row_groups,
    }

def read_parquet_sample(s3, bucket: str, key: str, n: int = 100):
    """Read a sample of rows for statistics."""
    import pandas as pd
    obj = s3.get_object(Bucket=bucket, Key=key)
    buf = io.BytesIO(obj["Body"].read())
    df = pd.read_parquet(buf)
    return df.head(n), df

# ---------------------------------------------------------------------------
# BI TEAM ANNOTATIONS (manual, domain-specific context)
# ---------------------------------------------------------------------------
# In production these come from a BI knowledge base or manual input.
# For the demo, we pre-populate realistic annotations.

TABLE_DETAIL_BI = {
    "exchanges": "Reference table for ISO 10383 MIC codes. Static data, refreshed quarterly when new MICs are assigned. Key for venue analysis and regulatory reporting (RTS 25).",
    "instruments": "Master security reference with ISINs (ISO 6166), SEDOL, Bloomberg tickers. Covers equities, fixed income, ETFs, structured products. PII flag on ISIN for client-linked derivatives. Refresh: nightly from Bloomberg BVAL feed.",
    "counterparties": "Broker, custodian, and clearing house directory. LEI codes follow ISO 17442. Critical for counterparty risk aggregation and EMIR reporting. Refresh: monthly from GLEIF.",
    "clients": "Client master with KYC data. Contains PII (legal_name, country fields). PEP flags from WorldCheck. Risk ratings assigned by Risk Committee quarterly. CRITICAL: country_domicile + is_pep drive sanctions screening. FINMA audit requirement: 10yr retention.",
    "accounts": "Account hierarchy linked to clients. IBAN-like numbers. Types: trading (active orders), custody (hold only), cash (settlement). Subtype drives fee schedule: discretionary > advisory > execution_only.",
    "orders": "FIX-protocol aligned order book. cl_ord_id is the client order UUID. Status follows FIX lifecycle: NEW→PARTIALLY_FILLED→FILLED|CANCELLED|REJECTED. compliance_check field is pre-trade check result. Settlement is T+2 equities, T+1 FX.",
    "executions": "Fill-level execution data for TCA (Transaction Cost Analysis). Multiple fills per order for large orders (TWAP/VWAP). Commission in basis points varies by broker and venue. MiFID II best execution reporting requirement.",
    "positions": "End-of-day position snapshot. Grain: (account_id, instrument_id, valuation_date). P&L is mark-to-market. weight_pct is position weight in portfolio. Refresh: nightly batch at 06:00 CET after market close.",
    "cash_balances": "Cash position per account/currency. 'available' = balance minus pending settlements. 'reserved' = locked for margin or pending orders. Refresh: real-time via settlement feeds.",
    "market_prices": "OHLCV market data. Sources: Bloomberg, Reuters, exchange direct. adj_close reflects corporate actions (splits, dividends). Used by risk engine for VaR calculation. 30 trading days history.",
    "risk_limits": "Risk limit framework per client. Types: gross/net exposure, single-name concentration, sector, leverage. utilization_pct triggers breach_action. Approved by Risk Committee. FINMA Circular 2008/21 compliance.",
    "risk_metrics": "Daily risk metrics computed by risk engine. VaR (parametric, 1-day horizon, 95th/99th percentile), CVaR (Expected Shortfall), Sharpe, volatility, drawdown, beta. Used for FINMA regulatory reporting and client quarterly reviews.",
    "compliance_checks": "Pre-trade compliance check results. check_type: suitability (MiFID II), concentration (internal limits), sanctions (OFAC/EU/CH list). Override requires compliance_officer approval with reason. RESTRICTED classification — regulatory audit trail.",
    "aml_alerts": "Anti-Money Laundering alerts. Generated by transaction monitoring system (Actimize/NICE). severity 'critical' = immediate escalation to MLRO. status 'closed_sar_filed' = SAR sent to MROS (Swiss FIU). RESTRICTED — only compliance_officer access. FINMA AMLA compliance.",
}

COLUMN_DETAIL_BI = {
    ("clients", "legal_name"): "PII. Full legal name as per KYC documents. Must be masked for non-compliance roles.",
    ("clients", "short_name"): "PII. Abbreviated name for internal reference. Must be masked.",
    ("clients", "is_pep"): "Politically Exposed Person flag. Source: WorldCheck + manual review. PEP clients require Enhanced Due Diligence (EDD) and senior management approval for onboarding.",
    ("clients", "risk_rating"): "Assigned by Risk Committee quarterly. Scale: low, medium, high, critical. 'critical' = immediate review and enhanced monitoring. Drives risk limit allocation.",
    ("clients", "mifid_category"): "MiFID II client categorization. Determines level of investor protection. retail_client has highest protection, eligible_counterparty has lowest.",
    ("accounts", "account_number"): "PII. IBAN-like format. Must be masked for non-compliance roles.",
    ("accounts", "account_subtype"): "Fee and service model: discretionary (bank manages), advisory (bank advises, client decides), execution_only (client decides, lowest fees).",
    ("orders", "compliance_check"): "Pre-trade check result. 'flagged' = soft warning, order proceeds with note. 'rejected' = hard block, requires compliance override.",
    ("orders", "cl_ord_id"): "Client Order ID — unique UUID per order. Links to FIX protocol ClOrdID(11) field.",
    ("executions", "venue"): "Execution venue MIC code. 'OTC' = over-the-counter. MiFID II RTS 28 requires venue reporting.",
    ("risk_metrics", "var_1d_95"): "Value at Risk, 1-day horizon, 95th percentile. Parametric VaR using variance-covariance method. Currency: same as client base currency.",
    ("risk_metrics", "cvar_1d_95"): "Conditional VaR (Expected Shortfall) at 95%. Average loss beyond VaR threshold. More conservative than VaR for tail risk.",
    ("aml_alerts", "severity"): "Alert priority. 'critical' triggers immediate escalation to MLRO and potential account freeze. SLA: critical=2h, high=24h, medium=5d, low=30d.",
    ("instruments", "isin"): "International Securities Identification Number (ISO 6166). 12 characters: 2-letter country + 9 alphanumeric + 1 check digit.",
    ("positions", "unrealized_pnl"): "Mark-to-market unrealized P&L = (market_price - avg_cost) × quantity. Updated nightly. Used for client reporting and performance attribution.",
    ("risk_limits", "utilization_pct"): "Current usage as percentage of limit. Above 80% = alert. Above 95% = block new orders. Above 100% = force position reduction.",
}

# Classification rules
RESTRICTED_TABLES = {"aml_alerts", "compliance_checks"}
CONFIDENTIAL_TABLES = {"risk_metrics", "risk_limits", "orders", "executions"}
INTERNAL_TABLES = {"positions", "cash_balances", "market_prices", "clients", "accounts"}
PUBLIC_TABLES = {"exchanges", "instruments", "counterparties"}

PII_COLUMNS = {
    ("clients", "legal_name"): "partial_mask",
    ("clients", "short_name"): "partial_mask",
    ("accounts", "account_number"): "partial_mask",
}

def classify_table(name: str) -> str:
    if name in RESTRICTED_TABLES: return "restricted"
    if name in CONFIDENTIAL_TABLES: return "confidential"
    if name in INTERNAL_TABLES: return "internal"
    return "public"

# ---------------------------------------------------------------------------
# KNOWN RELATIONSHIPS (inferred from domain knowledge, not FK constraints)
# ---------------------------------------------------------------------------

RELATIONSHIPS = [
    ("orders", "account_id", "accounts", "account_id", "one_to_many"),
    ("orders", "instrument_id", "instruments", "instrument_id", "one_to_many"),
    ("orders", "broker_id", "counterparties", "counterparty_id", "one_to_many"),
    ("executions", "order_id", "orders", "order_id", "one_to_many"),
    ("accounts", "client_id", "clients", "client_id", "one_to_many"),
    ("accounts", "custodian_id", "counterparties", "counterparty_id", "one_to_many"),
    ("positions", "account_id", "accounts", "account_id", "one_to_many"),
    ("positions", "instrument_id", "instruments", "instrument_id", "one_to_many"),
    ("cash_balances", "account_id", "accounts", "account_id", "one_to_many"),
    ("market_prices", "instrument_id", "instruments", "instrument_id", "one_to_many"),
    ("risk_limits", "client_id", "clients", "client_id", "one_to_many"),
    ("risk_metrics", "client_id", "clients", "client_id", "one_to_many"),
    ("compliance_checks", "order_id", "orders", "order_id", "one_to_many"),
    ("aml_alerts", "client_id", "clients", "client_id", "one_to_many"),
    ("instruments", "exchange_id", "exchanges", "exchange_id", "one_to_many"),
]

# ---------------------------------------------------------------------------
# KPI PATTERNS
# ---------------------------------------------------------------------------

KPI_PATTERNS = [
    {"kpi_name":"Portfolio Exposure by Sector","kpi_description":"Total market value aggregated by GICS sector. Used for concentration monitoring and diversification analysis.","domain":"portfolio","required_tables":["positions","instruments"],"required_columns":["positions.market_value","positions.weight_pct","instruments.sector","instruments.asset_class"],"sql_template":"SELECT i.sector, SUM(p.market_value) AS total_exposure, AVG(p.weight_pct) AS avg_weight FROM lake.positions p JOIN lake.instruments i ON p.instrument_id = i.instrument_id GROUP BY i.sector ORDER BY total_exposure DESC","classification":"internal"},
    {"kpi_name":"Client AUM by Segment","kpi_description":"Assets Under Management by client segment (retail/HNWI/UHNWI/institutional). Core wealth management KPI.","domain":"portfolio","required_tables":["positions","accounts","clients"],"required_columns":["positions.market_value","clients.segment","clients.client_code"],"sql_template":"SELECT c.segment, COUNT(DISTINCT c.client_id) AS num_clients, SUM(p.market_value) AS total_aum FROM lake.positions p JOIN lake.accounts a ON p.account_id = a.account_id JOIN lake.clients c ON a.client_id = c.client_id GROUP BY c.segment","classification":"confidential"},
    {"kpi_name":"Execution Quality (TCA)","kpi_description":"Transaction Cost Analysis — fill price vs arrival price, slippage, commission rates. MiFID II best execution reporting.","domain":"trading","required_tables":["orders","executions","instruments"],"required_columns":["orders.order_type","orders.side","executions.fill_price","executions.commission","executions.venue"],"sql_template":"SELECT o.order_type, o.side, COUNT(*) AS fills, AVG(e.fill_price) AS avg_fill, SUM(e.commission) AS total_comm FROM lake.executions e JOIN lake.orders o ON e.order_id = o.order_id GROUP BY o.order_type, o.side","classification":"confidential"},
    {"kpi_name":"VaR Dashboard by Client","kpi_description":"Value at Risk (1-day, 95%/99%) and CVaR per client. FINMA regulatory reporting.","domain":"risk","required_tables":["risk_metrics","clients"],"required_columns":["risk_metrics.var_1d_95","risk_metrics.var_1d_99","risk_metrics.cvar_1d_95","risk_metrics.gross_exposure","clients.segment","clients.risk_rating"],"sql_template":"SELECT c.client_code, c.risk_rating, rm.metric_date, rm.var_1d_95, rm.var_1d_99, rm.cvar_1d_95, rm.gross_exposure FROM lake.risk_metrics rm JOIN lake.clients c ON rm.client_id = c.client_id ORDER BY rm.var_1d_95 DESC","classification":"confidential"},
    {"kpi_name":"Risk Limit Utilization","kpi_description":"Current utilization of risk limits. Flags breaches and near-breaches. Pre-trade risk controls.","domain":"risk","required_tables":["risk_limits","clients"],"required_columns":["risk_limits.limit_type","risk_limits.utilization_pct","risk_limits.breach_action"],"sql_template":"SELECT c.client_code, rl.limit_type, rl.limit_value, rl.current_usage, rl.utilization_pct, rl.breach_action FROM lake.risk_limits rl JOIN lake.clients c ON rl.client_id = c.client_id WHERE rl.utilization_pct > 70 ORDER BY rl.utilization_pct DESC","classification":"confidential"},
    {"kpi_name":"AML Investigation Pipeline","kpi_description":"AML alerts by type, severity, investigation status. SAR filing metrics for FINMA/MROS compliance.","domain":"compliance","required_tables":["aml_alerts","clients"],"required_columns":["aml_alerts.alert_type","aml_alerts.severity","aml_alerts.status","clients.is_pep","clients.country_domicile"],"sql_template":"SELECT a.alert_type, a.severity, a.status, COUNT(*) AS count, c.country_domicile FROM lake.aml_alerts a JOIN lake.clients c ON a.client_id = c.client_id GROUP BY a.alert_type, a.severity, a.status, c.country_domicile","classification":"restricted"},
    {"kpi_name":"Trading Volume by Desk","kpi_description":"Daily order volume and fill rates by desk and asset class. Capacity planning.","domain":"trading","required_tables":["orders","instruments"],"required_columns":["orders.trading_desk","orders.order_date","orders.quantity","orders.filled_qty","instruments.asset_class"],"sql_template":"SELECT o.trading_desk, i.asset_class, o.order_date, COUNT(*) AS orders, SUM(o.filled_qty) AS filled FROM lake.orders o JOIN lake.instruments i ON o.instrument_id = i.instrument_id GROUP BY o.trading_desk, i.asset_class, o.order_date","classification":"internal"},
    {"kpi_name":"Portfolio P&L Attribution","kpi_description":"Unrealized/realized P&L per client and instrument. Performance reporting and fee calculation.","domain":"portfolio","required_tables":["positions","accounts","clients","instruments"],"required_columns":["positions.unrealized_pnl","positions.realized_pnl","positions.market_value","instruments.instrument_name","clients.client_code"],"sql_template":"SELECT c.client_code, i.instrument_name, p.market_value, p.cost_basis, p.unrealized_pnl, p.realized_pnl FROM lake.positions p JOIN lake.accounts a ON p.account_id = a.account_id JOIN lake.clients c ON a.client_id = c.client_id JOIN lake.instruments i ON p.instrument_id = i.instrument_id","classification":"confidential"},
    {"kpi_name":"Compliance Check Summary","kpi_description":"Pre-trade check pass/fail/override rates by type. Suitability, sanctions screening effectiveness.","domain":"compliance","required_tables":["compliance_checks","orders"],"required_columns":["compliance_checks.check_type","compliance_checks.check_result","orders.order_date"],"sql_template":"SELECT cc.check_type, cc.check_result, COUNT(*) AS count FROM lake.compliance_checks cc GROUP BY cc.check_type, cc.check_result","classification":"restricted"},
    {"kpi_name":"Cash & Liquidity Position","kpi_description":"Cash balances by account/currency with available-for-trading. Liquidity management.","domain":"operations","required_tables":["cash_balances","accounts","clients"],"required_columns":["cash_balances.balance","cash_balances.available","cash_balances.currency","accounts.account_type"],"sql_template":"SELECT a.account_type, cb.currency, SUM(cb.balance) AS total, SUM(cb.available) AS available FROM lake.cash_balances cb JOIN lake.accounts a ON cb.account_id = a.account_id GROUP BY a.account_type, cb.currency","classification":"internal"},
]

# ---------------------------------------------------------------------------
# BUILD METADATA TEXT (for embedding)
# ---------------------------------------------------------------------------

def build_table_text(table_name: str, info: Dict) -> str:
    parts = [
        f"Table: {table_name}",
        f"Columns: {info['num_columns']}",
        f"Rows: ~{info['num_rows']:,}",
        f"Format: Parquet on S3",
    ]
    if table_name in TABLE_DETAIL_BI:
        parts.append(f"Business context: {TABLE_DETAIL_BI[table_name]}")
    return " | ".join(parts)

def build_column_text(table_name: str, col_name: str, arrow_type: str, stats: Dict) -> str:
    parts = [
        f"Column: {table_name}.{col_name}",
        f"Type: {arrow_type}",
    ]
    bi_key = (table_name, col_name)
    if bi_key in COLUMN_DETAIL_BI:
        parts.append(f"Business context: {COLUMN_DETAIL_BI[bi_key]}")
    if stats.get("n_distinct"):
        parts.append(f"Distinct: {stats['n_distinct']}")
    if stats.get("sample_values"):
        parts.append(f"Samples: {', '.join(str(v) for v in stats['sample_values'][:5])}")
    return " | ".join(parts)

def build_relationship_text(r) -> str:
    return f"Relationship: {r[0]}.{r[1]} → {r[2]}.{r[3]} ({r[4]})"

# ---------------------------------------------------------------------------
# EMBEDDING
# ---------------------------------------------------------------------------

def embed_batch(texts: List[str], input_type: str = "document") -> List[List[float]]:
    result = vo.embed(texts, model=config.EMBEDDING_MODEL, input_type=input_type)
    return result.embeddings

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run():
    import pandas as pd
    s3 = get_s3()
    conn = psycopg2.connect(config.DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    bucket = config.S3_BUCKET
    resp = s3.list_objects_v2(Bucket=bucket)
    parquet_files = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".parquet")]

    print("="*60)
    print("  Scanning Parquet schemas from S3 → pgvector catalog")
    print("="*60)
    print(f"\n📂 Bucket: {bucket} ({len(parquet_files)} Parquet files)")

    # Truncate catalog
    for t in ["table_metadata", "column_metadata", "relationship_metadata", "kpi_patterns"]:
        cur.execute(f"TRUNCATE catalog.{t} RESTART IDENTITY CASCADE")

    all_texts = []
    table_entries = []
    column_entries = []

    for key in sorted(parquet_files):
        table_name = key.replace(".parquet", "")
        print(f"\n   📊 {table_name}")

        # Read schema + sample
        info = read_parquet_schema(s3, bucket, key)
        _, full_df = read_parquet_sample(s3, bucket, key, n=1000)

        # Table metadata
        detail_bi = TABLE_DETAIL_BI.get(table_name, "")
        # Auto-generate detail_agent from data statistics
        detail_agent_parts = [f"{info['num_rows']:,} rows, {info['num_columns']} columns."]
        for col in full_df.columns[:5]:
            nunique = full_df[col].nunique()
            detail_agent_parts.append(f"{col}: {nunique} distinct")
        detail_agent = " ".join(detail_agent_parts)

        txt = build_table_text(table_name, info)
        all_texts.append(txt)
        table_entries.append({
            "table_name": table_name,
            "lake_path": f"s3://{bucket}/{key}",
            "row_count": info["num_rows"],
            "file_size_bytes": info["file_size"],
            "column_count": info["num_columns"],
            "description": TABLE_DETAIL_BI.get(table_name, "")[:200] if TABLE_DETAIL_BI.get(table_name) else None,
            "detail_bi": detail_bi,
            "detail_agent": detail_agent,
            "classification": classify_table(table_name),
            "contains_pii": any((table_name, c) in PII_COLUMNS for c in full_df.columns),
            "metadata_text": txt,
        })

        # Column metadata
        for i, field in enumerate(info["schema"]):
            col_name = field.name
            arrow_type = str(field.type)
            # Compute stats from sample
            stats = {}
            if col_name in full_df.columns:
                col_series = full_df[col_name]
                stats["n_distinct"] = int(col_series.nunique())
                stats["null_fraction"] = float(col_series.isnull().mean())
                try:
                    stats["sample_values"] = [str(v) for v in col_series.dropna().value_counts().head(5).index.tolist()]
                    stats["min_value"] = str(col_series.dropna().min()) if len(col_series.dropna()) > 0 else None
                    stats["max_value"] = str(col_series.dropna().max()) if len(col_series.dropna()) > 0 else None
                except Exception:
                    pass

            bi_detail = COLUMN_DETAIL_BI.get((table_name, col_name), "")
            agent_detail = f"{stats.get('n_distinct', '?')} distinct values. Null rate: {stats.get('null_fraction', 0):.1%}."
            if stats.get("sample_values"):
                agent_detail += f" Common values: {', '.join(stats['sample_values'][:3])}."

            is_pii = (table_name, col_name) in PII_COLUMNS
            masking = PII_COLUMNS.get((table_name, col_name), "none")
            cls = "confidential" if is_pii else classify_table(table_name)

            txt = build_column_text(table_name, col_name, arrow_type, stats)
            all_texts.append(txt)
            column_entries.append({
                "table_name": table_name, "column_name": col_name,
                "ordinal_position": i + 1, "data_type": arrow_type,
                "arrow_type": arrow_type, "is_nullable": True,
                "n_distinct": stats.get("n_distinct"),
                "null_fraction": stats.get("null_fraction"),
                "sample_values": stats.get("sample_values"),
                "min_value": stats.get("min_value"),
                "max_value": stats.get("max_value"),
                "detail_bi": bi_detail, "detail_agent": agent_detail,
                "is_pii": is_pii, "classification": cls,
                "masking_rule": masking, "metadata_text": txt,
            })
        print(f"      {info['num_columns']} columns, {info['num_rows']:,} rows, {classify_table(table_name)}")

    # Relationships
    rel_texts = [build_relationship_text(r) for r in RELATIONSHIPS]
    all_texts.extend(rel_texts)

    # KPI patterns
    kpi_texts = [f"{k['kpi_name']}: {k['kpi_description']}" for k in KPI_PATTERNS]
    all_texts.extend(kpi_texts)

    # Embed all texts
    print(f"\n🧠 Generating Voyage finance-2 embeddings ({len(all_texts)} texts)...")
    BATCH = 50
    all_embeddings = []
    for i in range(0, len(all_texts), BATCH):
        batch = all_texts[i:i+BATCH]
        embs = embed_batch(batch, input_type="document")
        all_embeddings.extend(embs)
        print(f"   Embedded {min(i+BATCH, len(all_texts))}/{len(all_texts)}")

    # Split embeddings
    idx = 0
    table_embs = all_embeddings[idx:idx+len(table_entries)]; idx += len(table_entries)
    column_embs = all_embeddings[idx:idx+len(column_entries)]; idx += len(column_entries)
    rel_embs = all_embeddings[idx:idx+len(RELATIONSHIPS)]; idx += len(RELATIONSHIPS)
    kpi_embs = all_embeddings[idx:]

    # Store tables
    print(f"\n💾 Storing in catalog schema...")
    for t, emb in zip(table_entries, table_embs):
        cur.execute("""
            INSERT INTO catalog.table_metadata
            (schema_name, table_name, lake_path, file_format, row_count, file_size_bytes,
             column_count, description, detail_bi, detail_agent, classification,
             contains_pii, metadata_text, embedding)
            VALUES ('raw_lake',%s,%s,'parquet',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
        """, (t["table_name"], t["lake_path"], t["row_count"], t["file_size_bytes"],
              t["column_count"], t.get("description"), t["detail_bi"], t["detail_agent"],
              t["classification"], t["contains_pii"], t["metadata_text"], str(emb)))
    print(f"   ✓ {len(table_entries)} table metadata records")

    # Store columns
    for c, emb in zip(column_entries, column_embs):
        cur.execute("""
            INSERT INTO catalog.column_metadata
            (schema_name, table_name, column_name, ordinal_position, data_type,
             arrow_type, is_nullable, n_distinct, null_fraction,
             sample_values, min_value, max_value,
             detail_bi, detail_agent, is_pii, classification, masking_rule,
             metadata_text, embedding)
            VALUES ('raw_lake',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
        """, (c["table_name"], c["column_name"], c["ordinal_position"],
              c["data_type"], c["arrow_type"], c["is_nullable"],
              c.get("n_distinct"), c.get("null_fraction"),
              c.get("sample_values"), c.get("min_value"), c.get("max_value"),
              c["detail_bi"], c["detail_agent"],
              c["is_pii"], c["classification"], c["masking_rule"],
              c["metadata_text"], str(emb)))
    print(f"   ✓ {len(column_entries)} column metadata records")

    # Store relationships
    for r, emb in zip(RELATIONSHIPS, rel_embs):
        txt = build_relationship_text(r)
        cur.execute("""
            INSERT INTO catalog.relationship_metadata
            (source_schema, source_table, source_column,
             target_schema, target_table, target_column,
             relationship_type, join_condition, metadata_text, embedding)
            VALUES ('raw_lake',%s,%s,'raw_lake',%s,%s,%s,%s,%s,%s::vector)
        """, (r[0], r[1], r[2], r[3], r[4],
              f"{r[0]}.{r[1]} = {r[2]}.{r[3]}", txt, str(emb)))
    print(f"   ✓ {len(RELATIONSHIPS)} relationship records")

    # Store KPI patterns
    for k, emb in zip(KPI_PATTERNS, kpi_embs):
        txt = f"{k['kpi_name']}: {k['kpi_description']}"
        cur.execute("""
            INSERT INTO catalog.kpi_patterns
            (kpi_name, kpi_description, domain, required_tables, required_columns,
             sql_template, classification, metadata_text, embedding)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
        """, (k["kpi_name"], k["kpi_description"], k["domain"],
              k["required_tables"], k.get("required_columns"),
              k.get("sql_template"), k["classification"], txt, str(emb)))
    print(f"   ✓ {len(KPI_PATTERNS)} KPI patterns")

    conn.commit()
    conn.close()

    print(f"\n{'='*60}")
    print(f"✅ Metadata catalog populated!")
    print(f"   Tables:        {len(table_entries)}")
    print(f"   Columns:       {len(column_entries)}")
    print(f"   Relationships: {len(RELATIONSHIPS)}")
    print(f"   KPI patterns:  {len(KPI_PATTERNS)}")
    print(f"   Total vectors: {len(all_texts)}")
    print(f"   Index type:    HNSW (vector_cosine_ops)")
    print(f"   Embedding:     Voyage finance-2 (1024d)")
    print(f"{'='*60}")

if __name__ == "__main__":
    run()
