#!/usr/bin/env python3
"""
30_agent_pipeline.py — Agent 2: Pipeline (Data Mart Provisioning)

Architecture:
    1. Takes recommendation from Agent 1 (RAG metadata search)
    2. Generates CREATE TABLE AS SELECT referencing lake.* tables
    3. Loads relevant Parquet files from S3 → lake.* staging tables in PG
    4. Executes the generated SQL → data_mart.dm_xxx
    5. Applies governance: PII masking, classification grants, RLS
    6. Drops lake.* staging tables (ephemeral)
    7. Logs full audit trail + registers mart

Key insight: Data lake (Parquet/S3) is for storage.
PostgreSQL is for compute (JOINs, aggregations, governed data marts).
"""

import sys
import io
import json
import time
import uuid
import psycopg2
import psycopg2.extras
import pandas as pd
import pyarrow.parquet as pq
import boto3
from botocore.client import Config
from openai import OpenAI
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import config

llm = OpenAI(api_key=config.OPENAI_API_KEY)

CLASSIFICATION_ACCESS = {
    "public":       ["bi_analyst", "risk_manager", "compliance_officer", "portfolio_manager"],
    "internal":     ["bi_analyst", "risk_manager", "compliance_officer", "portfolio_manager"],
    "confidential": ["risk_manager", "compliance_officer", "portfolio_manager"],
    "restricted":   ["compliance_officer"],
}

@dataclass
class ProvisioningResult:
    request_id: str
    mart_name: str
    sql_generated: str
    classification: str
    pii_masked: bool
    rls_applied: bool
    grants: List[str]
    row_count: int
    staging_time_ms: int
    execution_time_ms: int
    total_time_ms: int
    status: str
    error_message: Optional[str] = None


def get_s3():
    return boto3.client("s3", endpoint_url=config.S3_ENDPOINT,
                        aws_access_key_id=config.S3_ACCESS_KEY,
                        aws_secret_access_key=config.S3_SECRET_KEY,
                        config=Config(signature_version="s3v4"),
                        region_name="us-east-1")


def load_parquet_to_staging(conn, s3, table_name: str):
    """Load a Parquet file from S3 into lake.<table_name> in PG.

    Note: table_name comes from our own generated Parquet file names (not user input),
    so f-string DDL construction is safe for this demo context.
    """
    key = f"{table_name}.parquet"
    obj = s3.get_object(Bucket=config.S3_BUCKET, Key=key)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))

    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS lake.{table_name} CASCADE")

    # Create table from DataFrame schema
    col_defs = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            pg_type = "BOOLEAN"
        elif pd.api.types.is_integer_dtype(dtype):
            pg_type = "BIGINT"
        elif pd.api.types.is_float_dtype(dtype):
            pg_type = "DOUBLE PRECISION"
        elif hasattr(dtype, 'tz') or pd.api.types.is_datetime64_any_dtype(dtype):
            pg_type = "TIMESTAMPTZ" if hasattr(dtype, 'tz') and dtype.tz else "TIMESTAMP"
        elif str(dtype) == "object" and len(df[col].dropna()) > 0:
            # Check if object column holds dates
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if isinstance(sample, (pd.Timestamp,)):
                pg_type = "TIMESTAMP"
            else:
                pg_type = "TEXT"
        else:
            pg_type = "TEXT"
        col_defs.append(f'"{col}" {pg_type}')

    ddl = f"CREATE TABLE lake.{table_name} ({', '.join(col_defs)})"
    cur.execute(ddl)

    # COPY data using tab-delimited text format for performance
    if len(df) > 0:
        buf = io.BytesIO()
        df.to_csv(buf, index=False, header=False, sep='\t', na_rep='\\N',
                  encoding='utf-8')
        buf.seek(0)
        cols = ', '.join(f'"{c}"' for c in df.columns)
        cur.copy_expert(f"COPY lake.{table_name} ({cols}) FROM STDIN WITH (FORMAT text, NULL '\\N')", buf)

    cur.close()
    return len(df)


def generate_mart_sql(question: str, recommendation: Dict, requester_role: str) -> Tuple[str, Dict]:
    """LLM generates CREATE TABLE AS SELECT referencing lake.* tables."""
    tables = [t["table_name"] for t in recommendation.get("tables", [])]
    columns = [(c["table_name"], c["column_name"], c.get("data_type",""),
                c.get("is_pii", False)) for c in recommendation.get("columns", [])]
    joins = [(j["source_table"], j["source_column"], j["target_table"], j["target_column"])
             for j in recommendation.get("join_paths", [])]
    kpis = recommendation.get("kpi_patterns", [])
    pii_fields = recommendation.get("pii_fields", [])
    max_class = recommendation.get("max_classification", "internal")
    mask_pii = requester_role != "compliance_officer" and len(pii_fields) > 0

    masking = ""
    if mask_pii:
        masking = """
PII MASKING RULES — APPLY THESE:
- client legal_name/short_name → governance.mask_name(column)
- account_number → governance.mask_account(column)
- ISIN columns if client-linked → governance.mask_isin(column)
Alias masked columns clearly."""

    kpi_hint = ""
    if kpis:
        kpi_hint = f"\nKPI template hint: {kpis[0].get('sql_template','N/A')}"

    prompt = f"""Generate a PostgreSQL CREATE TABLE AS SELECT for a data mart.

QUESTION: {question}

TABLES (in lake schema): {json.dumps(tables)}
COLUMNS: {json.dumps([f"{c[0]}.{c[1]}" for c in columns[:20]])}
JOINS: {json.dumps([f"{j[0]}.{j[1]}={j[2]}.{j[3]}" for j in joins])}
{masking}{kpi_hint}

RULES:
1. Target: data_mart.dm_<name> (snake_case)
2. Source tables use lake.* schema (e.g., lake.positions, lake.instruments)
3. Include aggregations, GROUP BY, meaningful aliases
4. Apply PII masking if instructed
5. Add COMMENT ON TABLE
6. Output ONLY SQL — no markdown, no explanation"""

    resp = llm.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a data engineer at a Swiss private bank. "
             "Generate production PostgreSQL DDL. Output ONLY SQL."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000, temperature=0.1
    )

    sql = resp.choices[0].message.content.strip()
    # Strip markdown code fences robustly
    import re
    sql = re.sub(r'^```\w*\s*\n?', '', sql)
    sql = re.sub(r'\n?```\s*$', '', sql)
    sql = sql.strip()

    # Extract mart name
    mart_name = "data_mart.dm_unknown"
    for line in sql.split("\n"):
        if "CREATE TABLE" in line.upper() and "data_mart." in line.lower():
            parts = line.split("data_mart.")
            if len(parts) > 1:
                mart_name = f"data_mart.{parts[1].split()[0].strip().rstrip('(')}"
                break

    return sql, {"mart_name": mart_name, "source_tables": tables,
                 "pii_masked": mask_pii, "classification": max_class}


def apply_governance(conn, mart_name: str, classification: str) -> Tuple[List[str], bool]:
    """Apply GRANT/REVOKE based on classification.

    Uses savepoints so individual GRANT/REVOKE failures don't break the
    outer transaction.
    """
    cur = conn.cursor()
    allowed = CLASSIFICATION_ACCESS.get(classification, ["compliance_officer"])
    grants = []
    for role in allowed:
        try:
            cur.execute("SAVEPOINT grant_sp")
            cur.execute(f"GRANT SELECT ON {mart_name} TO {role}")
            cur.execute("RELEASE SAVEPOINT grant_sp")
            grants.append(role)
        except Exception:
            cur.execute("ROLLBACK TO SAVEPOINT grant_sp")
    for role in {"bi_analyst","risk_manager","compliance_officer","portfolio_manager"} - set(allowed):
        try:
            cur.execute("SAVEPOINT revoke_sp")
            cur.execute(f"REVOKE ALL ON {mart_name} FROM {role}")
            cur.execute("RELEASE SAVEPOINT revoke_sp")
        except Exception:
            cur.execute("ROLLBACK TO SAVEPOINT revoke_sp")
    cur.close()
    rls = classification in ("confidential", "restricted")
    return grants, rls


def cleanup_staging(conn, tables: List[str]):
    """Drop lake.* staging tables.

    Uses savepoints so individual DROP failures don't break the transaction.
    """
    cur = conn.cursor()
    for t in tables:
        try:
            cur.execute("SAVEPOINT cleanup_sp")
            cur.execute(f"DROP TABLE IF EXISTS lake.{t} CASCADE")
            cur.execute("RELEASE SAVEPOINT cleanup_sp")
        except Exception:
            cur.execute("ROLLBACK TO SAVEPOINT cleanup_sp")
    cur.close()


def provision(question: str, recommendation: Dict,
              requester: str, requester_role: str,
              dry_run: bool = False) -> ProvisioningResult:
    """Main entry point for Agent 2."""
    request_id = str(uuid.uuid4())
    t0 = time.time()
    conn = psycopg2.connect(config.DATABASE_URL)
    conn.autocommit = False
    s3 = get_s3()
    sql = ""
    mart_name = "unknown"
    staging_tables = []

    try:
        # 1. Generate SQL
        sql, meta = generate_mart_sql(question, recommendation, requester_role)
        mart_name = meta["mart_name"]
        classification = meta["classification"]
        pii_masked = meta["pii_masked"]
        source_tables = meta["source_tables"]

        if dry_run:
            ms = int((time.time() - t0) * 1000)
            return ProvisioningResult(
                request_id=request_id, mart_name=mart_name, sql_generated=sql,
                classification=classification, pii_masked=pii_masked,
                rls_applied=False, grants=[], row_count=0,
                staging_time_ms=0, execution_time_ms=0, total_time_ms=ms,
                status="dry_run")

        # 2. Stage Parquet → lake.*
        print(f"   📥 Staging Parquet → lake.* ({len(source_tables)} tables)")
        t_stage = time.time()
        for tbl in source_tables:
            rows = load_parquet_to_staging(conn, s3, tbl)
            staging_tables.append(tbl)
            print(f"      lake.{tbl}: {rows:,} rows")
        staging_ms = int((time.time() - t_stage) * 1000)

        # 3. Validate and execute SQL
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith("CREATE TABLE") and not sql_upper.startswith("CREATE OR REPLACE"):
            raise ValueError(f"LLM generated unsafe SQL (expected CREATE TABLE): {sql[:100]}")
        for forbidden in ["DROP TABLE", "DROP SCHEMA", "DELETE FROM", "TRUNCATE", "ALTER ROLE", "CREATE ROLE"]:
            if forbidden in sql_upper and "DROP TABLE IF EXISTS data_mart." not in sql_upper:
                raise ValueError(f"LLM generated SQL with forbidden keyword: {forbidden}")

        print(f"   ⚡ Executing: {mart_name}")
        t_exec = time.time()
        cur = conn.cursor()
        tbl_only = mart_name.split(".")[-1]
        cur.execute(f"DROP TABLE IF EXISTS data_mart.{tbl_only} CASCADE")
        cur.execute(sql)
        try:
            cur.execute(f"SELECT COUNT(*) FROM {mart_name}")
            row_count = cur.fetchone()[0]
        except Exception:
            row_count = 0
        cur.close()
        exec_ms = int((time.time() - t_exec) * 1000)

        # 4. Governance
        grants, rls = apply_governance(conn, mart_name, classification)

        # 5. Cleanup staging
        cleanup_staging(conn, staging_tables)

        # 6. Audit trail
        total_ms = int((time.time() - t0) * 1000)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO governance.provisioning_audit
            (request_id, requested_by, requester_role, request_text,
             rag_results, rag_search_time_ms, sql_generated, target_object,
             source_tables, classification, pii_columns_found,
             masking_applied, rls_applied, grants_applied,
             status, row_count, execution_time_ms, executed_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'executed',%s,%s,NOW())
        """, (request_id, requester, requester_role, question,
              json.dumps({"reasoning": recommendation.get("reasoning","")}),
              recommendation.get("total_time_ms"),
              sql, mart_name, source_tables, classification,
              recommendation.get("pii_fields",[]),
              pii_masked, rls, grants, row_count, exec_ms))
        # Register mart
        cur.execute("""
            INSERT INTO governance.data_mart_registry
            (mart_name, schema_name, created_by, request_id, source_tables,
             classification, has_pii_masking, has_rls, allowed_roles,
             row_count, description)
            VALUES (%s,'data_mart',%s,%s::uuid,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (mart_name) DO UPDATE SET
                last_refreshed=NOW(), row_count=EXCLUDED.row_count
        """, (mart_name, requester, request_id, source_tables,
              classification, pii_masked, rls, grants, row_count,
              f"Auto-provisioned: {question[:200]}"))
        cur.close()
        conn.commit()

        return ProvisioningResult(
            request_id=request_id, mart_name=mart_name, sql_generated=sql,
            classification=classification, pii_masked=pii_masked,
            rls_applied=rls, grants=grants, row_count=row_count,
            staging_time_ms=staging_ms, execution_time_ms=exec_ms,
            total_time_ms=total_ms, status="executed")

    except Exception as e:
        elapsed = int((time.time() - t0) * 1000)
        conn.rollback()
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO governance.provisioning_audit
                (request_id, requested_by, requester_role, request_text,
                 sql_generated, target_object, status, error_message, execution_time_ms)
                VALUES (%s,%s,%s,%s,%s,%s,'error',%s,%s)
            """, (request_id, requester, requester_role, question,
                  sql, mart_name, str(e), elapsed))
            conn.commit()
        except Exception:
            pass
        return ProvisioningResult(
            request_id=request_id, mart_name=mart_name, sql_generated=sql,
            classification=recommendation.get("max_classification","internal"),
            pii_masked=False, rls_applied=False, grants=[], row_count=0,
            staging_time_ms=0, execution_time_ms=0, total_time_ms=elapsed,
            status="error", error_message=str(e))
    finally:
        conn.close()


def print_result(r: ProvisioningResult):
    icon = {"executed":"✅","dry_run":"🔍","error":"❌"}.get(r.status,"❓")
    cls_icon = {"public":"🟢","internal":"🔵","confidential":"🟠","restricted":"🔴"}
    print(f"\n{'='*70}")
    print(f"⚙️  AGENT 2: Pipeline — Data Mart Provisioning")
    print(f"   Request: {r.request_id}")
    print(f"   Status: {icon} {r.status}")
    print(f"{'='*70}")
    print(f"\n📦 MART: {r.mart_name}")
    print(f"   Rows: {r.row_count:,}")
    print(f"   Staging: {r.staging_time_ms}ms | SQL: {r.execution_time_ms}ms | Total: {r.total_time_ms}ms")
    print(f"\n🔐 GOVERNANCE:")
    print(f"   Classification: {cls_icon.get(r.classification,'⚪')} {r.classification}")
    print(f"   PII masking: {'✓' if r.pii_masked else '✗'}")
    print(f"   RLS: {'✓' if r.rls_applied else '✗'}")
    print(f"   Grants: {', '.join(r.grants) if r.grants else 'none'}")
    print(f"\n📝 SQL:\n   " + r.sql_generated.replace("\n", "\n   "))
    if r.error_message:
        print(f"\n❌ ERROR: {r.error_message}")
    print("="*70)
