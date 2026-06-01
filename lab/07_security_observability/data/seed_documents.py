"""Seed the 3-tenant synthetic finance corpus and build the labelled query set.

No OpenAI calls here (no key needed). Inserts documents and writes
data/test_cases.json with exact expected_doc_ids (relevance is known by
construction). Run embed.py afterwards to create the embeddings.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from _common import admin_conn, LAB_DIR  # noqa: E402

random.seed(42)

TENANTS = ["bank_a", "bank_b", "bank_c"]

# Distinct (fictional) clients per bank — cross-tenant leaks are about other
# banks' companies, which is the realistic confidentiality breach.
CLIENTS = {
    "bank_a": ["Helvetia Industrials AG", "Aare Logistics SA", "Rhone Pharma Holding",
               "Jura Precision GmbH", "Leman Energy AG"],
    "bank_b": ["Matterhorn Capital AG", "Ticino Foods SpA", "Basilea Chemicals AG",
               "Engadin Resorts SA", "Reuss Robotics GmbH"],
    "bank_c": ["Saentis Mobility AG", "Limmat Media Group", "Gotthard Freight SA",
               "Vierwald Insurance AG", "Furka Mining Ltd"],
}

TOPICS = [
    ("Q3 trading positions", "trading", "confidential"),
    ("credit exposure review", "credit", "confidential"),
    ("FX hedging strategy", "fx", "internal"),
    ("liquidity stress test", "liquidity", "internal"),
]


def make_body(client, key):
    bodies = {
        "trading": (f"{client} increased its equity book this quarter. The desk reports "
                    f"net long exposure in industrial and energy names, with {client} "
                    f"rotating out of short-dated rates."),
        "credit": (f"The credit committee reviewed {client}'s counterparty exposure. "
                   f"{client} sits within limits, though the unsecured line to two "
                   f"suppliers was flagged for renewal."),
        "fx": (f"{client} hedges EURCHF and USDCHF cash flows on a rolling six-month "
               f"basis. Treasury proposes extending {client}'s hedge ratio given franc "
               f"strength."),
        "liquidity": (f"Under the severe scenario, {client} maintains a 30-day liquidity "
                      f"buffer above the regulatory floor. {client}'s drawdown "
                      f"assumptions were tightened after the last review."),
    }
    return bodies[key]


def main():
    conn = admin_conn()
    cur = conn.cursor()
    cur.execute("TRUNCATE app.embeddings, app.documents RESTART IDENTITY CASCADE;")

    docs = []  # (id, tenant, client, key)
    for tenant in TENANTS:
        for client in CLIENTS[tenant]:
            for label, key, cls in TOPICS:
                title = f"{client}: {label}"
                body = make_body(client, key)
                cur.execute(
                    "INSERT INTO app.documents(tenant_id, client_name, title, body, classification) "
                    "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (tenant, client, title, body, cls),
                )
                docs.append((cur.fetchone()[0], tenant, client, key))

    # Build labelled queries (relevance known by construction).
    cases = []
    for tenant in TENANTS:
        # client queries — most sensitive to client-name masking
        for client in CLIENTS[tenant]:
            rel = [d[0] for d in docs if d[1] == tenant and d[2] == client]
            cases.append({
                "query": f"What are {client}'s positions and exposures?",
                "tenant_id": tenant, "query_type": "client",
                "expected_doc_ids": rel, "metadata": {"client": client},
            })
        # topic queries — robust to client-name masking
        for label, key, _cls in TOPICS:
            rel = [d[0] for d in docs if d[1] == tenant and d[3] == key]
            cases.append({
                "query": f"{label} across the portfolio",
                "tenant_id": tenant, "query_type": "topic",
                "expected_doc_ids": rel, "metadata": {"topic": key},
            })

    out = LAB_DIR / "data" / "test_cases.json"
    out.write_text(json.dumps(cases, indent=2))
    print(f"Inserted {len(docs)} documents across {len(TENANTS)} tenants.")
    print(f"Wrote {len(cases)} labelled queries to {out}")


if __name__ == "__main__":
    main()
