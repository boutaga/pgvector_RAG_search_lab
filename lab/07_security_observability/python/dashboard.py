"""Streamlit dashboard — security cost across three states, with k-variation and
per-query detail. Built as talk support. Reads mock/frozen_run.json (offline-capable,
the on-stage fallback). Run with:

    streamlit run dashboard.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

LAB_DIR = Path(__file__).resolve().parents[1]
FROZEN = LAB_DIR / "mock" / "frozen_run.json"
STATE_COLORS = {"baseline": "#ef4444", "rls": "#22c55e", "masked": "#f59e0b"}

st.set_page_config(page_title="Security Observability — Swiss PGDay 2026",
                   page_icon="🔐", layout="wide")

if not FROZEN.exists():
    st.error(f"No measurement found at {FROZEN}.\n\nRun:  python measure_security_cost.py")
    st.stop()

run = json.loads(FROZEN.read_text())
states = run["states"]
k_values = run["k_values"]

st.title("The cost of security debt — measured")
st.caption(f"{run['n_queries']} queries · generated {run['generated_at']} · "
           "apply a control, measure the delta, make an informed decision.")

k = st.sidebar.selectbox("Top-k", k_values,
                         index=k_values.index(run.get("default_k", k_values[0])))
ks = str(k)
st.sidebar.markdown("**States**")
for s in states:
    st.sidebar.markdown(f"- {s['label']}")
st.sidebar.caption("Offline view of `mock/frozen_run.json`. "
                   "Re-run `measure_security_cost.py` to refresh.")


def delta(v, ref):
    d = v - ref
    return None if abs(d) < 1e-9 else f"{d:+.1%}"


tab_overview, tab_k, tab_q, tab_setup = st.tabs(
    ["Overview", "k variation", "Query detail", "Setup (SQL)"])

# ---------------------------------------------------------------- Overview ----
with tab_overview:
    cols = st.columns(len(states))
    for i, (col, s) in enumerate(zip(cols, states)):
        m = s["by_k"][ks]
        prev = states[i - 1]["by_k"][ks]
        with col:
            st.subheader(s["label"])
            if m["leak_rate"] > 0:
                st.error(f"LEAK: {m['leak_rate']:.0%} of queries return other tenants "
                         f"({m['leak_docs']} rows)")
            else:
                st.success("No cross-tenant leak")
            st.metric(f"Recall@{k}", f"{m['recall']:.3f}",
                      None if i == 0 else delta(m["recall"], prev["recall"]))
            st.metric(f"Precision@{k}", f"{m['precision']:.3f}",
                      None if i == 0 else delta(m["precision"], prev["precision"]))
            st.metric(f"nDCG@{k}", f"{m['ndcg']:.3f}",
                      None if i == 0 else delta(m["ndcg"], prev["ndcg"]))

    fig = go.Figure()
    for s in states:
        m = s["by_k"][ks]
        fig.add_bar(name=s["label"], x=["Recall", "Precision", "nDCG"],
                    y=[m["recall"], m["precision"], m["ndcg"]],
                    marker_color=STATE_COLORS.get(s["key"]))
    fig.update_layout(barmode="group", yaxis_range=[0, 1], height=380,
                      title=f"Metrics @ k={k}", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)
    st.info("RLS strips the cross-tenant noise, so it **improves** retrieval *and* closes the "
            "leak — a pure win. The measurable quality cost is the anonymization step.")

# ------------------------------------------------------------- k variation ----
with tab_k:
    st.markdown("How each metric behaves as **k** grows, per state. Deltas between the lines "
                "are the cost (or gain) of each control at that k.")
    for metric, title in [("recall", "Recall@k"), ("ndcg", "nDCG@k"),
                          ("leak_rate", "Leak rate @k")]:
        fig = go.Figure()
        for s in states:
            fig.add_trace(go.Scatter(
                x=k_values, y=[s["by_k"][str(kk)][metric] for kk in k_values],
                mode="lines+markers", name=s["label"],
                line=dict(color=STATE_COLORS.get(s["key"]), width=3)))
        fig.update_layout(title=title, xaxis_title="k", yaxis_range=[0, 1], height=300,
                          legend=dict(orientation="h", y=-0.3),
                          margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------- Query detail ----
with tab_q:
    queries = run["queries"]
    types = sorted(set(q["query_type"] for q in queries))
    ftype = st.selectbox("Filter by query type", ["all"] + types,
                         help="client queries carry the client name (sensitive to masking); "
                              "topic queries do not")
    pool = [q for q in queries if ftype == "all" or q["query_type"] == ftype]
    labels = [f"[{q['query_type']}] {q['tenant_id']}: {q['query']}" for q in pool]
    j = st.selectbox("Query", range(len(pool)), format_func=lambda i: labels[i])
    q = pool[j]
    st.markdown(f"**{q['query']}** · tenant `{q['tenant_id']}` · type `{q['query_type']}` · "
                f"{len(q['expected_doc_ids'])} relevant docs · expected ids "
                f"`{q['expected_doc_ids']}`")

    mcols = st.columns(len(states))
    for col, s in zip(mcols, states):
        rec = q["states"][s["key"]]["by_k"][ks]
        with col:
            st.caption(s["label"])
            st.write(f"Recall **{rec['recall']:.2f}** · Prec **{rec['precision']:.2f}** · "
                     f"nDCG **{rec['ndcg']:.2f}**")

    for s in states:
        st.markdown(f"**{s['label']}** — top {k} retrieved")
        rows = q["states"][s["key"]]["retrieved"][:k]
        df = pd.DataFrame([{
            "rank": r["rank"], "tenant": r["tenant"], "doc": r["doc_id"],
            "dist": r["dist"],
            "relevant": "✅" if r["relevant"] else "",
            "leak": "🚨" if r["leak"] else "",
            "snippet": r["snippet"],
        } for r in rows])
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(f"**All {len(queries)} queries — Recall@{k} by state** (click a header to sort)")
    table = []
    for qq in queries:
        row = {"type": qq["query_type"], "tenant": qq["tenant_id"], "query": qq["query"][:48]}
        for s in states:
            row[s["key"]] = qq["states"][s["key"]]["by_k"][ks]["recall"]
        table.append(row)
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ Setup ----
with tab_setup:
    st.markdown("The SQL behind the demo — run these live in `psql` to show the setup while "
                "the audience watches the metrics move.")
    sql_files = [
        ("Schema + RLS on documents (baseline state)", "sql/init/01_schema.sql", False),
        ("pgaudit configuration", "sql/init/02_pgaudit.sql", False),
        ("THE FIX — RLS on the embeddings table", "sql/demo/10_rls_embeddings.sql", True),
        ("Anonymization (postgresql_anonymizer)", "sql/demo/20_anonymize.sql", False),
    ]
    for title, rel, default_open in sql_files:
        p = LAB_DIR / rel
        with st.expander(title, expanded=default_open):
            st.code(p.read_text() if p.exists() else f"(missing {rel})", language="sql")

    with st.expander("Run it live (psql / cli)"):
        st.code(
            "# connect as the app role (RLS applies)\n"
            "docker exec -it lab07_pg18 psql -U app_user -d secobs\n"
            "SET app.tenant_id='bank_a';\n"
            "SELECT count(DISTINCT tenant_id) FROM app.documents;    -- 1  (RLS works)\n"
            "SELECT count(DISTINCT tenant_id) FROM app.embeddings;   -- 3  (LEAK: no RLS)\n\n"
            "# the leak via the RAG path\n"
            "./.venv/bin/python python/ask.py --tenant bank_a \\\n"
            "    --query 'Q3 trading positions across the portfolio'\n\n"
            "# apply the fix, then re-run ask.py -> no leak\n"
            "docker exec -i lab07_pg18 psql -U dba_admin -d secobs < sql/demo/10_rls_embeddings.sql\n\n"
            "# watch pgaudit log the (authorized) reads\n"
            "docker logs -f lab07_pg18 2>&1 | grep AUDIT",
            language="bash")
