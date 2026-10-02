"""The agent: OpenAI function calling over governed database tools.

  --filtering tokenized (default)
      The question is tokenized before anything leaves. Tools run as app_agent
      inside a transaction scoped to the bank (row-level security, column
      privileges, pgAudit), and only ever return tokens. Every request to OpenAI
      passes the egress gate; one scanner hit blocks it. With --reviewer, the
      final answer is re-identified through the vault.
  --filtering off
      The naive setup, for contrast: raw question, raw document text read by the
      pipeline account. The egress gate logs what leaves but does not block.

    python python/agent.py --bank bank_a "What did Sihltal Trading AG discuss about pensions?"
    python python/agent.py --bank bank_a --reviewer "..."
    python python/agent.py --bank bank_a --filtering off "..."
"""
import argparse
import json

from openai import OpenAI

from _common import (agent_conn, embed_queries, gateway_conn, sparse_embed, sparse_literal,
                     tenant_cursor, vec_literal)
from egress import EgressBlocked, EgressGate
from filtering import Dictionary
from reidentify import TOKEN, reidentify
from tracing import langfuse, propagate_attributes

SYSTEM = ("You are an assistant for relationship managers and operations staff of a bank. "
          "Answer only from tool results. Names, hosts, accounts and addresses appear as tokens "
          "such as CLIENT_1a2b3c4d5e6f or HOST_...; keep tokens exactly as they are, never invent "
          "or alter one, and use them to call tools. Be brief and factual. Cite the documents you "
          "used as [doc N]. If the tools return nothing relevant, say so.")

def access_context(cur):
    """Who the tools run as, and which row-level security applies: recorded on every tool trace."""
    cur.execute("SELECT current_user, rolbypassrls FROM pg_roles WHERE rolname = current_user")
    role, bypass = cur.fetchone()
    cur.execute("SELECT tablename || '.' || policyname FROM pg_policies WHERE schemaname = 'bank' ORDER BY 1")
    return dict(db_role=role, bypass_rls=bypass, rls_policies=[r[0] for r in cur.fetchall()])


TOOLS = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Hybrid semantic and keyword search over advisor notes, emails and incident tickets.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for. Keep tokens verbatim."},
            "k": {"type": "integer", "description": "Number of documents, default 5", "default": 5}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "client_profile",
        "description": "Profile of one client: type, domicile, relationship manager, accounts, document count.",
        "parameters": {"type": "object", "properties": {
            "client_token": {"type": "string", "description": "A CLIENT_ token"}},
            "required": ["client_token"]}}},
    {"type": "function", "function": {
        "name": "host_documents",
        "description": "Documents that mention one server, by HOST_ or IP_ token.",
        "parameters": {"type": "object", "properties": {
            "host_token": {"type": "string", "description": "A HOST_ or IP_ token"}},
            "required": ["host_token"]}}},
]


class GovernedTools:
    """Tools run as app_agent, one transaction each, tenant set for that transaction only."""

    def __init__(self, bank_id):
        self.bank_id = bank_id
        self.conn = agent_conn()
        with self.conn.transaction(), self.conn.cursor() as cur:
            self.access = access_context(cur)

    def call(self, name, args):
        with tenant_cursor(self.conn, self.bank_id) as cur:
            if name == "search_documents":
                d = embed_queries([args["query"]])[0]
                s = sparse_embed([args["query"]])[0]
                # tokens named in the query make the search entity-aware (labelling as relevance lever)
                cur.execute("SELECT doc_id, doc_type, created_at, title, body FROM bank.search_documents("
                            "%s::vector, %s::sparsevec, %s, %s)",
                            (vec_literal(d), sparse_literal(s), int(args.get("k", 5)),
                             sorted(set(TOKEN.findall(args["query"])))))
                cols = [c.name for c in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
            if name == "client_profile":
                cur.execute("SELECT bank.client_profile(%s)", (args["client_token"],))
                return cur.fetchone()[0]
            if name == "host_documents":
                cur.execute("SELECT * FROM bank.host_documents(%s)", (args["host_token"],))
                cols = [c.name for c in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
        raise ValueError(f"unknown tool {name}")


class NaiveTools:
    """The contrast: raw text through the pipeline account, which sees everything."""

    def __init__(self, bank_id, gw_conn):
        self.bank_id = bank_id
        self.gw = gw_conn
        with gw_conn.cursor() as cur:
            self.access = access_context(cur)

    def call(self, name, args):
        if name != "search_documents":
            return {"error": "only search_documents in the naive setup"}
        d = embed_queries([args["query"]])[0]
        s = sparse_embed([args["query"]])[0]
        with self.gw.cursor() as cur:
            cur.execute("SELECT d.doc_id, d.doc_type, d.created_at::date, d.title, d.body "
                        "FROM bank.retrieve(%s::vector, %s::sparsevec, 'raw', 'hybrid', 50) r "
                        "JOIN bank.documents d USING (doc_id) WHERE d.bank_id = %s "
                        "ORDER BY r.score DESC LIMIT %s",
                        (vec_literal(d), sparse_literal(s), self.bank_id, int(args.get("k", 5))))
            cols = ["doc_id", "doc_type", "created_at", "title", "body"]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


def dry_run(question, bank_id, filtering, reviewer):
    """The agent's first step without the model: search, build the exact payload, scan, log, stop."""
    gw_conn = gateway_conn()
    gate = EgressGate(gw_conn, None)
    if filtering == "tokenized":
        with gw_conn.cursor() as cur:
            asked = Dictionary.load(cur).tokenize(question)
        tools = GovernedTools(bank_id)
    else:
        asked = question
        tools = NaiveTools(bank_id, gw_conn)
    print(f"question sent : {asked}")
    result = tools.call("search_documents", {"query": asked, "k": 5})
    context = json.dumps(result, default=str)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": asked},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                 "function": {"name": "search_documents", "arguments": json.dumps({"query": asked, "k": 5})}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": context}]
    print("  [tool] search_documents returned:")
    for r in result:
        print(f"    doc {r['doc_id']} {r['doc_type']}: {r['body'][:150]}")
    try:
        gate.send_chat(messages, TOOLS if filtering == "tokenized" else TOOLS[:1], bank_id, filtering,
                       purpose="agent_turn", dry_run=True)
        print("  payload cleared the gate (dry run: not sent)")
    except EgressBlocked as e:
        print(f"  stopped at the egress gate: {e}")
    if reviewer and filtering == "tokenized":
        print("  reviewer view, re-identified through the vault:")
        for r in result[:2]:
            print(f"    doc {r['doc_id']}: {reidentify(r['body'])[:150]}")


def run(question, bank_id, filtering, reviewer, max_turns=6, quiet=False, run_label=None):
    """Ask one question. Returns a dict with the answer, its re-identified form, the
    documents the tools returned, token usage, the sensitive values sent and the trace id;
    None if the gate blocked or turns ran out. One Langfuse trace per question."""
    with propagate_attributes(session_id=run_label, tags=[filtering, bank_id]):
        with langfuse.start_as_current_observation(name="question", as_type="agent",
                                                   metadata=dict(bank_id=bank_id, filtering=filtering)):
            result = _run(question, bank_id, filtering, reviewer, max_turns, quiet, run_label)
            if result is not None:
                result["trace_id"] = langfuse.get_current_trace_id()
    langfuse.flush()
    return result


def _run(question, bank_id, filtering, reviewer, max_turns, quiet, run_label):
    gw_conn = gateway_conn()
    gate = EgressGate(gw_conn, OpenAI(), quiet=quiet, run_label=run_label)
    say = (lambda *a: None) if quiet else print
    if filtering == "tokenized":
        with gw_conn.cursor() as cur:
            asked = Dictionary.load(cur).tokenize(question)
        tools = GovernedTools(bank_id)
    else:
        asked = question
        tools = NaiveTools(bank_id, gw_conn)
    say(f"question sent : {asked}")
    langfuse.update_current_span(input=asked)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": asked}]
    tool_specs = TOOLS if filtering == "tokenized" else TOOLS[:1]
    seen_docs, usage = [], {"prompt": 0, "completion": 0}

    for _ in range(max_turns):
        try:
            response = gate.send_chat(messages, tool_specs, bank_id, filtering)
        except EgressBlocked as e:
            say(f"stopped at the egress gate: {e}")
            return None
        if response.usage:
            usage["prompt"] += response.usage.prompt_tokens
            usage["completion"] += response.usage.completion_tokens
        msg = response.choices[0].message
        if not msg.tool_calls:
            answer = msg.content or ""
            say(f"\nanswer (as the model wrote it):\n{answer}")
            # the vault is asked only when a reviewer asks; otherwise the answer stays in tokens
            readable = reidentify(answer) if filtering == "tokenized" and reviewer else answer
            if reviewer and filtering == "tokenized":
                say(f"\nanswer re-identified through the vault (reviewer only):\n{readable}")
            langfuse.update_current_span(output=answer)  # as the model wrote it, never re-identified
            return dict(asked=asked, answer=answer, readable=readable, seen_docs=seen_docs, usage=usage,
                        values_sent=gate.values_sent)
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            say(f"  [tool] {tc.function.name}({json.dumps(args)})")
            with langfuse.start_as_current_observation(
                    name=tc.function.name, input=args, metadata=dict(tools.access, bank_id=bank_id),
                    as_type="retriever" if tc.function.name == "search_documents" else "tool") as obs:
                result = tools.call(tc.function.name, args)
                doc_ids = [r["doc_id"] for r in result if "doc_id" in r] if isinstance(result, list) else []
                obs.update(output=json.loads(json.dumps(result, default=str)),
                           metadata=dict(tools.access, bank_id=bank_id, doc_ids=doc_ids))
            seen_docs += doc_ids
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)})
    say("stopped: too many tool turns")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--bank", required=True, choices=["bank_a", "bank_b", "bank_c"])
    ap.add_argument("--filtering", choices=["off", "tokenized"], default="tokenized")
    ap.add_argument("--reviewer", action="store_true", help="re-identify the final answer through the vault")
    ap.add_argument("--dry-run", action="store_true", help="first step only, nothing sent to the model")
    args = ap.parse_args()
    (dry_run if args.dry_run else run)(args.question, args.bank, args.filtering, args.reviewer)


if __name__ == "__main__":
    main()
