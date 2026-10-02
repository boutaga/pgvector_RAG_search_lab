"""The egress gate: every request to an external API goes through send_chat().

Before a request leaves, the scanner looks for any known raw value and any
sensitive shape in the serialized payload. The result is written to
gov.egress_log (hash and size of the payload, never the payload itself), with
its outcome: only 'sent' left the perimeter; 'blocked' and 'dry_run' never did.
With filtering on, a single hit blocks the request.
"""
import hashlib
import json

from _common import CHAT_MODEL
from filtering import Scanner
from tracing import langfuse


class EgressBlocked(Exception):
    pass


class EgressGate:
    def __init__(self, gw_conn, client, quiet=False, run_label=None):
        self.gw = gw_conn
        self.quiet = quiet
        self.run_label = run_label
        self.client = client
        self.values_sent = 0
        with gw_conn.cursor() as cur:
            self.scanner = Scanner(cur)

    def send_chat(self, messages, tools, bank_id, filtering, purpose="agent_turn", dry_run=False):
        payload = json.dumps({"model": CHAT_MODEL, "messages": messages, "tools": tools}, default=str,
                             ensure_ascii=False)  # keep accents: an escaped name would never match
        hits = self.scanner.scan(payload)
        blocked = filtering == "tokenized" and len(hits) > 0
        outcome = "blocked" if blocked else "dry_run" if dry_run else "attempted"
        with self.gw.cursor() as cur:
            cur.execute("INSERT INTO gov.egress_log (destination, purpose, bank_id, filtering, payload_sha256, "
                        "payload_chars, hits, hit_categories, blocked, outcome, run_label) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING egress_id",
                        (("dry-run:" if dry_run else "openai:") + CHAT_MODEL, purpose, bank_id, filtering,
                         hashlib.sha256(payload.encode()).hexdigest(), len(payload), len(hits),
                         sorted(set(hits)), blocked, outcome, self.run_label))
            egress_id = cur.fetchone()[0]
        if not self.quiet:
            print(f"  [egress] {len(payload)} chars, filtering={filtering}, scanner hits={len(hits)} "
                  f"{sorted(set(hits)) or ''}{'  BLOCKED' if blocked else ''}")
        meta = dict(egress_id=egress_id, filtering=filtering, scanner_hits=len(hits),
                    hit_categories=sorted(set(hits)), outcome=outcome)
        if blocked:
            # the blocked request is not traced as a model call; note that tool results
            # are traced when the tool returns, before this gate (see agent.py)
            with langfuse.start_as_current_observation(name="egress_gate", as_type="guardrail", level="WARNING",
                                                       status_message="blocked", metadata=meta):
                pass
            raise EgressBlocked(f"{len(hits)} sensitive value(s) in outbound payload: {sorted(set(hits))}")
        if dry_run:
            return payload
        with langfuse.start_as_current_observation(name="chat", as_type="generation", model=CHAT_MODEL,
                                                   input=messages, metadata=meta) as gen:
            try:
                # gpt-6-luna accepts function tools on /v1/chat/completions only with reasoning off
                response = self.client.chat.completions.create(model=CHAT_MODEL, messages=messages, tools=tools,
                                                               reasoning_effort="none")
            except Exception:
                self._outcome(egress_id, "failed")
                gen.update(level="ERROR", metadata=dict(meta, outcome="failed"))
                raise
            self._outcome(egress_id, "sent")
            self.values_sent += len(hits)
            gen.update(output=response.choices[0].message.model_dump(exclude_none=True),
                       metadata=dict(meta, outcome="sent"),
                       usage_details=dict(input=response.usage.prompt_tokens,
                                          output=response.usage.completion_tokens) if response.usage else None)
        return response

    def _outcome(self, egress_id, outcome):
        with self.gw.cursor() as cur:
            cur.execute("UPDATE gov.egress_log SET outcome = %s WHERE egress_id = %s", (outcome, egress_id))
