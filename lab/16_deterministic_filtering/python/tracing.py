"""Optional Langfuse tracing (self-hosted, docker/langfuse/).

Tracing is on when LANGFUSE_PUBLIC_KEY is set in .env, off otherwise; with it off,
every observation below is a no-op and the lab runs exactly as before.

What is traced is what a tracing integration records by default: the messages
sent to the model, the documents a tool returned, the answer. Tool results are
recorded when the tool returns, before the egress gate scans the next request;
on the governed path they are tokens only because the tools only return tokens.
On the naive path it is the raw text, so the trace store becomes one more copy
of the data (measured by python/scan_traces.py).
"""
import os

import _common  # noqa: F401  loads .env before the client reads its settings
from langfuse import Langfuse, propagate_attributes

langfuse = Langfuse(tracing_enabled=bool(os.getenv("LANGFUSE_PUBLIC_KEY")))

__all__ = ["langfuse", "propagate_attributes"]
