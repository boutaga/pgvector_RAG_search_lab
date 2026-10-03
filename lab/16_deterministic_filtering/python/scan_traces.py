"""Scan the Langfuse trace store with the egress scanner.

Self-hosted Langfuse keeps each span twice: the raw OpenTelemetry event in object
storage (MinIO here, S3 in production) and the queryable row in ClickHouse
(events_full in Langfuse v4). For one run (its session id is the run label), this
counts the sensitive values each store now holds in inputs, outputs and metadata.
Same scanner as the egress gate.

    python python/scan_traces.py --label langfuse
"""
import argparse
import base64
import json
import os
import urllib.request
from collections import Counter

import boto3

from _common import gateway_conn
from filtering import Scanner

SQL = """
SELECT arrayElement(tags, 1) AS mode, trace_id, name, input, output,
       toJSONString(arrayZip(metadata_names, metadata_values)) AS metadata
FROM events_full FINAL  -- ReplacingMergeTree: FINAL removes duplicate span versions
WHERE session_id = {label:String} OR startsWith(session_id, {label:String} || ':')
FORMAT JSONEachRow
"""


def clickhouse(sql, **params):
    url = os.getenv("LANGFUSE_CLICKHOUSE_URL", "http://localhost:8123") + "/?" + "&".join(
        f"param_{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url, data=sql.encode(), method="POST")
    auth = f"{os.getenv('LANGFUSE_CLICKHOUSE_USER')}:{os.getenv('LANGFUSE_CLICKHOUSE_PASSWORD')}"
    req.add_header("Authorization", "Basic " + base64.b64encode(auth.encode()).decode())
    with urllib.request.urlopen(req) as r:
        return [json.loads(line) for line in r.read().decode().splitlines()]


def object_store_spans(label):
    """(mode, trace_id, text) for every span of this run in the raw event files."""
    s3 = boto3.client("s3", endpoint_url=os.getenv("LANGFUSE_S3_URL", "http://localhost:9090"),
                      aws_access_key_id=os.getenv("LANGFUSE_S3_USER"),
                      aws_secret_access_key=os.getenv("LANGFUSE_S3_PASSWORD"), region_name="auto")
    pages = s3.get_paginator("list_objects_v2").paginate(Bucket="langfuse", Prefix="events/otel/")
    for key in (o["Key"] for page in pages for o in page.get("Contents", [])):
        for resource in json.loads(s3.get_object(Bucket="langfuse", Key=key)["Body"].read()):
            for scope in resource.get("scopeSpans", []):
                for span in scope.get("spans", []):
                    attrs = {a["key"]: a["value"] for a in span.get("attributes", [])}
                    session = attrs.get("session.id", {}).get("stringValue", "")
                    if session != label and not session.startswith(label + ":"):
                        continue
                    tags = [v["stringValue"] for v in attrs.get("langfuse.trace.tags", {})
                            .get("arrayValue", {}).get("values", [])]
                    yield tags[0] if tags else "?", json.dumps(span.get("traceId")), json.dumps(attrs, ensure_ascii=False)


def report(title, spans, scanner):
    by_mode = {}
    for mode, trace_id, text in spans:
        m = by_mode.setdefault(mode, dict(traces=set(), spans=0, values=0, spans_with=0, cats=Counter()))
        hits = scanner.scan(text)
        m["traces"].add(trace_id)
        m["spans"] += 1
        m["values"] += len(hits)
        m["spans_with"] += bool(hits)
        m["cats"].update(hits)
    print(title)
    print(f"  {'mode':<10} {'traces':>6} {'spans':>6} {'spans with values':>18} {'sensitive values':>17}  categories")
    for mode in sorted(by_mode):
        m = by_mode[mode]
        print(f"  {mode:<10} {len(m['traces']):>6} {m['spans']:>6} {m['spans_with']:>18} {m['values']:>17}  "
              f"{dict(m['cats'].most_common())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    with gateway_conn().cursor() as cur:
        scanner = Scanner(cur)
    rows = clickhouse(SQL, label=args.label)
    if not rows:
        raise SystemExit(f"no spans for session {args.label!r} (is tracing on, and the run flushed?)")
    report(f"ClickHouse, table events_full, session {args.label}",
           [(r["mode"], r["trace_id"], r["input"] + "\n" + r["output"] + "\n" + r["metadata"]) for r in rows],
           scanner)
    report(f"object storage, bucket langfuse/events/otel, session {args.label}",
           list(object_store_spans(args.label)), scanner)


if __name__ == "__main__":
    main()
