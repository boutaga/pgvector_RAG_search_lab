"""Count the sensitive values the egress scanner finds in every document, per text state.

    python python/scan_corpus.py
"""
from collections import Counter

from _common import gateway_conn
from filtering import Scanner

STATES = {"raw": ("title", "body"), "redacted": ("title_redacted", "body_redacted"),
          "tokenized": ("title_tokenized", "body_tokenized")}


def main():
    with gateway_conn().cursor() as cur:
        scanner = Scanner(cur)
        for state, (t, b) in STATES.items():
            cur.execute(f"SELECT {t} || ' ' || {b} FROM bank.documents ORDER BY doc_id")
            counts, docs_hit = Counter(), 0
            for (text,) in cur.fetchall():
                hits = scanner.scan(text)
                counts.update(hits)
                docs_hit += bool(hits)
            detail = ", ".join(f"{k} {v}" for k, v in counts.most_common()) or "none"
            print(f"{state:<10} {sum(counts.values()):>5} values in {docs_hit:>4} documents: {detail}")


if __name__ == "__main__":
    main()
