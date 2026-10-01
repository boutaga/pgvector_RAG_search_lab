"""Turn tokens back into real values, through the vault, as the reidentifier role.

Only the reidentifier role can read vault.mapping, and only on the vault server.
The agent's role has no account there. Run standalone on any text:

    python python/reidentify.py "CLIENT_4b67b6ac63f7 asked about PERSON_2dd109cdc431"
"""
import re
import sys

from _common import reidentifier_conn

TOKEN = re.compile(r"\b(?:CLIENT|PERSON|EMAIL|IBAN|HOST|IP|PHONE)_[0-9a-f]{12}\b")


def reidentify(text):
    tokens = sorted(set(TOKEN.findall(text)))
    if not tokens:
        return text
    with reidentifier_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT ON (token) token, value FROM vault.mapping "
                    "WHERE token = ANY(%s) ORDER BY token, key_version DESC", (tokens,))
        mapping = dict(cur.fetchall())
    return TOKEN.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)


if __name__ == "__main__":
    print(reidentify(" ".join(sys.argv[1:])))
