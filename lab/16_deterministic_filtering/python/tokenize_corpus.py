"""Tokenize the bank: structured columns through the vault, then free text by dictionary.

  1. For every registered column, send its distinct values to vault.tokenize()
     (the key never leaves the vault) and write the tokens back into the bank.
     Entities, not strings: a server's FQDN gets its hostname's token.
  2. Build the dictionary from the tokenized columns and filter every document
     into a redacted copy and a tokenized copy, recording which tokens it mentions.

Re-run after changing the registry (for example after labelling contact_phone)
or after rotating the vault key.

    python python/tokenize_corpus.py
"""
from _common import gateway_conn, tokenizer_conn
from filtering import Dictionary


def tokenize_columns(gw, vault):
    gw.execute("SELECT table_name, column_name, category, token_column "
               "FROM gov.sensitive_columns WHERE token_column IS NOT NULL ORDER BY 1, 2")
    registry = gw.fetchall()
    with vault.cursor() as vc:
        vc.execute("SELECT vault.active_key_version()")
        key_version = vc.fetchone()[0]
        for table, column, category, token_column in registry:
            source = "hostname" if (table, column) == ("servers", "fqdn") else column
            gw.execute(f"SELECT DISTINCT {source} FROM bank.{table}")
            values = [r[0] for r in gw.fetchall()]
            vc.execute("SELECT value, token FROM vault.tokenize(%s, %s)", (category, values))
            pairs = vc.fetchall()
            gw.executemany(f"UPDATE bank.{table} SET {token_column} = %s WHERE {source} = %s",
                           [(tok, val) for val, tok in pairs])
            print(f"  {table}.{column:<12} {category:<7} {len(pairs):>4} tokens")
    # columns that were unregistered since the last run lose their tokens
    gw.execute("SELECT 1 FROM gov.sensitive_columns WHERE table_name = 'clients' AND column_name = 'contact_phone'")
    if gw.fetchone() is None:
        gw.execute("UPDATE bank.clients SET phone_token = NULL")
    return key_version


def filter_documents(gw, key_version):
    dictionary = Dictionary.load(gw)
    gw.execute("SELECT doc_id, bank_id, title, body FROM bank.documents ORDER BY doc_id")
    docs = gw.fetchall()
    updates, mentions = [], []
    for doc_id, bank_id, title, body in docs:
        updates.append((dictionary.redact(title), dictionary.redact(body),
                        dictionary.tokenize(title), dictionary.tokenize(body), key_version, doc_id))
        for token, category in dictionary.mentions(title + " " + body).items():
            mentions.append((doc_id, bank_id, token, category))
    gw.executemany("UPDATE bank.documents SET title_redacted = %s, body_redacted = %s, "
                   "title_tokenized = %s, body_tokenized = %s, key_version = %s WHERE doc_id = %s",
                   updates)
    gw.execute("DELETE FROM bank.document_mentions")
    gw.executemany("INSERT INTO bank.document_mentions VALUES (%s, %s, %s, %s)", mentions)
    print(f"  {len(docs)} documents filtered, {len(mentions)} mentions, "
          f"{len(dictionary.token_of)} dictionary entries")


def main():
    gw_conn = gateway_conn()
    with gw_conn.cursor() as gw, tokenizer_conn() as vault:
        print("tokenizing registered columns through the vault")
        key_version = tokenize_columns(gw, vault)
        print(f"filtering free text (key version {key_version})")
        filter_documents(gw, key_version)


if __name__ == "__main__":
    main()
