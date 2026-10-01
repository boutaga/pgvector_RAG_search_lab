"""Deterministic filtering: the dictionary, the text filter and the egress scanner.

The dictionary is built from the labelled columns themselves (value -> token),
so filtering is exact matching on known values, not a guessing model. What is
not labelled is not in the dictionary: the scanner's patterns are the second
line, and the post states the limit (a person named only in free text, in no
table, passes both).
"""
import re

# Patterns for the scanner: catch sensitive shapes even when no dictionary entry exists.
PATTERNS = {
    "IBAN": re.compile(r"\bCH\d{2}(?: ?[0-9A-Z]{4}){4} ?[0-9A-Z]\b"),
    "IP": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "PHONE": re.compile(r"\+41(?: ?\d){9}"),
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "HOST": re.compile(r"\b[a-z]{3}-[a-z]{2,3}-(?:prd|tst)-\d{2}\b"),
}

# Which registry entries feed the dictionary, and the spelling variants found in free text.
DICTIONARY_SQL = """
SELECT c.client_name, c.client_token, 'CLIENT' FROM bank.clients c WHERE c.client_token IS NOT NULL
UNION ALL SELECT r.full_name, r.rm_token, 'PERSON' FROM bank.relationship_managers r WHERE r.rm_token IS NOT NULL
UNION ALL SELECT r.email, r.email_token, 'EMAIL' FROM bank.relationship_managers r WHERE r.email_token IS NOT NULL
UNION ALL SELECT a.iban, a.iban_token, 'IBAN' FROM bank.accounts a WHERE a.iban_token IS NOT NULL
UNION ALL SELECT s.hostname, s.host_token, 'HOST' FROM bank.servers s WHERE s.host_token IS NOT NULL
UNION ALL SELECT s.fqdn, s.fqdn_token, 'HOST' FROM bank.servers s WHERE s.fqdn_token IS NOT NULL
UNION ALL SELECT s.ip_address, s.ip_token, 'IP' FROM bank.servers s WHERE s.ip_token IS NOT NULL
UNION ALL SELECT c.contact_phone, c.phone_token, 'PHONE' FROM bank.clients c WHERE c.phone_token IS NOT NULL
"""

# Every raw sensitive value known to the bank, tokenized or not: what the scanner hunts for.
RAW_VALUES_SQL = """
SELECT client_name, 'CLIENT' FROM bank.clients
UNION ALL SELECT contact_phone, 'PHONE' FROM bank.clients
UNION ALL SELECT full_name, 'PERSON' FROM bank.relationship_managers
UNION ALL SELECT email, 'EMAIL' FROM bank.relationship_managers
UNION ALL SELECT iban, 'IBAN' FROM bank.accounts
UNION ALL SELECT hostname, 'HOST' FROM bank.servers
UNION ALL SELECT fqdn, 'HOST' FROM bank.servers
UNION ALL SELECT ip_address, 'IP' FROM bank.servers
"""


def _variants(value, category):
    if category == "IBAN":  # stored compact, written in groups of four
        return {value, " ".join(value[i:i + 4] for i in range(0, len(value), 4))}
    return {value}


def _alternation(values):
    # longest first, so an FQDN wins over the hostname it contains
    parts = sorted(values, key=len, reverse=True)
    return re.compile(r"(?<![\w-])(" + "|".join(re.escape(p) for p in parts) + r")(?![\w-])",
                      re.IGNORECASE)


class Dictionary:
    def __init__(self, rows):
        self.token_of = {}
        self.category_of = {}
        for value, token, category in rows:
            for v in _variants(value, category):
                self.token_of[v.lower()] = token
                self.category_of[v.lower()] = category
        self.regex = _alternation(self.token_of) if self.token_of else None

    @classmethod
    def load(cls, cur):
        cur.execute(DICTIONARY_SQL)
        return cls(cur.fetchall())

    def tokenize(self, text):
        if not self.regex:
            return text
        return self.regex.sub(lambda m: self.token_of[m.group(1).lower()], text)

    def redact(self, text):
        if not self.regex:
            return text
        return self.regex.sub("[REDACTED]", text)

    def mentions(self, text):
        if not self.regex:
            return {}
        return {self.token_of[m.group(1).lower()]: self.category_of[m.group(1).lower()]
                for m in self.regex.finditer(text)}


class Scanner:
    """Looks for any known raw value, and any sensitive shape, in an outbound payload."""

    def __init__(self, cur):
        cur.execute(RAW_VALUES_SQL)
        values = {}
        for value, category in cur.fetchall():
            for v in _variants(value, category):
                values[v.lower()] = category
        self.category_of = values
        self.regex = _alternation(values)

    def scan(self, text):
        hits = []
        for m in self.regex.finditer(text):
            hits.append(self.category_of[m.group(1).lower()])
        known = {m.group(1).lower() for m in self.regex.finditer(text)}
        for category, pattern in PATTERNS.items():
            for m in pattern.finditer(text):
                if m.group(0).lower() not in known:
                    hits.append(category)
        return hits
