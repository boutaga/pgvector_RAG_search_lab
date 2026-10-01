"""Generate the synthetic bank: three tenants, clients, staff, servers and free text.

Seeded: the same seed gives the same data on every run. Writes straight into the
bank database as lab_admin, and writes the labelled question set and the list of
planted residual values to data/.

Everything is invented. IBANs use a fictitious clearing number (99xxx) with a
valid checksum; no name, host or number belongs to a real party.

    python data/generate_bank.py                    # data and questions
    python data/generate_bank.py --questions-only   # rewrite questions.json, leave the database alone
"""
import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from _common import DATA_DIR, admin_conn  # noqa: E402

SEED = 16
rng = random.Random(SEED)

BANKS = [
    ("bank_a", "Alder Private Bank", "alb", "alder", 11),
    ("bank_b", "Birchwood Bank", "bwb", "birchwood", 12),
    ("bank_c", "Cedar Trust", "cdt", "cedar", 13),
]

FIRST = ["Anna", "Luca", "Sophie", "Marco", "Elena", "David", "Laura", "Thomas", "Chiara", "Nicolas",
         "Julia", "Stefan", "Camille", "Matteo", "Nora", "Pascal", "Lea", "Fabian", "Ines", "Simon",
         "Martina", "Yves", "Sara", "Reto", "Aline", "Jonas", "Valentina", "Hugo", "Mirjam", "Olivier",
         "Petra", "Cedric", "Livia", "Urs", "Delphine", "Gian", "Monika", "Loic", "Selina", "Andrea"]
LAST = ["Keller", "Brunner", "Favre", "Rossi", "Meier", "Gerber", "Bonvin", "Huber", "Marti", "Rochat",
        "Steiner", "Fontana", "Baumann", "Perret", "Schmid", "Bianchi", "Graf", "Moser", "Vuilleumier",
        "Frei", "Zbinden", "Lombardi", "Hofer", "Python", "Wyss", "Monnier", "Kuhn", "Ferrari",
        "Aebischer", "Bachmann", "Chappuis", "Galli", "Imhof", "Jaquet", "Lehmann", "Maurer", "Pittet",
        "Ruegg", "Sutter", "Tschopp", "Vogel", "Widmer", "Zanetti", "Ammann", "Berset", "Caflisch",
        "Dubois", "Egli", "Fasel", "Gisler"]
CO_WORD = ["Alpenrose", "Seeland", "Rhonetal", "Jura", "Lindenhof", "Gotthard", "Silvretta", "Lavaux",
           "Bernina", "Emmental", "Aaretal", "Toggenburg", "Pilatus", "Saane", "Engadin", "Leman",
           "Rigi", "Napf", "Chasseral", "Furka", "Sihltal", "Glarus", "Thurgau", "Mendrisio", "Brenta",
           "Simplon", "Grimsel", "Albula", "Maggia", "Areuse"]
CO_TRADE = ["Logistik", "Immobilien", "Pharma", "Trading", "Maschinenbau", "Holding", "Uhren",
            "Energie", "Bau", "Textil", "Medtech", "Agro", "Software", "Transport"]
CO_SUFFIX = ["AG", "SA", "GmbH", "Holding AG", "Sarl"]
DOMICILES = ["Zurich", "Geneva", "Lausanne", "Basel", "Lugano", "Bern", "Zug", "Luxembourg",
             "Monaco", "London", "Milan", "Lyon"]

# Advisor topics: (key, question phrase, document phrasings)
# The question phrase must cover EVERY phrasing of its topic: a question about "voluntary
# pension buy-ins" whose expected documents talk about pillar 3a or early withdrawal
# rewards a correct "nothing found" as if it were an answer (found in review, 2026-10-01).
ADVISOR_TOPICS = [
    ("succession", "succession planning for the family business",
     ["wants to organise the transfer of the family business to the next generation",
      "raised the question of who takes over the company when the founder retires",
      "asked for a succession plan covering shares, governance and taxes"]),
    ("mortgage", "their mortgage",
     ["wants to refinance the mortgage on the main residence before the fixed rate expires",
      "asked whether to roll the mortgage into a SARON-based product",
      "discussed amortisation options for the property loan coming due"]),
    ("esg", "moving the portfolio to sustainable investments",
     ["asked to exclude fossil fuel producers from the portfolio",
      "wants the mandate aligned with a sustainability rating",
      "requested an ESG screening of current equity holdings"]),
    ("fx", "hedging currency exposure",
     ["is worried about the euro weakening against the franc",
      "asked for a forward contract to cover expected USD receivables",
      "wants to reduce the currency risk on foreign revenues"]),
    ("fees", "the fees they are charged",
     ["complained that the custody fees rose without notice",
      "contested the fee statement for the last quarter",
      "threatened to move assets because of the new fee schedule"]),
    ("kyc", "renewing know-your-customer documents",
     ["has not yet returned the updated passport copy for the KYC review",
      "needs to confirm the source of wealth for the periodic review",
      "was reminded that the beneficial owner declaration has expired"]),
    ("credit", "a Lombard credit line",
     ["asked for a Lombard loan against the securities portfolio",
      "wants to increase the credit line pledged on the custody account",
      "discussed the loan-to-value limits of the pledged securities"]),
    ("pension", "their pension planning",
     ["asked about a voluntary buy-in into the second pillar",
      "wants to optimise pillar 3a contributions before year end",
      "discussed early withdrawal of pension assets to buy property"]),
]
RARE_TOPICS = [
    ("sanctions", "a sanctions screening alert",
     ["triggered a sanctions screening alert on an incoming wire, payment held pending review"]),
    ("dispute", "an inheritance dispute",
     ["is involved in an inheritance dispute among heirs, assets partially frozen"]),
]
INCIDENT_TOPICS = [
    ("disk", "disk space running out",
     ["the data filesystem reached 95 percent and writes started to fail",
      "the archive volume filled up overnight and the backup job stopped",
      "free disk space dropped below the alert threshold"]),
    ("replication", "replication lag",
     ["the standby fell several minutes behind the primary",
      "replication lag grew during the batch window and failover was blocked",
      "the replica stopped applying changes after a network blip"]),
    ("cert", "an expired TLS certificate",
     ["clients failed to connect because the TLS certificate expired",
      "the certificate on the listener was not renewed in time",
      "TLS handshakes failed after the certificate validity ended"]),
    ("backup", "a failed backup",
     ["the nightly backup failed with a checksum error",
      "the full backup did not complete and the retention chain is broken",
      "the backup job timed out before the end of the window"]),
    ("cpu", "CPU saturation",
     ["CPU usage stayed at 100 percent and queries queued up",
      "a runaway report saturated all cores for an hour",
      "load average spiked and response times degraded"]),
    ("patch", "a failed patch",
     ["the monthly OS patch left the service unable to start",
      "patching was rolled back after the kernel update broke the driver",
      "the database minor upgrade failed on restart"]),
]
SERVER_ROLES = [("pg", "PostgreSQL database"), ("ora", "Oracle database"), ("app", "application server"),
                ("mq", "message broker"), ("web", "web front end")]


def iban(clearing, account):
    bban = f"{clearing:05d}{account:012d}"
    # mod-97 check digits: move country code + 00 to the end, letters to numbers (C=12, H=17)
    num = int(bban + "121700")
    check = 98 - num % 97
    return f"CH{check:02d}{bban}"


def fmt_iban(s):
    return " ".join(s[i:i + 4] for i in range(0, len(s), 4))


def unique(pool_fn, used):
    while True:
        v = pool_fn()
        if v not in used:
            used.add(v)
            return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-only", action="store_true")
    args = ap.parse_args()
    used_names, used_phones = set(), set()
    banks, rms, clients, accounts, servers, docs = [], [], [], [], [], []
    questions, planted = [], []
    rm_id = client_id = account_id = server_id = doc_id = 0
    t0 = datetime(2026, 1, 5, 9, 0)

    def stamp():
        return t0 + timedelta(days=rng.randint(0, 260), minutes=rng.randint(0, 600))

    def person():
        return unique(lambda: f"{rng.choice(FIRST)} {rng.choice(LAST)}", used_names)

    def phone():
        return unique(lambda: f"+41 7{rng.randint(5, 9)} {rng.randint(100, 999)} "
                              f"{rng.randint(10, 99)} {rng.randint(10, 99)}", used_phones)

    for bank_id, bank_name, prefix, domain, octet in BANKS:
        banks.append((bank_id, bank_name))
        bank_rms = []
        for _ in range(6):
            rm_id += 1
            name = person()
            email = f"{name.lower().replace(' ', '.')}@{domain}-bank.example"
            rms.append((rm_id, bank_id, name, email))
            bank_rms.append((rm_id, name, email))

        bank_clients = []
        for i in range(80):
            client_id += 1
            if i < 50:
                name = unique(lambda: f"{rng.choice(CO_WORD)} {rng.choice(CO_TRADE)} {rng.choice(CO_SUFFIX)}",
                              used_names)
                ctype = "company"
            else:
                name, ctype = person(), "private"
            rm = rng.choice(bank_rms)
            ph = phone()
            clients.append((client_id, bank_id, name, ctype, rng.choice(DOMICILES), rm[0], ph))
            ibans = []
            for _ in range(rng.randint(1, 3)):
                account_id += 1
                ib = iban(99000 + octet, rng.randint(10**9, 10**11))
                ibans.append(ib)
                accounts.append((account_id, client_id, bank_id, ib, rng.choice(["CHF", "EUR", "USD"]),
                                 round(rng.uniform(5_000, 4_000_000), 2)))
            bank_clients.append(dict(id=client_id, name=name, rm=rm, phone=ph, ibans=ibans))

        bank_servers = []
        for n in range(10):
            server_id += 1
            role, role_label = SERVER_ROLES[n % len(SERVER_ROLES)]
            env = "prd" if n < 6 else "tst"
            host = f"{prefix}-{role}-{env}-{n + 1:02d}"
            fqdn = f"{host}.{domain}.internal"
            ip = f"10.{octet}.{rng.randint(1, 30)}.{rng.randint(2, 250)}"
            servers.append((server_id, bank_id, host, fqdn, ip, role_label,
                            "production" if env == "prd" else "test"))
            bank_servers.append(dict(host=host, fqdn=fqdn, ip=ip, role=role_label, env=env))

        # Advisor notes and emails: each client gets two topics, two or three documents each.
        client_topic_docs = {}
        for c in bank_clients:
            for key, phrase, sentences in rng.sample(ADVISOR_TOPICS, 2):
                ids = []
                for j in range(rng.choice([2, 2, 3])):
                    doc_id += 1
                    rm_name, rm_email = c["rm"][1], c["rm"][2]
                    s1 = rng.choice(sentences)
                    if j == 1:
                        title = f"Email to {c['name']}: follow-up"
                        body = (f"From {rm_email}. Dear client, following our call, {c['name']} {s1}. "
                                f"Please confirm the instructions for account {fmt_iban(rng.choice(c['ibans']))}. "
                                f"Kind regards, {rm_name}.")
                        dtype = "email"
                    else:
                        title = f"Meeting note {c['name']}"
                        body = (f"{rm_name} met {c['name']}. The client {s1}. "
                                f"Next step agreed with {c['name']}: {rm_name} prepares a proposal.")
                        if rng.random() < 0.25:
                            body += f" The client is reachable on {c['phone']}."
                        dtype = "advisor_note"
                    docs.append((doc_id, bank_id, dtype, title, body, stamp()))
                    ids.append(doc_id)
                client_topic_docs[(c["id"], key)] = (c, phrase, ids)

        # Rare topics: four clients each, one note each. Used by generic questions.
        rare_docs = {}
        for key, phrase, sentences in RARE_TOPICS:
            ids = []
            for c in rng.sample(bank_clients, 4):
                doc_id += 1
                body = (f"{c['rm'][1]} recorded that {c['name']} {sentences[0]}. "
                        f"Escalated to compliance by {c['rm'][1]}.")
                docs.append((doc_id, bank_id, "advisor_note", f"Compliance note {c['name']}", body, stamp()))
                ids.append(doc_id)
            rare_docs[key] = (phrase, ids)

        # Planted residuals: a person who is in no table (the honest limit of dictionary filtering).
        for c in rng.sample(bank_clients, 3):
            doc_id += 1
            outsider = person()
            body = (f"{c['rm'][1]} met {c['name']}, accompanied by their lawyer {outsider}, "
                    f"about the estate. The lawyer asked for copies of all statements.")
            docs.append((doc_id, bank_id, "advisor_note", f"Meeting note {c['name']}", body, stamp()))
            planted.append(dict(doc_id=doc_id, bank_id=bank_id, value=outsider,
                                why="person named only in free text, in no labelled column"))

        # Incidents: each server gets three topics, four tickets each.
        host_topic_docs = {}
        for s in bank_servers:
            for key, phrase, sentences in rng.sample(INCIDENT_TOPICS, 3):
                ids = []
                for j in range(4):
                    doc_id += 1
                    ref = [s["host"], s["fqdn"], s["ip"]][j % 3]
                    body = f"Incident on {ref} ({s['role']}): {rng.choice(sentences)}."
                    if j == 3 and s["env"] == "prd":
                        c = rng.choice(bank_clients)
                        body += f" The overnight payment batch for {c['name']} was delayed."
                    body += " Resolved by the on-call team, root cause analysis pending."
                    docs.append((doc_id, bank_id, "incident", f"INC {s['host']} {key}", body, stamp()))
                    ids.append(doc_id)
                host_topic_docs[(s["host"], key)] = (s, phrase, ids)

        # Questions: 6 client-topic, 4 host-topic (entity), 2 generic per bank.
        for (cid, key), (c, phrase, ids) in rng.sample(sorted(client_topic_docs.items()), 6):
            questions.append(dict(bank_id=bank_id, type="entity",
                                  question=f"What did {c['name']} discuss about {phrase}?",
                                  expected=ids))
        for (host, key), (s, phrase, ids) in rng.sample(sorted(host_topic_docs.items()), 4):
            questions.append(dict(bank_id=bank_id, type="entity",
                                  question=f"What happened on {host} regarding {phrase}?",
                                  expected=ids))
        for key, (phrase, ids) in rare_docs.items():
            questions.append(dict(bank_id=bank_id, type="generic",
                                  question=f"Which clients were affected by {phrase}?",
                                  expected=ids))

    (DATA_DIR / "questions.json").write_text(json.dumps(questions, indent=1) + "\n")
    if args.questions_only:
        print(f"questions {len(questions)} written; database untouched")
        return

    conn = admin_conn()
    with conn.cursor() as cur:
        cur.execute("TRUNCATE bank.banks, bank.relationship_managers, bank.clients, bank.accounts, "
                    "bank.servers, bank.documents, bank.document_mentions, bank.embedding_versions, "
                    "bank.embeddings, gov.egress_log, gov.quality_runs CASCADE")
        cur.executemany("INSERT INTO bank.banks VALUES (%s,%s)", banks)
        cur.executemany("INSERT INTO bank.relationship_managers (rm_id, bank_id, full_name, email) "
                        "VALUES (%s,%s,%s,%s)", rms)
        cur.executemany("INSERT INTO bank.clients (client_id, bank_id, client_name, client_type, domicile, "
                        "rm_id, contact_phone) VALUES (%s,%s,%s,%s,%s,%s,%s)", clients)
        cur.executemany("INSERT INTO bank.accounts (account_id, client_id, bank_id, iban, currency, balance) "
                        "VALUES (%s,%s,%s,%s,%s,%s)", accounts)
        cur.executemany("INSERT INTO bank.servers (server_id, bank_id, hostname, fqdn, ip_address, role, "
                        "environment) VALUES (%s,%s,%s,%s,%s,%s,%s)", servers)
        cur.executemany("INSERT INTO bank.documents (doc_id, bank_id, doc_type, title, body, created_at) "
                        "VALUES (%s,%s,%s,%s,%s,%s)", docs)

    (DATA_DIR / "planted.json").write_text(json.dumps(planted, indent=1) + "\n")
    print(f"banks {len(banks)}, relationship managers {len(rms)}, clients {len(clients)}, "
          f"accounts {len(accounts)}, servers {len(servers)}, documents {len(docs)}, "
          f"questions {len(questions)}, planted residuals {len(planted)}")


if __name__ == "__main__":
    main()
