-- 2_tokens.sql - layer 3, deterministic tokens: same value, same token, everywhere
-- Run as lab_admin.
\set ECHO queries

-- One client, its token, and every document that mentions it.
SELECT client_name, client_token FROM bank.clients WHERE client_id = 1;

SELECT d.doc_id, d.doc_type, left(d.body, 90) AS raw, left(d.body_tokenized, 90) AS tokenized
FROM bank.documents d
JOIN bank.document_mentions m USING (doc_id)
WHERE m.token = (SELECT client_token FROM bank.clients WHERE client_id = 1)
ORDER BY d.doc_id;

-- Entities, not strings: a server's hostname and FQDN share one token, the IP has its own.
SELECT hostname, fqdn, ip_address, host_token, fqdn_token, ip_token FROM bank.servers LIMIT 3;

-- Redacted versus tokenized: the redacted copy loses who, the tokenized copy keeps it.
SELECT left(body_redacted, 110) AS redacted, left(body_tokenized, 110) AS tokenized
FROM bank.documents WHERE doc_type = 'advisor_note' ORDER BY doc_id LIMIT 3;
