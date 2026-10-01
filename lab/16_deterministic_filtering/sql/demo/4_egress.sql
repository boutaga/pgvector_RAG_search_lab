-- 4_egress.sql - layer 4, what left the perimeter (hashes and scan results only)
-- Run as lab_admin after some agent runs.
\set ECHO queries
SELECT egress_id, ts::time(0), purpose, bank_id, filtering, payload_chars, hits, hit_categories, blocked
FROM gov.egress_log ORDER BY egress_id;

SELECT filtering, count(*) AS requests, sum(hits) AS total_hits, count(*) FILTER (WHERE blocked) AS blocked
FROM gov.egress_log GROUP BY filtering;
