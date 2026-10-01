-- 8_maturity.sql - where the database stands, level 0 to 5
-- Run as lab_admin (or gateway).
\set ECHO queries
SELECT level, check_name, passed FROM gov.maturity_checks ORDER BY level, check_name;
SELECT * FROM gov.maturity;
