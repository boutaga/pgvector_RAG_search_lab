-- 6_quality.sql - relevance kept? the latest measurement per state, method and question set
-- Run as lab_admin after python/measure.py.
\set ECHO queries
SELECT state, method, question_set, k, recall, ndcg, n_questions
FROM gov.quality_runs
WHERE run_id > (SELECT coalesce(max(run_id), 0) - 144 FROM gov.quality_runs)
  AND k IN (5, 10) AND question_set <> 'all'
ORDER BY question_set, method, k, state;
