-- =============================================================================
-- Script to find Wikipedia article IDs for test cases
-- =============================================================================
-- Run this on your Wikipedia database to find relevant article IDs
-- Then update test_cases_demo.json with the actual IDs
-- =============================================================================

-- Query 1: "What is machine learning?" (conceptual)
-- Looking for: Machine Learning article and related AI articles
SELECT id, title, length(content) as content_len
FROM articles
WHERE LOWER(title) LIKE '%machine learning%'
   OR LOWER(title) LIKE '%artificial intelligence%'
   OR LOWER(title) LIKE '%supervised learning%'
   OR LOWER(title) LIKE '%deep learning%'
ORDER BY
    CASE WHEN LOWER(title) = 'machine learning' THEN 0 ELSE 1 END,
    title
LIMIT 10;

-- Query 2: "Who invented the telephone?" (factual)
SELECT id, title, length(content) as content_len
FROM articles
WHERE LOWER(title) LIKE '%alexander graham bell%'
   OR LOWER(title) LIKE '%telephone%'
   OR LOWER(title) LIKE '%elisha gray%'
ORDER BY
    CASE WHEN LOWER(title) LIKE '%bell%' THEN 0 ELSE 1 END,
    title
LIMIT 10;

-- Query 3: "neural networks deep learning" (conceptual - title should help)
SELECT id, title, length(content) as content_len
FROM articles
WHERE LOWER(title) LIKE '%neural network%'
   OR LOWER(title) LIKE '%deep learning%'
   OR LOWER(title) LIKE '%artificial neural%'
   OR LOWER(title) LIKE '%perceptron%'
ORDER BY
    CASE WHEN LOWER(title) LIKE '%neural network%' THEN 0 ELSE 1 END,
    title
LIMIT 10;

-- Query 4: "whale marine mammal" (factual - specific entity)
SELECT id, title, length(content) as content_len
FROM articles
WHERE LOWER(title) LIKE '%whale%'
   OR LOWER(title) LIKE '%cetacea%'
   OR LOWER(title) LIKE '%blue whale%'
   OR LOWER(title) LIKE '%orca%'
   OR LOWER(title) LIKE '%dolphin%'
ORDER BY
    CASE WHEN LOWER(title) = 'whale' THEN 0 ELSE 1 END,
    title
LIMIT 10;

-- =============================================================================
-- NEW TEST CASES - Better for demonstrating strategy differences
-- =============================================================================

-- Query 5: "quantum mechanics physics" (conceptual - abstract topic)
-- Title-weighted should find "Quantum mechanics" article faster
SELECT id, title, length(content) as content_len
FROM articles
WHERE LOWER(title) LIKE '%quantum%'
   OR LOWER(title) LIKE '%heisenberg%'
   OR LOWER(title) LIKE '%schrödinger%'
   OR LOWER(title) LIKE '%wave function%'
ORDER BY
    CASE WHEN LOWER(title) = 'quantum mechanics' THEN 0
         WHEN LOWER(title) LIKE '%quantum%' THEN 1
         ELSE 2 END,
    title
LIMIT 15;

-- Query 6: "Napoleon Bonaparte French emperor" (factual - historical figure)
-- Should prioritize the Napoleon article over French Revolution articles
SELECT id, title, length(content) as content_len
FROM articles
WHERE LOWER(title) LIKE '%napoleon%'
   OR LOWER(title) LIKE '%french revolution%'
   OR LOWER(title) LIKE '%waterloo%'
   OR LOWER(title) LIKE '%josephine%'
ORDER BY
    CASE WHEN LOWER(title) LIKE '%napoleon%' THEN 0 ELSE 1 END,
    title
LIMIT 15;

-- Query 7: "climate change global warming" (exploratory)
-- Multiple related articles should be found
SELECT id, title, length(content) as content_len
FROM articles
WHERE LOWER(title) LIKE '%climate change%'
   OR LOWER(title) LIKE '%global warming%'
   OR LOWER(title) LIKE '%greenhouse%'
   OR LOWER(title) LIKE '%carbon dioxide%'
   OR LOWER(title) LIKE '%kyoto protocol%'
ORDER BY
    CASE WHEN LOWER(title) LIKE '%climate change%' THEN 0
         WHEN LOWER(title) LIKE '%global warming%' THEN 1
         ELSE 2 END,
    title
LIMIT 15;

-- Query 8: "Albert Einstein relativity theory" (factual - person + concept)
-- Should find both Einstein biography and relativity articles
SELECT id, title, length(content) as content_len
FROM articles
WHERE LOWER(title) LIKE '%einstein%'
   OR LOWER(title) LIKE '%relativity%'
   OR LOWER(title) LIKE '%special relativity%'
   OR LOWER(title) LIKE '%general relativity%'
ORDER BY
    CASE WHEN LOWER(title) LIKE '%einstein%' THEN 0
         WHEN LOWER(title) LIKE '%relativity%' THEN 1
         ELSE 2 END,
    title
LIMIT 15;

-- Query 9: "DNA genetics biology" (conceptual - scientific topic)
-- Multiple overlapping articles about genetics
SELECT id, title, length(content) as content_len
FROM articles
WHERE LOWER(title) LIKE '%dna%'
   OR LOWER(title) LIKE '%genetics%'
   OR LOWER(title) LIKE '%gene%'
   OR LOWER(title) LIKE '%chromosome%'
   OR LOWER(title) LIKE '%genome%'
   OR LOWER(title) = 'dna'
ORDER BY
    CASE WHEN LOWER(title) = 'dna' THEN 0
         WHEN LOWER(title) = 'genetics' THEN 1
         ELSE 2 END,
    title
LIMIT 15;

-- Query 10: "World War II history" (exploratory - broad historical topic)
-- Many related articles, title matching important
SELECT id, title, length(content) as content_len
FROM articles
WHERE LOWER(title) LIKE '%world war ii%'
   OR LOWER(title) LIKE '%second world war%'
   OR LOWER(title) LIKE '%d-day%'
   OR LOWER(title) LIKE '%pearl harbor%'
   OR LOWER(title) LIKE '%holocaust%'
   OR LOWER(title) LIKE '%normandy%'
ORDER BY
    CASE WHEN LOWER(title) LIKE '%world war ii%' THEN 0
         WHEN LOWER(title) LIKE '%second world war%' THEN 0
         ELSE 1 END,
    title
LIMIT 15;

-- Query 11: "Python programming language" (factual - technology)
-- Should distinguish programming language from the snake
SELECT id, title, length(content) as content_len
FROM articles
WHERE LOWER(title) LIKE '%python%'
ORDER BY
    CASE WHEN LOWER(title) LIKE '%programming%' THEN 0
         WHEN LOWER(title) LIKE '%language%' THEN 1
         ELSE 2 END,
    title
LIMIT 15;

-- Query 12: "Renaissance art history" (exploratory - art/history)
SELECT id, title, length(content) as content_len
FROM articles
WHERE LOWER(title) LIKE '%renaissance%'
   OR LOWER(title) LIKE '%michelangelo%'
   OR LOWER(title) LIKE '%leonardo da vinci%'
   OR LOWER(title) LIKE '%sistine chapel%'
ORDER BY
    CASE WHEN LOWER(title) = 'renaissance' THEN 0
         ELSE 1 END,
    title
LIMIT 15;

-- =============================================================================
-- VERIFICATION: Check if we have embedding coverage
-- =============================================================================

-- Check how many articles have both title and content vectors
SELECT
    COUNT(*) as total_articles,
    COUNT(title_vector_3072) as with_title_vector,
    COUNT(content_vector_3072) as with_content_vector,
    COUNT(content_sparse) as with_sparse_vector
FROM articles;

-- Check sample article to verify column names
SELECT id, title,
       title_vector_3072 IS NOT NULL as has_title_vec,
       content_vector_3072 IS NOT NULL as has_content_vec,
       content_sparse IS NOT NULL as has_sparse
FROM articles
LIMIT 5;
