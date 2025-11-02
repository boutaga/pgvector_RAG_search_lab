# RAG k-Balance Experiment

## Overview

This experiment helps you understand and optimize the trade-offs between two critical parameters in Retrieval-Augmented Generation (RAG) systems:

- **`k_retrieve`**: Number of candidate documents fetched from the vector database
- **`k_context`**: Number of top documents fed into the LLM after optional re-ranking

## Why This Matters

### The k_retrieve vs k_context Trade-off

**Higher k_retrieve:**
- ✅ Increases **Recall** (less likely to miss relevant documents)
- ✅ Improves **Evidence Sufficiency** (more comprehensive candidate pool)
- ❌ May decrease **Precision** (includes more marginal documents)
- ❌ Increases **Latency** (more database I/O and computation)

**Lower k_context:**
- ✅ Reduces **LLM Cost** (fewer tokens in prompt)
- ✅ May improve **Answer Quality** (less noise, more focused)
- ✅ Faster **LLM Response Time**
- ❌ May miss relevant context if filtering/re-ranking is poor

### Recommended Strategy

1. Use **high k_retrieve** (100-200) to build a large candidate pool
2. Apply optional **re-ranking** (e.g., cross-encoder) to refine results
3. Use **moderate k_context** (5-10) for final LLM input
4. Monitor metrics to find your optimal balance

## Prerequisites

### 1. Database Setup

Ensure you have a PostgreSQL database with pgvector extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Your table should have:
- An ID column (e.g., `id`)
- A vector embedding column (e.g., `content_vector_3072`)
- Content columns (e.g., `title`, `content`)

### 2. Environment Variables

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/dbname"
export OPENAI_API_KEY="sk-..."
```

### 3. Python Dependencies

```bash
pip install psycopg2-binary openai numpy tqdm
```

## Quick Start

### 1. Prepare Test Cases

Create a JSON file with test queries and ground truth document IDs:

```json
[
  {
    "query": "What is machine learning?",
    "expected_doc_ids": [1, 2, 3, 15, 22],
    "metadata": {
      "category": "conceptual",
      "difficulty": "easy"
    }
  },
  {
    "query": "Who invented the telephone?",
    "expected_doc_ids": [4, 5],
    "metadata": {
      "category": "factual",
      "difficulty": "easy"
    }
  }
]
```

A sample file is provided at `lab/evaluation/test_cases.json`.

### 2. Run Single Experiment

Test a single k configuration:

```bash
python lab/evaluation/examples/k_balance_experiment.py \
    --test-file lab/evaluation/test_cases.json \
    --k-retrieve 100 \
    --k-context 8 \
    --vector-column content_vector_3072
```

### 3. Compare Multiple k Values

Test multiple configurations to find the optimal balance:

```bash
python lab/evaluation/examples/k_balance_experiment.py \
    --test-file lab/evaluation/test_cases.json \
    --k-retrieve-values 50 100 200 \
    --k-context-values 5 8 10 \
    --output results.json
```

## Understanding the Metrics

The script computes several retrieval quality metrics:

### Precision@k
**What it measures:** Proportion of retrieved documents that are relevant

**Formula:** `relevant_retrieved / k`

**Interpretation:**
- 1.0 = All retrieved documents are relevant (perfect precision)
- 0.5 = Half of retrieved documents are relevant
- 0.0 = No relevant documents retrieved

**When to act:**
- Low precision → Add re-ranker or decrease k_context

### Recall@k
**What it measures:** Proportion of relevant documents that are retrieved

**Formula:** `relevant_retrieved / total_relevant`

**Interpretation:**
- 1.0 = All relevant documents were retrieved (perfect recall)
- 0.5 = Half of relevant documents were retrieved
- 0.0 = No relevant documents retrieved

**When to act:**
- Low recall → Increase k_retrieve or improve embedding quality

### F1@k
**What it measures:** Harmonic mean of precision and recall

**Formula:** `2 * (precision * recall) / (precision + recall)`

**Interpretation:**
- Balances precision and recall
- Good for overall quality assessment

### nDCG@k (Normalized Discounted Cumulative Gain)
**What it measures:** Quality of ranking (penalizes relevant docs appearing late)

**Interpretation:**
- 1.0 = Perfect ranking (all relevant docs at top)
- 0.5 = Moderate ranking quality
- 0.0 = Poor ranking

**When to act:**
- Low nDCG → Use hybrid search or adaptive search for better ranking

### MRR (Mean Reciprocal Rank)
**What it measures:** Position of first relevant document

**Formula:** `1 / rank_of_first_relevant`

**Interpretation:**
- 1.0 = First document is relevant
- 0.5 = Second document is relevant
- 0.1 = Tenth document is relevant

### Latency
**What it measures:** Time to retrieve documents (milliseconds)

**Interpretation:**
- Monitor for performance optimization
- Consider database indexes if latency is high

### Context Tokens
**What it measures:** Approximate token count for LLM input

**Interpretation:**
- Directly impacts LLM cost
- Typical models: ~$0.001-0.01 per 1K tokens

## Example Output

### Single Experiment Output

```
================================================================================
RAG k-Balance Experiment
================================================================================

Configuration:
  Database: postgresql://localhost/wikipedia
  Table: articles
  Vector Column: content_vector_3072
  Test File: test_cases.json
  k_retrieve values: [100]
  k_context values: [8]

✓ Loaded 10 test cases

Running experiments (k_retrieve=100, k_context=8)...
================================================================================

[1/10] Query: What is machine learning?...
  ⏱  Latency: 145.2 ms
  📊 Retrieved 100 documents
  ✓ Found 4/5 relevant docs
  📈 Precision@100: 0.040 | Recall@100: 0.800 | nDCG@100: 0.745
  🎯 Context size: 2847 tokens (top 8 docs)

...

✓ Completed 10 experiments
```

### Multiple Experiments Output

```
================================================================================
SUMMARY STATISTICS
================================================================================

Config               Precision    Recall       F1           nDCG         MRR          Latency      Ctx Tokens
----------------------------------------------------------------------------------------------------
k_r=50, k_c=5        0.035        0.612        0.066        0.651        0.723        89.4         1823
k_r=100, k_c=5       0.038        0.742        0.072        0.721        0.756        145.2        1845
k_r=200, k_c=5       0.041        0.823        0.078        0.768        0.782        267.8        1867
k_r=100, k_c=8       0.040        0.742        0.076        0.721        0.756        145.5        2847
k_r=200, k_c=10      0.042        0.823        0.080        0.768        0.782        268.1        3521
================================================================================

INTERPRETATION GUIDE
================================================================================

Key Insights:
------------
1. **Increasing k_retrieve**:
   - ✓ Improves Recall (finds more relevant documents)
   - ✓ Improves Evidence Sufficiency (less likely to miss important info)
   - ✗ May decrease Precision (more marginal documents included)
   - ✗ Increases latency (more computation)

2. **Decreasing k_context**:
   - ✓ Reduces LLM cost (fewer tokens in prompt)
   - ✓ May improve answer quality (less noise)
   - ✓ Faster LLM response
   - ✗ May miss relevant context if filtering is poor

3. **Optimal Strategy**:
   - Use high k_retrieve (100-200) to build large candidate pool
   - Use moderate k_context (5-10) after re-ranking
   - Consider implementing re-ranker (cross-encoder) between retrieve and context

4. **Metric Guidance**:
   - Low Recall → Increase k_retrieve
   - Good Recall but Low Precision → Add re-ranker, decrease k_context
   - High Latency → Optimize indexes, consider smaller k_context
   - Low nDCG → Improve ranking (use hybrid/adaptive search)
```

## Advanced Usage

### Using Different Vector Columns

If you have multiple embedding dimensions:

```bash
# Test with 1536-dimensional embeddings
python k_balance_experiment.py \
    --test-file test_cases.json \
    --k-retrieve 100 \
    --k-context 8 \
    --vector-column content_vector

# Test with 3072-dimensional embeddings (default)
python k_balance_experiment.py \
    --test-file test_cases.json \
    --k-retrieve 100 \
    --k-context 8 \
    --vector-column content_vector_3072
```

### Custom Database Schema

If your database uses different column names:

```bash
python k_balance_experiment.py \
    --test-file test_cases.json \
    --table my_documents \
    --id-column doc_id \
    --vector-column embedding \
    --content-columns document_text summary \
    --k-retrieve 100 \
    --k-context 8
```

### Quiet Mode (Minimal Output)

For automated testing or batch processing:

```bash
python k_balance_experiment.py \
    --test-file test_cases.json \
    --k-retrieve 100 \
    --k-context 8 \
    --output results.json \
    --quiet
```

## Interpreting Results for Different Query Types

### Factual Queries
- **Expected:** High MRR (answer is usually in 1-2 documents)
- **Expected:** Moderate recall needed (specific facts, not broad concepts)
- **Recommendation:** k_retrieve=50-100, k_context=3-5

### Conceptual Queries
- **Expected:** Lower MRR (concept explained across multiple docs)
- **Expected:** Higher recall needed (comprehensive explanation)
- **Recommendation:** k_retrieve=100-200, k_context=8-10

### Exploratory Queries
- **Expected:** Lower precision (broad topic)
- **Expected:** High recall critical (gather diverse perspectives)
- **Recommendation:** k_retrieve=200+, k_context=10-15, use re-ranker

## Next Steps

### 1. Implement Re-ranking
Add a cross-encoder between k_retrieve and k_context:

```python
from sentence_transformers import CrossEncoder

# After retrieval at k_retrieve
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = cross_encoder.predict([(query, doc.content) for doc in candidates])
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
final_context = [doc for doc, _ in reranked[:k_context]]
```

### 2. Try Hybrid Search
Combine dense and sparse vectors for better retrieval:

```bash
# Compare dense-only vs hybrid
python lab/search/hybrid_search.py --query "machine learning"
```

### 3. Evaluate End-to-End Quality
Measure final answer quality (not just retrieval):

```python
from lab.evaluation.evaluator import RAGEvaluator

evaluator = RAGEvaluator()
results = evaluator.run_evaluation(test_cases)
```

### 4. Cost Analysis
Calculate LLM costs based on context tokens:

```python
# For GPT-4 Turbo: $0.01 per 1K input tokens
avg_tokens = 2847  # From experiment results
cost_per_query = (avg_tokens / 1000) * 0.01
monthly_cost = cost_per_query * queries_per_month
```

## Troubleshooting

### Error: "Database URL must be provided"
**Solution:** Set the DATABASE_URL environment variable:
```bash
export DATABASE_URL="postgresql://user:password@host:port/database"
```

### Error: "Test file not found"
**Solution:** Use absolute path or path relative to current directory:
```bash
python lab/evaluation/examples/k_balance_experiment.py \
    --test-file /absolute/path/to/test_cases.json
```

### Error: "No module named 'lab'"
**Solution:** Run from repository root or add to PYTHONPATH:
```bash
cd /path/to/Movies_pgvector_lab
python -m lab.evaluation.examples.k_balance_experiment --test-file test_cases.json ...
```

### Low Recall Across All k Values
**Possible causes:**
1. Embedding model mismatch (query vs documents use different models)
2. Ground truth document IDs are incorrect
3. Documents don't actually contain relevant information

**Solutions:**
1. Verify same embedding model for queries and documents
2. Manually verify ground truth annotations
3. Check document content quality

### High Latency
**Possible causes:**
1. Missing vector indexes
2. Large k_retrieve value
3. Slow database connection

**Solutions:**
```sql
-- Create HNSW index for better performance
CREATE INDEX ON articles USING hnsw (content_vector_3072 vector_cosine_ops);

-- Or IVFFlat index
CREATE INDEX ON articles USING ivfflat (content_vector_3072 vector_cosine_ops)
WITH (lists = 100);
```

## References

- [Precision and Recall in Information Retrieval](https://en.wikipedia.org/wiki/Precision_and_recall)
- [nDCG Explanation](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
- [RAG Evaluation Best Practices](https://www.dbi-services.com/blog/rag-series-adaptive-rag-understanding-confidence-precision-ndcg/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)

## Contributing

To add new metrics or features:
1. Edit `lab/evaluation/evaluator.py` to add metric functions
2. Update `k_balance_experiment.py` to compute and display new metrics
3. Update this README with usage examples

## License

This is part of the Movies_pgvector_lab educational repository.
