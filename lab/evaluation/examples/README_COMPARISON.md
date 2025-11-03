# Search Configuration Comparison Guide

## Overview

The `compare_search_configs.py` script allows you to compare different search configurations side-by-side, including:
- ✅ Different k_retrieve values (10, 50, 100, 200)
- ✅ Different vector models (1536 vs 3072 dimensions)
- ✅ Different search methods (Vector, Sparse, Hybrid)
- ✅ Different hybrid weight combinations

## Quick Start

### 1. Basic Comparison (Vector only)

```bash
python lab/evaluation/examples/compare_search_configs.py \
    --test-file lab/evaluation/test_dcc2.json
```

This will compare:
- Vector search with k=10, 50, 100, 200
- Different vector dimensions (1536 vs 3072)

### 2. Full Comparison (Including Sparse/Hybrid)

```bash
python lab/evaluation/examples/compare_search_configs.py \
    --test-file lab/evaluation/test_cases_expanded.json \
    --detailed
```

This includes:
- All vector configurations
- Sparse vector search
- Hybrid search (balanced, dense-heavy, sparse-heavy)

### 3. Skip Sparse Search (if not available)

```bash
python lab/evaluation/examples/compare_search_configs.py \
    --test-file lab/evaluation/test_dcc2.json \
    --skip-sparse
```

### 4. Save Results

```bash
python lab/evaluation/examples/compare_search_configs.py \
    --test-file lab/evaluation/test_dcc2.json \
    --output comparison_results.json \
    --output-csv comparison_results.csv
```

## Understanding the Output

### Comparison Table

```
╒════════════════════╤═══════════╤═════════╤══════════╤═════════════╤════════╤════════╤═══════╤═════════╤═════════╕
│ Configuration      │ Type      │ k_r/k_c │ Recall   │ Precision   │ F1     │ nDCG   │ MRR    │ Latency │ Found   │
╞════════════════════╪═══════════╪═════════╪══════════╪═════════════╪════════╪════════╪═══════╪═════════╪═════════╡
│ Vec-3072-k100      │ vector    │ 100/8   │ 1.000    │ 0.040       │ 0.077  │ 0.890  │ 0.750  │ 420     │ 6/6     │
│ Hybrid-Balanced    │ hybrid    │ 100/8   │ 1.000    │ 0.040       │ 0.077  │ 0.920  │ 0.833  │ 650     │ 6/6     │
│ Vec-3072-k50       │ vector    │ 50/8    │ 0.833    │ 0.067       │ 0.124  │ 0.850  │ 0.750  │ 280     │ 5/6     │
└────────────────────┴───────────┴─────────┴──────────┴─────────────┴────────┴────────┴───────┴─────────┴─────────┘
```

**Columns:**
- **Configuration**: Name of the configuration
- **Type**: Search method (vector, sparse, hybrid)
- **k_r/k_c**: k_retrieve / k_context
- **Recall**: % of relevant docs found (higher is better)
- **Precision**: % of retrieved docs that are relevant
- **F1**: Harmonic mean of precision and recall
- **nDCG**: Ranking quality (1.0 = perfect, higher is better)
- **MRR**: Mean reciprocal rank (position of first relevant doc)
- **Latency**: Average query time in milliseconds
- **Found**: Relevant docs found / Total relevant docs

### Recommendations Section

The script automatically identifies:
- 🏆 **Best Recall**: Which config finds the most relevant documents
- 🏆 **Best Ranking (nDCG)**: Which config ranks relevant docs highest
- ⚡ **Fastest**: Which config has lowest latency

## What Configurations Are Tested?

The script tests these configurations by default:

### Vector Search Variations
```python
Vec-3072-k10    → 3072-dim vectors, k_retrieve=10
Vec-3072-k50    → 3072-dim vectors, k_retrieve=50
Vec-3072-k100   → 3072-dim vectors, k_retrieve=100
Vec-3072-k200   → 3072-dim vectors, k_retrieve=200
Vec-1536-k100   → 1536-dim vectors, k_retrieve=100
```

### Sparse Search
```python
Sparse-k100     → SPLADE sparse vectors, k_retrieve=100
```

### Hybrid Search
```python
Hybrid-Balanced      → 50% dense + 50% sparse
Hybrid-DenseHeavy    → 70% dense + 30% sparse
Hybrid-SparseHeavy   → 30% dense + 70% sparse
```

## Typical Patterns You'll See

### 1. **k_retrieve Trade-off**
- ✅ Higher k (100, 200) → Better recall, finds more docs
- ❌ Higher k (100, 200) → Lower precision, more noise
- ❌ Higher k (100, 200) → Higher latency

### 2. **Vector Dimensions**
- 3072-dim usually slightly better nDCG than 1536-dim
- 3072-dim takes more storage and slightly higher latency

### 3. **Hybrid vs Pure Vector**
- Hybrid often has better nDCG (better ranking)
- Hybrid costs more (2 searches instead of 1)
- For factual queries: Sparse-heavy hybrid works well
- For conceptual queries: Dense-heavy hybrid works well

### 4. **Sparse vs Dense**
- Sparse: Better for exact keyword matching
- Dense: Better for semantic/conceptual queries
- Hybrid: Best of both worlds

## Advanced: Customizing Configurations

Edit `compare_search_configs.py` and modify the `configs` list:

```python
configs = [
    # Add your custom config
    SearchConfig(
        name="Custom-k150",
        search_type="vector",
        vector_column="content_vector_3072",
        k_retrieve=150,
        k_context=10
    ),

    # Custom hybrid weights
    SearchConfig(
        name="Hybrid-Custom",
        search_type="hybrid",
        k_retrieve=100,
        k_context=8,
        dense_weight=0.6,
        sparse_weight=0.4
    ),
]
```

## Example Output Interpretation

**Scenario:** You run the comparison and get:

```
Best Recall: Vec-3072-k200 (1.000)
Best Ranking (nDCG): Hybrid-Balanced (0.950)
Fastest: Vec-3072-k10 (150ms)
```

**Interpretation:**
1. **If recall is critical** (don't miss any docs): Use Vec-3072-k200
2. **If ranking quality matters** (relevant docs at top): Use Hybrid-Balanced
3. **If latency is critical** (need fast responses): Use Vec-3072-k10

**Recommendation:** Use **Hybrid-Balanced with k=100** as a good balance of all factors.

## Troubleshooting

### "Sparse embedder not available"

The sparse search requires SPLADE embeddings in your database. If you see this:
```
⚠️  Skipping Hybrid-Balanced - sparse embedder not available
```

Use `--skip-sparse` flag to only test vector search.

### "tabulate module not found"

Install it:
```bash
pip install tabulate
```

Or the script will fall back to simple text output.

## Tips for Creating Good Test Cases

1. **Use real queries** from your application
2. **Manually verify ground truth** - use SQL queries to find truly relevant docs
3. **Diverse query types** - mix factual, conceptual, and exploratory queries
4. **Sufficient test cases** - at least 10-20 queries for meaningful results

## Example Workflow

```bash
# 1. Create test cases with real ground truth
vi lab/evaluation/my_test_cases.json

# 2. Run comparison
python lab/evaluation/examples/compare_search_configs.py \
    --test-file lab/evaluation/my_test_cases.json \
    --output results.json \
    --detailed

# 3. Analyze results and pick best configuration

# 4. Re-run with focused configs
# (Edit compare_search_configs.py to test only top 3 configs)

# 5. Document your findings
cat results.json | jq '.aggregated'
```

## Integration with k_balance_experiment

The `k_balance_experiment.py` focuses on:
- Single search method
- Multiple k values
- Detailed per-query analysis

The `compare_search_configs.py` focuses on:
- Multiple search methods
- Fixed k values
- Cross-method comparison

Use both together for comprehensive evaluation!
