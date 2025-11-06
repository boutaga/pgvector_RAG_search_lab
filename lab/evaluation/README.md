# RAG Evaluation Framework

This module provides comprehensive evaluation and benchmarking tools for RAG systems.

## Components

### 1. Evaluation Framework (`evaluator.py`)

Comprehensive evaluation of RAG system quality:

#### Retrieval Metrics
- **Precision@k**: Proportion of retrieved documents that are relevant
- **Recall@k**: Proportion of relevant documents that are retrieved
- **F1@k**: Harmonic mean of precision and recall
- **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks of first relevant result
- **NDCG@k**: Normalized Discounted Cumulative Gain for ranking quality

#### Answer Quality Metrics
- **Answer Relevance**: How well the answer addresses the query (LLM-evaluated)
- **Answer Faithfulness**: How well the answer is grounded in retrieved context
- **Token Cost**: Cost of generating the answer

### 2. Performance Benchmark (`benchmark.py`)

System performance and scalability testing:

#### Benchmark Types
- **Latency Testing**: Response time distribution (mean, median, p95, p99)
- **Throughput Testing**: Operations per second under load
- **Concurrent Load**: Performance under multiple simultaneous requests
- **Scalability Analysis**: Performance with varying worker counts
- **Resource Monitoring**: CPU and memory usage tracking

### 3. K-Balance Experiment (`examples/k_balance_experiment.py`)

Educational experiment for optimizing RAG retrieval parameters:

#### Purpose
Systematically explore the trade-offs between two critical RAG parameters:
- **k_retrieve**: Number of candidate documents fetched from vector database
- **k_context**: Number of documents fed into LLM after optional re-ranking

#### Key Trade-offs
**Increasing k_retrieve:**
- ✅ Improves Recall (finds more relevant documents)
- ✅ Better Evidence Sufficiency (comprehensive candidate pool)
- ❌ May decrease Precision (includes marginal documents)
- ❌ Increases Latency (more database I/O)

**Decreasing k_context:**
- ✅ Reduces LLM Cost (fewer tokens in prompt)
- ✅ May improve Answer Quality (less noise)
- ✅ Faster LLM Response Time
- ❌ May miss relevant context if filtering is poor

#### Metrics Computed
- **Precision@k**: Proportion of retrieved documents that are relevant
- **Recall@k**: Proportion of relevant documents retrieved
- **F1@k**: Harmonic mean of precision and recall
- **nDCG@k**: Ranking quality (penalizes relevant docs appearing late)
- **MRR**: Position of first relevant document
- **Latency**: Retrieval time in milliseconds
- **Context Tokens**: Approximate token count for LLM input (cost proxy)

#### Usage Examples

**Single Configuration:**
```bash
python lab/evaluation/examples/k_balance_experiment.py \
    --test-file lab/evaluation/test_cases.json \
    --k-retrieve 100 \
    --k-context 8 \
    --vector-column content_vector_3072
```

**Compare Multiple Configurations:**
```bash
python lab/evaluation/examples/k_balance_experiment.py \
    --test-file lab/evaluation/test_cases.json \
    --k-retrieve-values 50 100 200 \
    --k-context-values 5 8 10 \
    --output results.json
```

**Custom Database Schema:**
```bash
python lab/evaluation/examples/k_balance_experiment.py \
    --test-file test_cases.json \
    --table my_documents \
    --id-column doc_id \
    --vector-column embedding \
    --k-retrieve 100 \
    --k-context 8
```

### 4. Search Configuration Comparison (`examples/compare_search_configs.py`)

Comprehensive comparison of different search methods and configurations:

#### Supported Search Configurations
- **Vector Search**: Different k values (k=10, 50, 100, 200)
- **Sparse Search**: SPLADE-based keyword matching
- **Hybrid Search**: Dense + Sparse with configurable weights (50/50, 70/30, 30/70)
- **Adaptive Search**: Automatic weight adjustment based on query type

#### Metrics Computed
- **Precision@k**: Quality of retrieved documents
- **Recall@k**: Completeness of retrieval
- **F1@k**: Harmonic mean of precision and recall
- **nDCG@k**: Ranking quality (position-aware)
- **MRR**: Position of first relevant result
- **Latency**: Retrieval time in milliseconds

#### Usage
```bash
# Run all configurations
python lab/evaluation/examples/compare_search_configs.py \
    --test-file lab/evaluation/test_cases_expanded.json \
    --detailed \
    --output results.json

# Skip sparse/hybrid (vector only)
python lab/evaluation/examples/compare_search_configs.py \
    --test-file lab/evaluation/test_cases_expanded.json \
    --skip-sparse
```

### 5. Ranking Improvement Demo (`examples/demo_ranking_improvement.py`)

Live demonstration comparing content-only vs title-weighted search:

#### Purpose
Show measurable ranking quality improvements for conference presentations and teaching.

#### What It Demonstrates
- **Before**: Content-only vector search
- **After**: 70% title + 30% content weighted search
- **Real Results**: +6-25% recall, +8-49% nDCG improvement

#### Usage
```bash
# Run live demo (takes ~10 seconds)
python lab/evaluation/examples/demo_ranking_improvement.py
```

#### Expected Output
- Query-by-query comparison tables
- Metrics: Recall, Precision, nDCG, MRR
- Overall summary with improvements
- Perfect for live presentations!

### 6. Educational Documentation (`examples/*.md`)

Comprehensive guides explaining RAG evaluation concepts:

#### Available Guides
1. **UNDERSTANDING_K.md** - What is k? (retrieval pool size)
2. **RECALL_VS_PRECISION.md** - Complete explanation with analogies
3. **UNDERSTANDING_NDCG.md** - Deep dive into ranking quality metrics
4. **PRESENTATION_GUIDE.md** - Complete demo presentation guide with Q&A
5. **IMPROVING_RANKING_DEMO.md** - 5 optimization strategies with code
6. **OPTIMIZATION_QUICK_REF.md** - Quick reference cheat sheet
7. **DEMO_SCRIPT_EXPLAINED.md** - How the demo works
8. **COMBINED_VS_SEPARATE_EMBEDDINGS.md** - Embedding strategies

#### Key Features
- Fishing analogies for non-technical audiences
- Step-by-step calculations
- Real examples from actual data
- Quiz questions to test understanding
- Conference presentation scripts

#### Test Case Format
Test cases require queries with ground truth document IDs:

```json
[
  {
    "query": "What is machine learning?",
    "expected_doc_ids": [1, 2, 3, 15, 22],
    "metadata": {
      "category": "conceptual",
      "difficulty": "easy"
    }
  }
]
```

#### Interpreting Results

**Low Recall** → Increase k_retrieve to expand candidate pool

**Good Recall, Low Precision** → Add re-ranker, decrease k_context

**High Latency** → Optimize indexes, reduce k_context

**Low nDCG** → Use hybrid/adaptive search for better ranking

#### Recommended Strategy
1. Use **high k_retrieve** (100-200) to build large candidate pool
2. Apply optional **re-ranking** (e.g., cross-encoder) to refine results
3. Use **moderate k_context** (5-10) for final LLM input
4. Monitor metrics to find optimal balance for your use case

#### Output Example
```
================================================================================
SUMMARY STATISTICS
================================================================================

Config               Precision    Recall       F1           nDCG         MRR
----------------------------------------------------------------------------------------------------
k_r=50, k_c=5        0.035        0.612        0.066        0.651        0.723
k_r=100, k_c=5       0.038        0.742        0.072        0.721        0.756
k_r=200, k_c=5       0.041        0.823        0.078        0.768        0.782
k_r=100, k_c=8       0.040        0.742        0.076        0.721        0.756
```

**Full Documentation**: See `lab/evaluation/examples/README_K_BALANCE.md` for comprehensive guide including:
- Setup instructions
- Detailed metric explanations
- Query type recommendations
- Troubleshooting guide
- Next steps (re-ranking, hybrid search, cost analysis)

## Quick Start

### Running Evaluations

#### Basic Evaluation
```python
from evaluator import RAGEvaluator, TestCase

# Create test cases
test_cases = [
    TestCase(
        query="What is machine learning?",
        expected_doc_ids=[1, 2, 3],  # Ground truth document IDs
        metadata={'category': 'conceptual'}
    ),
    TestCase(
        query="Who invented the telephone?",
        expected_doc_ids=[4, 5],
        metadata={'category': 'factual'}
    )
]

# Run evaluation
evaluator = RAGEvaluator()
results = evaluator.run_evaluation(
    test_cases,
    search_methods=['vector', 'hybrid', 'adaptive'],
    k_values=[5, 10, 20]
)
```

#### Command Line Usage
```bash
# Run evaluation with test file
python evaluator.py --test-file test_cases.json --methods hybrid adaptive --k-values 10 20

# Run comparison between methods
python evaluator.py --compare --test-file test_cases.json --output results.json
```

### Running Benchmarks

#### Basic Benchmark
```python
from benchmark import PerformanceBenchmark

# Initialize benchmark
benchmark = PerformanceBenchmark()

# Run latency benchmark
queries = ["What is AI?", "Explain machine learning", "How do neural networks work?"]
suite = benchmark.benchmark_search_latency(queries, search_method='hybrid')
print(suite.summary)
```

#### Command Line Usage
```bash
# Run full benchmark suite
python benchmark.py --num-queries 100 --benchmark full --output benchmark_results.json

# Run specific benchmark
python benchmark.py --benchmark latency --queries-file queries.txt

# Run scalability test
python benchmark.py --benchmark scalability --num-queries 50
```

## Test Case Format

Test cases should be in JSON format:

```json
[
  {
    "query": "What is machine learning?",
    "expected_doc_ids": [1, 2, 3],
    "expected_answer": "Machine learning is...",
    "metadata": {
      "category": "conceptual",
      "difficulty": "easy"
    }
  },
  {
    "query": "Who invented the telephone?",
    "expected_doc_ids": [4, 5],
    "metadata": {
      "category": "factual"
    }
  }
]
```

## Metrics Explained

### Retrieval Metrics

1. **Precision@k = |Relevant ∩ Retrieved| / k**
   - Measures accuracy of retrieved results
   - Higher is better (0-1 scale)

2. **Recall@k = |Relevant ∩ Retrieved| / |Relevant|**
   - Measures completeness of retrieval
   - Higher is better (0-1 scale)

3. **F1@k = 2 × (Precision × Recall) / (Precision + Recall)**
   - Balanced metric between precision and recall
   - Higher is better (0-1 scale)

4. **MRR = 1/rank_of_first_relevant**
   - Measures how quickly relevant results appear
   - Higher is better (0-1 scale)

5. **NDCG@k**
   - Measures ranking quality with graded relevance
   - Higher is better (0-1 scale)

### Performance Metrics

1. **Latency Percentiles**
   - p50 (median): Typical response time
   - p95: Response time for 95% of requests
   - p99: Response time for 99% of requests

2. **Throughput**
   - Operations per second
   - Higher is better

3. **Resource Usage**
   - CPU utilization (%)
   - Memory consumption (MB)

## A/B Testing

Compare different search methods:

```python
# Compare methods against baseline
comparison = evaluator.compare_methods(
    test_cases,
    baseline_method='vector',
    k=10
)

# Results show improvement percentages
print(f"Hybrid improvement: {comparison['results']['hybrid']['improvement_percent']}")
```

## Integration with Main System

The evaluation framework integrates with the core system:

```python
from lab.core.config import Config
from lab.core.database import DatabaseService
from lab.07_evaluation.evaluator import RAGEvaluator

# Use custom configuration
config = Config()
config.hybrid_alpha = 0.7  # Test different weights

evaluator = RAGEvaluator(config)
results = evaluator.run_evaluation(test_cases)
```

## Best Practices

1. **Ground Truth Creation**
   - Use expert annotations for expected documents
   - Include diverse query types (factual, conceptual, exploratory)
   - Balance dataset across different categories

2. **Benchmark Design**
   - Warm up system before benchmarking
   - Use representative query distribution
   - Test under realistic load conditions
   - Monitor resource usage during tests

3. **Evaluation Strategy**
   - Evaluate multiple k values (5, 10, 20)
   - Test all search methods for comparison
   - Include both retrieval and generation metrics
   - Track performance over time

4. **Result Analysis**
   - Look for consistent patterns across metrics
   - Consider trade-offs (accuracy vs speed)
   - Identify failure modes and edge cases
   - Use results to tune system parameters

## Output Examples

### Evaluation Output
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "summary": {
    "hybrid": {
      "k_10": {
        "mean": {
          "precision_at_k": 0.75,
          "recall_at_k": 0.82,
          "f1_at_k": 0.78,
          "mrr": 0.89,
          "ndcg_at_k": 0.85,
          "latency_ms": 95.3
        }
      }
    }
  }
}
```

### Benchmark Output
```json
{
  "benchmarks": {
    "latency_hybrid": {
      "duration_ms": {
        "mean": 95.3,
        "median": 92.1,
        "p95": 145.2,
        "p99": 201.5
      },
      "throughput_ops": {
        "mean": 10.5
      }
    }
  }
}
```

## Troubleshooting

1. **Slow Evaluation**: Reduce batch size or k values
2. **Memory Issues**: Process test cases in smaller batches
3. **API Rate Limits**: Add delays between LLM calls
4. **Database Timeouts**: Check connection pool settings