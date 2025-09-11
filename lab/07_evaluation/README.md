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