#!/usr/bin/env python3
"""
RAG Evaluation Framework

This module provides comprehensive evaluation tools for RAG systems including:
- Retrieval quality metrics
- Answer quality assessment
- Performance benchmarking
- A/B testing framework
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import DatabaseService
from core.search import VectorSearch, HybridSearch, AdaptiveSearch
from core.generation import GenerationService
from core.config import Config


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    f1_score: float = 0.0
    mean_reciprocal_rank: float = 0.0
    normalized_dcg: float = 0.0
    answer_relevance: float = 0.0
    answer_faithfulness: float = 0.0
    latency_ms: float = 0.0
    token_cost: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            'precision_at_k': self.precision_at_k,
            'recall_at_k': self.recall_at_k,
            'f1_score': self.f1_score,
            'mean_reciprocal_rank': self.mean_reciprocal_rank,
            'normalized_dcg': self.normalized_dcg,
            'answer_relevance': self.answer_relevance,
            'answer_faithfulness': self.answer_faithfulness,
            'latency_ms': self.latency_ms,
            'token_cost': self.token_cost
        }


@dataclass
class TestCase:
    """Represents a single test case for evaluation"""
    query: str
    expected_doc_ids: List[int] = field(default_factory=list)
    expected_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetrievalEvaluator:
    """Evaluates retrieval quality metrics"""
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
        """Calculate precision@k"""
        retrieved_k = retrieved_ids[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_ids))
        return relevant_retrieved / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
        """Calculate recall@k"""
        retrieved_k = retrieved_ids[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_ids))
        return relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0
    
    @staticmethod
    def f1_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
        """Calculate F1 score@k"""
        precision = RetrievalEvaluator.precision_at_k(retrieved_ids, relevant_ids, k)
        recall = RetrievalEvaluator.recall_at_k(retrieved_ids, relevant_ids, k)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def dcg_at_k(relevance_scores: List[float], k: int) -> float:
        """Calculate Discounted Cumulative Gain@k"""
        relevance_scores = relevance_scores[:k]
        if not relevance_scores:
            return 0.0
        
        dcg = relevance_scores[0]
        for i, score in enumerate(relevance_scores[1:], 2):
            dcg += score / np.log2(i)
        return dcg
    
    @staticmethod
    def ndcg_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@k"""
        # Create relevance scores (1 for relevant, 0 for not)
        relevance_scores = [1.0 if doc_id in relevant_ids else 0.0 
                          for doc_id in retrieved_ids[:k]]
        
        dcg = RetrievalEvaluator.dcg_at_k(relevance_scores, k)
        
        # Calculate ideal DCG (all relevant docs at top)
        ideal_scores = [1.0] * min(len(relevant_ids), k)
        ideal_scores.extend([0.0] * (k - len(ideal_scores)))
        idcg = RetrievalEvaluator.dcg_at_k(ideal_scores, k)
        
        return dcg / idcg if idcg > 0 else 0.0


class AnswerEvaluator:
    """Evaluates answer quality using LLM-based metrics"""
    
    def __init__(self, generator: GenerationService):
        self.generator = generator
    
    def evaluate_relevance(self, query: str, answer: str) -> float:
        """Evaluate answer relevance to query (0-1 score)"""
        prompt = f"""Rate the relevance of this answer to the query on a scale of 0-10.

Query: {query}
Answer: {answer}

Provide only a number between 0 and 10."""

        try:
            response = self.generator.generate(prompt=prompt)
            score = float(response.content.strip()) / 10.0
            return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        except:
            return 0.5  # Default middle score on error
    
    def evaluate_faithfulness(self, context: str, answer: str) -> float:
        """Evaluate answer faithfulness to context (0-1 score)"""
        prompt = f"""Rate how faithful this answer is to the provided context on a scale of 0-10.
The answer should only contain information from the context.

Context: {context}
Answer: {answer}

Provide only a number between 0 and 10."""

        try:
            response = self.generator.generate(prompt=prompt)
            score = float(response.content.strip()) / 10.0
            return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        except:
            return 0.5  # Default middle score on error


class RAGEvaluator:
    """Main RAG evaluation framework"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.db = DatabaseService(self.config)
        self.generator = GenerationService()
        self.answer_evaluator = AnswerEvaluator(self.generator)
        
        # Initialize search methods
        self.search_methods = {
            'vector': VectorSearch(self.db, self.config),
            'hybrid': HybridSearch(self.db, self.config),
            'adaptive': AdaptiveSearch(self.db, self.config)
        }
    
    def evaluate_retrieval(
        self,
        test_case: TestCase,
        search_method: str,
        k: int = 10
    ) -> Dict[str, float]:
        """Evaluate retrieval performance for a test case"""
        
        # Perform search
        start_time = time.time()
        results = self.search_methods[search_method].search(
            test_case.query,
            table_name='articles',
            top_k=k
        )
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract retrieved IDs
        retrieved_ids = [r['id'] for r in results]
        
        # Calculate metrics
        metrics = {
            'precision_at_k': RetrievalEvaluator.precision_at_k(
                retrieved_ids, test_case.expected_doc_ids, k
            ),
            'recall_at_k': RetrievalEvaluator.recall_at_k(
                retrieved_ids, test_case.expected_doc_ids, k
            ),
            'f1_at_k': RetrievalEvaluator.f1_at_k(
                retrieved_ids, test_case.expected_doc_ids, k
            ),
            'mrr': RetrievalEvaluator.mean_reciprocal_rank(
                retrieved_ids, test_case.expected_doc_ids
            ),
            'ndcg_at_k': RetrievalEvaluator.ndcg_at_k(
                retrieved_ids, test_case.expected_doc_ids, k
            ),
            'latency_ms': latency_ms
        }
        
        return metrics
    
    def evaluate_generation(
        self,
        test_case: TestCase,
        search_method: str,
        k: int = 5
    ) -> Dict[str, float]:
        """Evaluate answer generation quality"""
        
        # Retrieve context
        results = self.search_methods[search_method].search(
            test_case.query,
            table_name='articles',
            top_k=k
        )
        
        # Generate answer
        context = "\n\n".join([r.get('content', '')[:500] for r in results])
        response = self.generator.generate(prompt=test_case.query, context=context)
        answer = response.content

        # Evaluate answer quality
        metrics = {
            'answer_relevance': self.answer_evaluator.evaluate_relevance(
                test_case.query, answer
            ),
            'answer_faithfulness': self.answer_evaluator.evaluate_faithfulness(
                context, answer
            ),
            'token_cost': response.cost
        }
        
        return metrics
    
    def run_evaluation(
        self,
        test_cases: List[TestCase],
        search_methods: List[str] = None,
        k_values: List[int] = None
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation across test cases"""
        
        if search_methods is None:
            search_methods = ['vector', 'hybrid', 'adaptive']
        if k_values is None:
            k_values = [5, 10, 20]
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'num_test_cases': len(test_cases),
            'search_methods': search_methods,
            'k_values': k_values,
            'detailed_results': [],
            'summary': {}
        }
        
        # Run evaluations
        for test_case in tqdm(test_cases, desc="Evaluating test cases"):
            case_results = {
                'query': test_case.query,
                'metadata': test_case.metadata,
                'results_by_method': {}
            }
            
            for method in search_methods:
                method_results = {}
                
                for k in k_values:
                    # Retrieval evaluation
                    retrieval_metrics = self.evaluate_retrieval(
                        test_case, method, k
                    )
                    
                    # Generation evaluation (only for smallest k)
                    if k == min(k_values):
                        generation_metrics = self.evaluate_generation(
                            test_case, method, k
                        )
                        retrieval_metrics.update(generation_metrics)
                    
                    method_results[f'k_{k}'] = retrieval_metrics
                
                case_results['results_by_method'][method] = method_results
            
            results['detailed_results'].append(case_results)
        
        # Calculate summary statistics
        results['summary'] = self._calculate_summary(results['detailed_results'])
        
        return results
    
    def _calculate_summary(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics across all test cases"""
        summary = {}
        
        # Aggregate metrics by method and k
        for result in detailed_results:
            for method, method_results in result['results_by_method'].items():
                if method not in summary:
                    summary[method] = {}
                
                for k_key, metrics in method_results.items():
                    if k_key not in summary[method]:
                        summary[method][k_key] = {
                            'metrics': [],
                            'mean': {},
                            'std': {}
                        }
                    
                    summary[method][k_key]['metrics'].append(metrics)
        
        # Calculate mean and std for each metric
        for method in summary:
            for k_key in summary[method]:
                all_metrics = summary[method][k_key]['metrics']
                
                # Get all metric names
                metric_names = all_metrics[0].keys() if all_metrics else []
                
                for metric_name in metric_names:
                    values = [m[metric_name] for m in all_metrics 
                             if metric_name in m]
                    if values:
                        summary[method][k_key]['mean'][metric_name] = np.mean(values)
                        summary[method][k_key]['std'][metric_name] = np.std(values)
                
                # Remove raw metrics to save space
                del summary[method][k_key]['metrics']
        
        return summary
    
    def compare_methods(
        self,
        test_cases: List[TestCase],
        baseline_method: str = 'vector',
        k: int = 10
    ) -> Dict[str, Any]:
        """Compare different search methods against a baseline"""
        
        comparison = {
            'baseline': baseline_method,
            'k': k,
            'results': {}
        }
        
        # Get baseline performance
        baseline_metrics = []
        for test_case in tqdm(test_cases, desc=f"Evaluating {baseline_method}"):
            metrics = self.evaluate_retrieval(test_case, baseline_method, k)
            baseline_metrics.append(metrics)
        
        comparison['results'][baseline_method] = {
            'mean_precision': np.mean([m['precision_at_k'] for m in baseline_metrics]),
            'mean_recall': np.mean([m['recall_at_k'] for m in baseline_metrics]),
            'mean_f1': np.mean([m['f1_at_k'] for m in baseline_metrics]),
            'mean_mrr': np.mean([m['mrr'] for m in baseline_metrics]),
            'mean_ndcg': np.mean([m['ndcg_at_k'] for m in baseline_metrics]),
            'mean_latency': np.mean([m['latency_ms'] for m in baseline_metrics])
        }
        
        # Compare other methods
        for method in self.search_methods:
            if method == baseline_method:
                continue
            
            method_metrics = []
            for test_case in tqdm(test_cases, desc=f"Evaluating {method}"):
                metrics = self.evaluate_retrieval(test_case, method, k)
                method_metrics.append(metrics)
            
            comparison['results'][method] = {
                'mean_precision': np.mean([m['precision_at_k'] for m in method_metrics]),
                'mean_recall': np.mean([m['recall_at_k'] for m in method_metrics]),
                'mean_f1': np.mean([m['f1_at_k'] for m in method_metrics]),
                'mean_mrr': np.mean([m['mrr'] for m in method_metrics]),
                'mean_ndcg': np.mean([m['ndcg_at_k'] for m in method_metrics]),
                'mean_latency': np.mean([m['latency_ms'] for m in method_metrics])
            }
            
            # Calculate improvement over baseline
            improvement = {}
            for metric in ['mean_precision', 'mean_recall', 'mean_f1', 'mean_mrr', 'mean_ndcg']:
                baseline_val = comparison['results'][baseline_method][metric]
                method_val = comparison['results'][method][metric]
                if baseline_val > 0:
                    improvement[metric] = ((method_val - baseline_val) / baseline_val) * 100
                else:
                    improvement[metric] = 0.0
            
            comparison['results'][method]['improvement_percent'] = improvement
        
        return comparison


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG System Evaluation')
    parser.add_argument('--test-file', help='JSON file with test cases')
    parser.add_argument('--methods', nargs='+', default=['vector', 'hybrid', 'adaptive'],
                       help='Search methods to evaluate')
    parser.add_argument('--k-values', nargs='+', type=int, default=[5, 10, 20],
                       help='k values for retrieval metrics')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison mode')
    
    args = parser.parse_args()
    
    # Load test cases
    if args.test_file:
        with open(args.test_file, 'r') as f:
            test_data = json.load(f)
            test_cases = [TestCase(**tc) for tc in test_data]
    else:
        # Default test cases
        test_cases = [
            TestCase(
                query="What is machine learning?",
                expected_doc_ids=[1, 2, 3],
                metadata={'category': 'conceptual'}
            ),
            TestCase(
                query="Who invented the telephone?",
                expected_doc_ids=[4, 5],
                metadata={'category': 'factual'}
            ),
            TestCase(
                query="Explain quantum computing applications",
                expected_doc_ids=[6, 7, 8, 9],
                metadata={'category': 'exploratory'}
            )
        ]
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    if args.compare:
        # Run comparison
        print("Running method comparison...")
        results = evaluator.compare_methods(test_cases, k=args.k_values[0])
    else:
        # Run full evaluation
        print("Running comprehensive evaluation...")
        results = evaluator.run_evaluation(
            test_cases,
            search_methods=args.methods,
            k_values=args.k_values
        )
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        if args.compare:
            for method, metrics in results['results'].items():
                print(f"\n{method.upper()}:")
                for metric, value in metrics.items():
                    if metric != 'improvement_percent':
                        print(f"  {metric}: {value:.4f}")
                if 'improvement_percent' in metrics:
                    print("  Improvement over baseline:")
                    for metric, value in metrics['improvement_percent'].items():
                        print(f"    {metric}: {value:+.2f}%")
        else:
            for method, method_results in results['summary'].items():
                print(f"\n{method.upper()}:")
                for k_key, metrics in method_results.items():
                    print(f"  {k_key}:")
                    for metric, value in metrics['mean'].items():
                        print(f"    {metric}: {value:.4f}")


if __name__ == "__main__":
    main()