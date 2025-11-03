#!/usr/bin/env python3
"""
Performance Benchmarking Suite

This module provides comprehensive performance benchmarking for RAG systems including:
- Throughput testing
- Latency profiling
- Scalability analysis
- Resource utilization monitoring
"""

import time
import json
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import DatabaseService
from core.search import VectorSearch, HybridSearch, AdaptiveSearch
from core.embeddings import OpenAIEmbedder


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    operation: str
    duration_ms: float
    throughput_ops: float
    cpu_percent: float
    memory_mb: float
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results"""
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    results: List[BenchmarkResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result"""
        self.results.append(result)
    
    def calculate_summary(self):
        """Calculate summary statistics"""
        if not self.results:
            return
        
        durations = [r.duration_ms for r in self.results if r.success]
        throughputs = [r.throughput_ops for r in self.results if r.success]
        cpu_usage = [r.cpu_percent for r in self.results if r.success]
        memory_usage = [r.memory_mb for r in self.results if r.success]
        
        self.summary = {
            'total_operations': len(self.results),
            'successful_operations': len([r for r in self.results if r.success]),
            'failed_operations': len([r for r in self.results if not r.success]),
            'duration_ms': {
                'mean': np.mean(durations) if durations else 0,
                'median': np.median(durations) if durations else 0,
                'p95': np.percentile(durations, 95) if durations else 0,
                'p99': np.percentile(durations, 99) if durations else 0,
                'min': np.min(durations) if durations else 0,
                'max': np.max(durations) if durations else 0
            },
            'throughput_ops': {
                'mean': np.mean(throughputs) if throughputs else 0,
                'total': np.sum(throughputs) if throughputs else 0
            },
            'resource_usage': {
                'cpu_percent_mean': np.mean(cpu_usage) if cpu_usage else 0,
                'memory_mb_mean': np.mean(memory_usage) if memory_usage else 0,
                'memory_mb_peak': np.max(memory_usage) if memory_usage else 0
            }
        }


class ResourceMonitor:
    """Monitors system resource usage during benchmarks"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.samples = []
        self.monitor_thread = None
    
    def start(self):
        """Start monitoring resources"""
        self.monitoring = True
        self.samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if not self.samples:
            return {'cpu_percent': 0, 'memory_mb': 0}
        
        cpu_samples = [s['cpu'] for s in self.samples]
        memory_samples = [s['memory'] for s in self.samples]
        
        return {
            'cpu_percent': np.mean(cpu_samples),
            'memory_mb': np.mean(memory_samples)
        }
    
    def _monitor_loop(self):
        """Monitor loop running in separate thread"""
        while self.monitoring:
            try:
                self.samples.append({
                    'cpu': self.process.cpu_percent(),
                    'memory': self.process.memory_info().rss / 1024 / 1024  # MB
                })
                time.sleep(0.1)  # Sample every 100ms
            except:
                pass


class PerformanceBenchmark:
    """Main performance benchmarking framework"""

    def __init__(
        self,
        db_service: Optional[DatabaseService] = None,
        embedder: Optional[OpenAIEmbedder] = None
    ):
        self.db = db_service or DatabaseService()
        self.embedder = embedder or OpenAIEmbedder()
        self.resource_monitor = ResourceMonitor()

        # Note: search_methods initialization removed - requires specific configuration
        # Users should create search instances separately and pass to benchmark methods
        self.search_methods = {}
    
    def benchmark_search_latency(
        self,
        queries: List[str],
        search_method: str = 'hybrid',
        k: int = 10,
        warmup: int = 5
    ) -> BenchmarkSuite:
        """Benchmark search latency"""
        
        suite = BenchmarkSuite(
            name=f"search_latency_{search_method}",
            start_time=datetime.now()
        )
        
        searcher = self.search_methods[search_method]
        
        # Warmup runs
        print(f"Running {warmup} warmup queries...")
        for i in range(min(warmup, len(queries))):
            searcher.search(queries[i], table_name='articles', top_k=k)
        
        # Actual benchmark
        print(f"Benchmarking {len(queries)} queries...")
        for query in queries:
            self.resource_monitor.start()
            start_time = time.time()
            
            try:
                results = searcher.search(query, table_name='articles', top_k=k)
                duration_ms = (time.time() - start_time) * 1000
                
                resources = self.resource_monitor.stop()
                
                suite.add_result(BenchmarkResult(
                    operation=f"search_{search_method}",
                    duration_ms=duration_ms,
                    throughput_ops=1000.0 / duration_ms,  # ops per second
                    cpu_percent=resources['cpu_percent'],
                    memory_mb=resources['memory_mb'],
                    success=True,
                    metadata={'query': query, 'num_results': len(results)}
                ))
            except Exception as e:
                resources = self.resource_monitor.stop()
                suite.add_result(BenchmarkResult(
                    operation=f"search_{search_method}",
                    duration_ms=0,
                    throughput_ops=0,
                    cpu_percent=resources['cpu_percent'],
                    memory_mb=resources['memory_mb'],
                    success=False,
                    error=str(e)
                ))
        
        suite.end_time = datetime.now()
        suite.calculate_summary()
        return suite
    
    def benchmark_embedding_generation(
        self,
        texts: List[str],
        batch_size: int = 50
    ) -> BenchmarkSuite:
        """Benchmark embedding generation performance"""
        
        suite = BenchmarkSuite(
            name="embedding_generation",
            start_time=datetime.now()
        )
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            self.resource_monitor.start()
            start_time = time.time()
            
            try:
                embeddings = self.embedder.embed_batch(batch)
                duration_ms = (time.time() - start_time) * 1000
                
                resources = self.resource_monitor.stop()
                
                suite.add_result(BenchmarkResult(
                    operation="embed_batch",
                    duration_ms=duration_ms,
                    throughput_ops=(len(batch) * 1000.0) / duration_ms,
                    cpu_percent=resources['cpu_percent'],
                    memory_mb=resources['memory_mb'],
                    success=True,
                    metadata={'batch_size': len(batch)}
                ))
            except Exception as e:
                resources = self.resource_monitor.stop()
                suite.add_result(BenchmarkResult(
                    operation="embed_batch",
                    duration_ms=0,
                    throughput_ops=0,
                    cpu_percent=resources['cpu_percent'],
                    memory_mb=resources['memory_mb'],
                    success=False,
                    error=str(e)
                ))
        
        suite.end_time = datetime.now()
        suite.calculate_summary()
        return suite
    
    def benchmark_concurrent_load(
        self,
        queries: List[str],
        search_method: str = 'hybrid',
        num_workers: int = 10,
        duration_seconds: int = 60
    ) -> BenchmarkSuite:
        """Benchmark system under concurrent load"""
        
        suite = BenchmarkSuite(
            name=f"concurrent_load_{search_method}",
            start_time=datetime.now()
        )
        
        def worker(query_idx: int) -> BenchmarkResult:
            """Worker function for concurrent execution"""
            query = queries[query_idx % len(queries)]
            searcher = self.search_methods[search_method]
            
            start_time = time.time()
            try:
                results = searcher.search(query, table_name='articles', top_k=10)
                duration_ms = (time.time() - start_time) * 1000
                
                return BenchmarkResult(
                    operation=f"concurrent_search_{search_method}",
                    duration_ms=duration_ms,
                    throughput_ops=1000.0 / duration_ms,
                    cpu_percent=0,  # Not measured per-operation
                    memory_mb=0,     # Not measured per-operation
                    success=True,
                    metadata={'query': query, 'num_results': len(results)}
                )
            except Exception as e:
                return BenchmarkResult(
                    operation=f"concurrent_search_{search_method}",
                    duration_ms=0,
                    throughput_ops=0,
                    cpu_percent=0,
                    memory_mb=0,
                    success=False,
                    error=str(e)
                )
        
        # Start resource monitoring for entire test
        self.resource_monitor.start()
        
        # Run concurrent load test
        print(f"Running concurrent load test with {num_workers} workers for {duration_seconds}s...")
        
        end_time = time.time() + duration_seconds
        query_idx = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            while time.time() < end_time:
                # Submit new tasks
                while len(futures) < num_workers and time.time() < end_time:
                    future = executor.submit(worker, query_idx)
                    futures.append(future)
                    query_idx += 1
                
                # Collect completed tasks
                done = []
                for future in as_completed(futures, timeout=0.1):
                    try:
                        result = future.result()
                        suite.add_result(result)
                        done.append(future)
                    except:
                        pass
                
                # Remove completed futures
                for future in done:
                    futures.remove(future)
        
        # Stop resource monitoring
        resources = self.resource_monitor.stop()
        
        # Add resource usage to summary
        suite.end_time = datetime.now()
        suite.calculate_summary()
        suite.summary['resource_usage'].update(resources)
        
        return suite
    
    def benchmark_scalability(
        self,
        queries: List[str],
        search_method: str = 'hybrid',
        worker_counts: List[int] = None
    ) -> Dict[str, BenchmarkSuite]:
        """Benchmark scalability with different worker counts"""
        
        if worker_counts is None:
            worker_counts = [1, 2, 5, 10, 20]
        
        results = {}
        
        for num_workers in worker_counts:
            print(f"\nBenchmarking with {num_workers} workers...")
            suite = self.benchmark_concurrent_load(
                queries,
                search_method=search_method,
                num_workers=num_workers,
                duration_seconds=30
            )
            results[f"workers_{num_workers}"] = suite
        
        return results
    
    def run_full_benchmark(
        self,
        queries: List[str],
        texts: List[str] = None
    ) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'num_queries': len(queries),
                'num_texts': len(texts) if texts else 0
            },
            'benchmarks': {}
        }
        
        # Search latency benchmarks
        print("\n=== Search Latency Benchmarks ===")
        for method in self.search_methods:
            print(f"\nBenchmarking {method} search...")
            suite = self.benchmark_search_latency(queries, search_method=method)
            results['benchmarks'][f'latency_{method}'] = suite.summary
        
        # Embedding generation benchmark
        if texts:
            print("\n=== Embedding Generation Benchmark ===")
            suite = self.benchmark_embedding_generation(texts)
            results['benchmarks']['embedding_generation'] = suite.summary
        
        # Concurrent load benchmark
        print("\n=== Concurrent Load Benchmark ===")
        suite = self.benchmark_concurrent_load(queries, num_workers=10)
        results['benchmarks']['concurrent_load'] = suite.summary
        
        # Scalability benchmark
        print("\n=== Scalability Benchmark ===")
        scalability_results = self.benchmark_scalability(queries)
        results['benchmarks']['scalability'] = {
            name: suite.summary for name, suite in scalability_results.items()
        }
        
        return results


def main():
    """Main benchmark script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG System Performance Benchmark')
    parser.add_argument('--queries-file', help='File with test queries (one per line)')
    parser.add_argument('--texts-file', help='File with test texts for embedding')
    parser.add_argument('--num-queries', type=int, default=100,
                       help='Number of queries to use')
    parser.add_argument('--benchmark', choices=['latency', 'embedding', 'concurrent', 'scalability', 'full'],
                       default='full', help='Type of benchmark to run')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    # Load test data
    if args.queries_file:
        with open(args.queries_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()][:args.num_queries]
    else:
        # Generate sample queries
        queries = [
            f"What is {topic}?" for topic in 
            ['machine learning', 'artificial intelligence', 'deep learning',
             'neural networks', 'computer vision', 'natural language processing',
             'reinforcement learning', 'supervised learning', 'unsupervised learning',
             'transfer learning'] * 10
        ][:args.num_queries]
    
    texts = None
    if args.texts_file:
        with open(args.texts_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark()
    
    # Run selected benchmark
    if args.benchmark == 'latency':
        results = {}
        for method in ['vector', 'hybrid', 'adaptive']:
            suite = benchmark.benchmark_search_latency(queries, search_method=method)
            results[method] = suite.summary
    
    elif args.benchmark == 'embedding':
        if not texts:
            texts = queries  # Use queries as texts if not provided
        suite = benchmark.benchmark_embedding_generation(texts)
        results = suite.summary
    
    elif args.benchmark == 'concurrent':
        suite = benchmark.benchmark_concurrent_load(queries)
        results = suite.summary
    
    elif args.benchmark == 'scalability':
        scalability_results = benchmark.benchmark_scalability(queries)
        results = {name: suite.summary for name, suite in scalability_results.items()}
    
    else:  # full
        results = benchmark.run_full_benchmark(queries, texts)
    
    # Save or print results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
    else:
        print("\n" + "="*50)
        print("BENCHMARK RESULTS")
        print("="*50)
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()