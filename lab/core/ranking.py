"""
Ranking service for result merging and re-ranking strategies.
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """Container for ranked results."""
    id: Any
    content: str
    score: float
    rank: int
    metadata: Dict[str, Any] = None
    sources: List[str] = None  # Which searches contributed this result


class RankingService:
    """
    Service for ranking and merging search results.
    
    Provides multiple ranking strategies:
    - Reciprocal Rank Fusion (RRF)
    - Weighted linear combination
    - Score normalization
    - Custom ranking functions
    """
    
    def __init__(self, default_k: int = 60):
        """
        Initialize ranking service.
        
        Args:
            default_k: Default k parameter for RRF (typically 60)
        """
        self.default_k = default_k
    
    def reciprocal_rank_fusion(
        self,
        result_lists: List[List[Any]],
        k: Optional[int] = None,
        id_func: Optional[Callable] = None,
        score_func: Optional[Callable] = None
    ) -> List[RankedResult]:
        """
        Merge multiple result lists using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank_i)) for each list i
        
        Args:
            result_lists: List of result lists to merge
            k: RRF parameter (default: 60)
            id_func: Function to extract ID from result
            score_func: Function to extract score from result
            
        Returns:
            Merged and ranked results
        """
        k = k or self.default_k
        id_func = id_func or (lambda x: x.id if hasattr(x, 'id') else x.get('id'))
        score_func = score_func or (lambda x: x.score if hasattr(x, 'score') else x.get('score', 0))
        
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        result_map = {}
        source_map = defaultdict(list)
        
        for list_idx, results in enumerate(result_lists):
            for rank, result in enumerate(results, 1):
                result_id = id_func(result)
                rrf_scores[result_id] += 1.0 / (k + rank)
                result_map[result_id] = result
                source_map[result_id].append(f"list_{list_idx}")
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Create ranked results
        ranked_results = []
        for rank, result_id in enumerate(sorted_ids, 1):
            result = result_map[result_id]
            
            # Extract content based on result type
            if hasattr(result, 'content'):
                content = result.content
            elif isinstance(result, dict) and 'content' in result:
                content = result['content']
            else:
                content = str(result)
            
            # Extract metadata
            if hasattr(result, 'metadata'):
                metadata = result.metadata
            elif isinstance(result, dict):
                metadata = {k: v for k, v in result.items() if k not in ['id', 'content', 'score']}
            else:
                metadata = {}
            
            ranked_result = RankedResult(
                id=result_id,
                content=content,
                score=rrf_scores[result_id],
                rank=rank,
                metadata=metadata,
                sources=source_map[result_id]
            )
            ranked_results.append(ranked_result)
        
        return ranked_results
    
    def weighted_combination(
        self,
        result_lists: List[List[Any]],
        weights: List[float],
        normalize: bool = True,
        id_func: Optional[Callable] = None,
        score_func: Optional[Callable] = None
    ) -> List[RankedResult]:
        """
        Merge results using weighted linear combination of scores.
        
        Args:
            result_lists: List of result lists to merge
            weights: Weight for each result list
            normalize: Whether to normalize scores before combining
            id_func: Function to extract ID from result
            score_func: Function to extract score from result
            
        Returns:
            Merged and ranked results
        """
        if len(result_lists) != len(weights):
            raise ValueError("Number of result lists must match number of weights")
        
        # Normalize weights
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 0.01:
            weights = [w / weight_sum for w in weights]
        
        id_func = id_func or (lambda x: x.id if hasattr(x, 'id') else x.get('id'))
        score_func = score_func or (lambda x: x.score if hasattr(x, 'score') else x.get('score', 0))
        
        # Normalize scores if requested
        if normalize:
            normalized_lists = []
            for results in result_lists:
                if results:
                    scores = [score_func(r) for r in results]
                    min_score = min(scores)
                    max_score = max(scores)
                    if max_score > min_score:
                        norm_factor = 1.0 / (max_score - min_score)
                        normalized = [
                            (r, (score_func(r) - min_score) * norm_factor)
                            for r in results
                        ]
                    else:
                        normalized = [(r, 1.0) for r in results]
                    normalized_lists.append(normalized)
                else:
                    normalized_lists.append([])
        else:
            normalized_lists = [
                [(r, score_func(r)) for r in results]
                for results in result_lists
            ]
        
        # Calculate weighted scores
        weighted_scores = defaultdict(float)
        result_map = {}
        source_map = defaultdict(list)
        
        for list_idx, (results, weight) in enumerate(zip(normalized_lists, weights)):
            for result, score in results:
                result_id = id_func(result)
                weighted_scores[result_id] += score * weight
                result_map[result_id] = result
                source_map[result_id].append(f"list_{list_idx}")
        
        # Sort by weighted score
        sorted_ids = sorted(weighted_scores.keys(), key=lambda x: weighted_scores[x], reverse=True)
        
        # Create ranked results
        ranked_results = []
        for rank, result_id in enumerate(sorted_ids, 1):
            result = result_map[result_id]
            
            # Extract content
            if hasattr(result, 'content'):
                content = result.content
            elif isinstance(result, dict) and 'content' in result:
                content = result['content']
            else:
                content = str(result)
            
            # Extract metadata
            if hasattr(result, 'metadata'):
                metadata = result.metadata
            elif isinstance(result, dict):
                metadata = {k: v for k, v in result.items() if k not in ['id', 'content', 'score']}
            else:
                metadata = {}
            
            ranked_result = RankedResult(
                id=result_id,
                content=content,
                score=weighted_scores[result_id],
                rank=rank,
                metadata=metadata,
                sources=source_map[result_id]
            )
            ranked_results.append(ranked_result)
        
        return ranked_results
    
    def normalize_scores(
        self,
        results: List[Any],
        method: str = 'minmax',
        score_func: Optional[Callable] = None
    ) -> List[Tuple[Any, float]]:
        """
        Normalize scores in a result list.
        
        Args:
            results: List of results to normalize
            method: Normalization method ('minmax', 'zscore', 'sigmoid')
            score_func: Function to extract score from result
            
        Returns:
            List of (result, normalized_score) tuples
        """
        if not results:
            return []
        
        score_func = score_func or (lambda x: x.score if hasattr(x, 'score') else x.get('score', 0))
        scores = np.array([score_func(r) for r in results])
        
        if method == 'minmax':
            min_score = scores.min()
            max_score = scores.max()
            if max_score > min_score:
                normalized = (scores - min_score) / (max_score - min_score)
            else:
                normalized = np.ones_like(scores)
        
        elif method == 'zscore':
            mean = scores.mean()
            std = scores.std()
            if std > 0:
                normalized = (scores - mean) / std
                # Convert to 0-1 range using sigmoid
                normalized = 1 / (1 + np.exp(-normalized))
            else:
                normalized = np.ones_like(scores) * 0.5
        
        elif method == 'sigmoid':
            # Direct sigmoid normalization
            normalized = 1 / (1 + np.exp(-scores))
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return list(zip(results, normalized))
    
    def deduplicate_results(
        self,
        results: List[RankedResult],
        similarity_threshold: float = 0.9,
        content_func: Optional[Callable] = None
    ) -> List[RankedResult]:
        """
        Remove duplicate or near-duplicate results.
        
        Args:
            results: List of ranked results
            similarity_threshold: Threshold for considering results duplicates
            content_func: Function to extract content for comparison
            
        Returns:
            Deduplicated results
        """
        if not results:
            return []
        
        content_func = content_func or (lambda x: x.content)
        
        # Simple deduplication based on ID
        seen_ids = set()
        deduplicated = []
        
        for result in results:
            if result.id not in seen_ids:
                deduplicated.append(result)
                seen_ids.add(result.id)
        
        # Could add more sophisticated content-based deduplication here
        # using similarity metrics like Jaccard, cosine similarity, etc.
        
        return deduplicated
    
    def filter_by_score(
        self,
        results: List[RankedResult],
        min_score: float = 0.0,
        max_score: Optional[float] = None
    ) -> List[RankedResult]:
        """
        Filter results by score range.
        
        Args:
            results: List of ranked results
            min_score: Minimum score threshold
            max_score: Maximum score threshold
            
        Returns:
            Filtered results
        """
        filtered = []
        for result in results:
            if result.score >= min_score:
                if max_score is None or result.score <= max_score:
                    filtered.append(result)
        
        return filtered
    
    def rerank_by_metadata(
        self,
        results: List[RankedResult],
        metadata_key: str,
        preferred_values: List[Any],
        boost_factor: float = 1.5
    ) -> List[RankedResult]:
        """
        Rerank results based on metadata preferences.
        
        Args:
            results: List of ranked results
            metadata_key: Metadata key to check
            preferred_values: Preferred values for the metadata
            boost_factor: Score multiplication factor for preferred results
            
        Returns:
            Reranked results
        """
        # Adjust scores based on metadata
        for result in results:
            if result.metadata and metadata_key in result.metadata:
                if result.metadata[metadata_key] in preferred_values:
                    result.score *= boost_factor
        
        # Re-sort by adjusted scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for rank, result in enumerate(results, 1):
            result.rank = rank
        
        return results
    
    def get_top_k(
        self,
        results: List[RankedResult],
        k: int
    ) -> List[RankedResult]:
        """
        Get top k results.
        
        Args:
            results: List of ranked results
            k: Number of results to return
            
        Returns:
            Top k results
        """
        return results[:k]
    
    def combine_with_fallback(
        self,
        primary_results: List[RankedResult],
        fallback_results: List[RankedResult],
        min_primary: int = 3,
        total_k: int = 10
    ) -> List[RankedResult]:
        """
        Combine primary results with fallback if insufficient primary results.
        
        Args:
            primary_results: Primary result list
            fallback_results: Fallback result list
            min_primary: Minimum number of primary results needed
            total_k: Total number of results to return
            
        Returns:
            Combined results
        """
        if len(primary_results) >= min_primary:
            return self.get_top_k(primary_results, total_k)
        
        # Combine primary with fallback
        combined = list(primary_results)
        seen_ids = {r.id for r in primary_results}
        
        for result in fallback_results:
            if result.id not in seen_ids:
                combined.append(result)
                seen_ids.add(result.id)
                if len(combined) >= total_k:
                    break
        
        return combined