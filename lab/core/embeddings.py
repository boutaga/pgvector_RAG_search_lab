"""
Embedding service for generating dense and sparse embeddings.
"""

import os
import time
import logging
import gc
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService(ABC):
    """
    Abstract base class for embedding generation services.
    """
    
    @abstractmethod
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[Union[List[float], Dict[int, float]]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Optional batch size override
            
        Returns:
            List of embeddings (dense vectors or sparse dicts)
        """
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """
        Get the dimensionality of embeddings.
        
        Returns:
            Number of dimensions
        """
        pass
    
    @abstractmethod
    def get_embedding_type(self) -> str:
        """
        Get the type of embeddings produced.
        
        Returns:
            'dense' or 'sparse'
        """
        pass


class OpenAIEmbedder(EmbeddingService):
    """
    OpenAI embedding service for generating dense embeddings.
    
    Features:
    - Batch processing with configurable size
    - Exponential backoff retry logic
    - Rate limit handling
    - Support for multiple OpenAI embedding models
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        dimensions: int = 3072,
        batch_size: int = 50,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize OpenAI embedder.
        
        Args:
            model: OpenAI embedding model name
            dimensions: Embedding dimensions
            batch_size: Default batch size for processing
            max_retries: Maximum retry attempts for API calls
            retry_delay: Base delay between retries
        """
        try:
            from openai import OpenAI
            self.client = OpenAI()
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"Initialized OpenAI embedder with model: {model}")
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate OpenAI embeddings for texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Optional batch size override
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._generate_batch_with_retry(batch)
            all_embeddings.extend(embeddings)
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return all_embeddings
    
    def _generate_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch with retry logic.
        
        Args:
            texts: Batch of texts to embed
            
        Returns:
            List of embedding vectors for the batch
        """
        for attempt in range(self.max_retries):
            try:
                # Clean texts (remove None, empty strings)
                cleaned_texts = [str(t).strip() for t in texts if t]
                if not cleaned_texts:
                    return [[0.0] * self.dimensions] * len(texts)
                
                # Call OpenAI API
                response = self.client.embeddings.create(
                    model=self.model,
                    input=cleaned_texts
                )
                
                # Extract embeddings
                embeddings = [data.embedding for data in response.data]
                
                # Handle any missing texts (that were filtered)
                if len(embeddings) < len(texts):
                    result = []
                    clean_idx = 0
                    for t in texts:
                        if t and str(t).strip():
                            result.append(embeddings[clean_idx])
                            clean_idx += 1
                        else:
                            result.append([0.0] * self.dimensions)
                    return result
                
                return embeddings
                
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    # Rate limit error - use exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                elif attempt == self.max_retries - 1:
                    logger.error(f"Failed to generate embeddings after {self.max_retries} attempts: {e}")
                    raise
                else:
                    logger.warning(f"Error generating embeddings (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay)
        
        # Shouldn't reach here, but return zeros if we do
        return [[0.0] * self.dimensions] * len(texts)
    
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.dimensions
    
    def get_embedding_type(self) -> str:
        """Get embedding type."""
        return "dense"


class SPLADEEmbedder(EmbeddingService):
    """
    SPLADE embedding service for generating sparse embeddings.
    
    Features:
    - Automatic CUDA/CPU detection
    - Memory-efficient batch processing
    - Sparse vector format for pgvector
    - Garbage collection for memory management
    """
    
    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        dimensions: int = 30522,
        batch_size: int = 5,
        max_length: int = 256,
        device: Optional[str] = None
    ):
        """
        Initialize SPLADE embedder.
        
        Args:
            model_name: Hugging Face model name
            dimensions: Sparse vector dimensions (vocabulary size)
            batch_size: Default batch size for processing
            max_length: Maximum sequence length
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            # Auto-detect device if not specified
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            
            logger.info(f"Loading SPLADE model: {model_name} on device: {self.device}")
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            self.torch = torch
            
        except ImportError:
            raise ImportError("transformers and torch packages not installed. Run: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SPLADE model: {e}")
        
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_length = max_length
        
        logger.info(f"SPLADE embedder initialized on {self.device}")
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[Dict[int, float]]:
        """
        Generate SPLADE sparse embeddings for texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Optional batch size override
            
        Returns:
            List of sparse embeddings as dictionaries {dimension: value}
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._generate_batch(batch)
            all_embeddings.extend(embeddings)
            
            # Memory management
            if i % (batch_size * 10) == 0:
                gc.collect()
                if self.device.type == 'cuda':
                    self.torch.cuda.empty_cache()
            
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return all_embeddings
    
    def _generate_batch(self, texts: List[str]) -> List[Dict[int, float]]:
        """
        Generate sparse embeddings for a batch.
        
        Args:
            texts: Batch of texts to embed
            
        Returns:
            List of sparse embeddings as dictionaries
        """
        try:
            # Tokenize texts
            tokens = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            
            # Generate embeddings
            with self.torch.no_grad():
                outputs = self.model(**tokens)
            
            # Extract embeddings and apply log(1 + ReLU(x))
            embeddings = outputs.last_hidden_state
            embeddings = self.torch.log1p(self.torch.relu(embeddings))
            
            # Max pooling over sequence dimension
            embeddings = self.torch.max(embeddings, dim=1).values
            
            # Convert to sparse format
            sparse_embeddings = []
            for vec in embeddings:
                # Get non-zero indices and values
                indices = vec.nonzero().squeeze().cpu().numpy()
                values = vec[indices].cpu().numpy()
                
                # Create sparse dictionary
                sparse_dict = {int(idx): float(val) for idx, val in zip(indices, values)}
                sparse_embeddings.append(sparse_dict)
            
            return sparse_embeddings
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"CUDA OOM with batch size {len(texts)}. Consider reducing batch_size.")
                if self.device.type == 'cuda':
                    self.torch.cuda.empty_cache()
                raise
            else:
                logger.error(f"Error generating SPLADE embeddings: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error in SPLADE embedding generation: {e}")
            raise
    
    def format_for_pgvector(self, sparse_dict: Dict[int, float]) -> str:
        """
        Format sparse embedding for pgvector sparsevec type.

        Args:
            sparse_dict: Sparse embedding dictionary

        Returns:
            Formatted string for pgvector
        """
        if not sparse_dict:
            return "{}/30522"

        # Filter out any indices that are out of bounds
        # Ensure indices are within the valid range (0 to dimensions-1)
        valid_items = [(idx, val) for idx, val in sparse_dict.items()
                       if 0 <= idx < self.dimensions and val != 0.0]

        if not valid_items:
            return "{}/30522"

        # Sort by index for consistent format
        sorted_items = sorted(valid_items)

        # Format for pgvector sparsevec
        formatted = "{" + ",".join(f"{idx}:{val:.6f}" for idx, val in sorted_items) + "}/" + str(self.dimensions)
        return formatted
    
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.dimensions
    
    def get_embedding_type(self) -> str:
        """Get embedding type."""
        return "sparse"
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            if hasattr(self, 'torch') and self.device.type == 'cuda':
                self.torch.cuda.empty_cache()
            gc.collect()
        except:
            pass


class HybridEmbedder(EmbeddingService):
    """
    Hybrid embedding service that combines dense and sparse embeddings.
    """
    
    def __init__(
        self,
        dense_embedder: EmbeddingService,
        sparse_embedder: EmbeddingService
    ):
        """
        Initialize hybrid embedder.
        
        Args:
            dense_embedder: Dense embedding service
            sparse_embedder: Sparse embedding service
        """
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        
        if dense_embedder.get_embedding_type() != "dense":
            raise ValueError("First embedder must be dense type")
        if sparse_embedder.get_embedding_type() != "sparse":
            raise ValueError("Second embedder must be sparse type")
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[Tuple[List[float], Dict[int, float]]]:
        """
        Generate both dense and sparse embeddings.
        
        Args:
            texts: List of texts to embed
            batch_size: Optional batch size override
            
        Returns:
            List of tuples (dense_embedding, sparse_embedding)
        """
        dense_embeddings = self.dense_embedder.generate_embeddings(texts, batch_size)
        sparse_embeddings = self.sparse_embedder.generate_embeddings(texts, batch_size)
        
        return list(zip(dense_embeddings, sparse_embeddings))
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get dimensions for both embedders."""
        return (
            self.dense_embedder.get_dimensions(),
            self.sparse_embedder.get_dimensions()
        )
    
    def get_embedding_type(self) -> str:
        """Get embedding type."""
        return "hybrid"