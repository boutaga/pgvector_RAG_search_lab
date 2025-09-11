"""
Smart chunking strategies for text processing.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib

from .processor import Document, ProcessedChunk

logger = logging.getLogger(__name__)


class ChunkingStrategy(ABC):
    """
    Abstract base class for chunking strategies.
    """
    
    @abstractmethod
    def chunk_document(self, document: Document) -> List[ProcessedChunk]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunks
        """
        pass
    
    @abstractmethod
    def get_chunk_size(self) -> int:
        """Get the target chunk size."""
        pass


class FixedSizeChunker(ChunkingStrategy):
    """
    Fixed-size chunking with optional overlap.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap_size: int = 100,
        preserve_words: bool = True
    ):
        """
        Initialize fixed-size chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            overlap_size: Overlap size between chunks
            preserve_words: Whether to preserve word boundaries
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.preserve_words = preserve_words
    
    def chunk_document(self, document: Document) -> List[ProcessedChunk]:
        """
        Chunk document with fixed size.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunks
        """
        content = document.content
        chunks = []
        position = 0
        chunk_index = 0
        
        while position < len(content):
            # Calculate end position
            end_pos = min(position + self.chunk_size, len(content))
            
            # Extract chunk content
            chunk_content = content[position:end_pos]
            
            # Preserve word boundaries if requested
            if self.preserve_words and end_pos < len(content):
                # Find last word boundary
                last_space = chunk_content.rfind(' ')
                if last_space > len(chunk_content) * 0.8:  # Only if we don't lose too much
                    chunk_content = chunk_content[:last_space]
                    end_pos = position + last_space
            
            # Calculate overlap
            overlap_prev = self.overlap_size if chunk_index > 0 else 0
            overlap_next = self.overlap_size if end_pos < len(content) else 0
            
            # Create chunk
            chunk_id = self._generate_chunk_id(document.id, chunk_index)
            chunk = ProcessedChunk(
                chunk_id=chunk_id,
                document_id=document.id,
                content=chunk_content.strip(),
                metadata={
                    'original_title': document.title,
                    'original_source': document.source,
                    'chunk_strategy': 'fixed_size',
                    'chunk_size': len(chunk_content),
                    **(document.metadata or {})
                },
                position=chunk_index,
                overlap_with_previous=overlap_prev,
                overlap_with_next=overlap_next
            )
            chunks.append(chunk)
            
            # Move to next position
            position = end_pos - self.overlap_size
            chunk_index += 1
            
            # Prevent infinite loop
            if position <= 0:
                break
        
        return chunks
    
    def get_chunk_size(self) -> int:
        """Get chunk size."""
        return self.chunk_size
    
    def _generate_chunk_id(self, document_id: Any, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{document_id}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class SemanticChunker(ChunkingStrategy):
    """
    Semantic chunking based on sentences and paragraphs.
    """
    
    def __init__(
        self,
        target_size: int = 1000,
        min_size: int = 200,
        max_size: int = 2000,
        paragraph_separator: str = "\n\n"
    ):
        """
        Initialize semantic chunker.
        
        Args:
            target_size: Target chunk size in characters
            min_size: Minimum chunk size
            max_size: Maximum chunk size
            paragraph_separator: Separator for paragraphs
        """
        self.target_size = target_size
        self.min_size = min_size
        self.max_size = max_size
        self.paragraph_separator = paragraph_separator
    
    def chunk_document(self, document: Document) -> List[ProcessedChunk]:
        """
        Chunk document semantically.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunks
        """
        content = document.content
        
        # First, split by paragraphs
        paragraphs = content.split(self.paragraph_separator)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            # If adding this paragraph would exceed max size, finalize current chunk
            if current_size + paragraph_size > self.max_size and current_chunk:
                chunk = self._create_chunk(document, current_chunk, chunk_index)
                chunks.append(chunk)
                current_chunk = []
                current_size = 0
                chunk_index += 1
            
            # If paragraph is too large, split it by sentences
            if paragraph_size > self.max_size:
                sentences = self._split_sentences(paragraph)
                for sentence in sentences:
                    sentence_size = len(sentence)
                    
                    if current_size + sentence_size > self.max_size and current_chunk:
                        chunk = self._create_chunk(document, current_chunk, chunk_index)
                        chunks.append(chunk)
                        current_chunk = []
                        current_size = 0
                        chunk_index += 1
                    
                    current_chunk.append(sentence)
                    current_size += sentence_size
                    
                    # If we've reached target size, consider finalizing
                    if current_size >= self.target_size:
                        chunk = self._create_chunk(document, current_chunk, chunk_index)
                        chunks.append(chunk)
                        current_chunk = []
                        current_size = 0
                        chunk_index += 1
            else:
                current_chunk.append(paragraph)
                current_size += paragraph_size
                
                # If we've reached target size, finalize chunk
                if current_size >= self.target_size:
                    chunk = self._create_chunk(document, current_chunk, chunk_index)
                    chunks.append(chunk)
                    current_chunk = []
                    current_size = 0
                    chunk_index += 1
        
        # Handle remaining content
        if current_chunk and current_size >= self.min_size:
            chunk = self._create_chunk(document, current_chunk, chunk_index)
            chunks.append(chunk)
        elif current_chunk and chunks:
            # Add to last chunk if too small
            last_chunk = chunks[-1]
            additional_content = self.paragraph_separator.join(current_chunk)
            last_chunk.content += self.paragraph_separator + additional_content
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunk(
        self,
        document: Document,
        content_parts: List[str],
        chunk_index: int
    ) -> ProcessedChunk:
        """Create chunk from content parts."""
        content = self.paragraph_separator.join(content_parts)
        chunk_id = self._generate_chunk_id(document.id, chunk_index)
        
        return ProcessedChunk(
            chunk_id=chunk_id,
            document_id=document.id,
            content=content,
            metadata={
                'original_title': document.title,
                'original_source': document.source,
                'chunk_strategy': 'semantic',
                'chunk_size': len(content),
                'num_paragraphs': len(content_parts),
                **(document.metadata or {})
            },
            position=chunk_index
        )
    
    def get_chunk_size(self) -> int:
        """Get target chunk size."""
        return self.target_size
    
    def _generate_chunk_id(self, document_id: Any, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{document_id}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class HierarchicalChunker(ChunkingStrategy):
    """
    Hierarchical chunking that respects document structure.
    """
    
    def __init__(
        self,
        chunk_sizes: List[int] = [2000, 1000, 500],
        overlap_ratio: float = 0.1,
        structure_patterns: Optional[Dict[str, str]] = None
    ):
        """
        Initialize hierarchical chunker.
        
        Args:
            chunk_sizes: List of chunk sizes for hierarchy levels
            overlap_ratio: Ratio of overlap between chunks
            structure_patterns: Regex patterns for detecting structure
        """
        self.chunk_sizes = sorted(chunk_sizes, reverse=True)
        self.overlap_ratio = overlap_ratio
        self.structure_patterns = structure_patterns or {
            'heading': r'^#+\s+(.+)$',
            'section': r'^[0-9]+\.\s+(.+)$',
            'subsection': r'^[0-9]+\.[0-9]+\s+(.+)$'
        }
    
    def chunk_document(self, document: Document) -> List[ProcessedChunk]:
        """
        Chunk document hierarchically.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunks
        """
        content = document.content
        lines = content.split('\n')
        
        # Detect document structure
        structure = self._detect_structure(lines)
        
        # Create chunks at multiple levels
        all_chunks = []
        
        for level, chunk_size in enumerate(self.chunk_sizes):
            level_chunks = self._create_level_chunks(
                document, structure, chunk_size, level
            )
            all_chunks.extend(level_chunks)
        
        return all_chunks
    
    def _detect_structure(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Detect document structure.
        
        Args:
            lines: Document lines
            
        Returns:
            List of structure elements
        """
        structure = []
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for structure patterns
            for pattern_name, pattern in self.structure_patterns.items():
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    structure.append({
                        'type': pattern_name,
                        'title': match.group(1) if match.groups() else line,
                        'line_num': line_num,
                        'level': pattern_name
                    })
                    break
        
        return structure
    
    def _create_level_chunks(
        self,
        document: Document,
        structure: List[Dict[str, Any]],
        chunk_size: int,
        level: int
    ) -> List[ProcessedChunk]:
        """Create chunks for a specific level."""
        chunks = []
        content = document.content
        overlap_size = int(chunk_size * self.overlap_ratio)
        
        # Use fixed-size chunking with structure awareness
        chunker = FixedSizeChunker(chunk_size, overlap_size, preserve_words=True)
        base_chunks = chunker.chunk_document(document)
        
        # Enhance chunks with hierarchical metadata
        for i, chunk in enumerate(base_chunks):
            chunk.metadata.update({
                'hierarchy_level': level,
                'hierarchy_chunk_size': chunk_size,
                'chunk_strategy': 'hierarchical'
            })
            
            # Add structure context
            chunk_start = i * (chunk_size - overlap_size)
            relevant_structure = [
                s for s in structure 
                if abs(s['line_num'] * 50 - chunk_start) < chunk_size  # Rough estimate
            ]
            
            if relevant_structure:
                chunk.metadata['structure_context'] = relevant_structure
            
            chunks.append(chunk)
        
        return chunks
    
    def get_chunk_size(self) -> int:
        """Get largest chunk size."""
        return max(self.chunk_sizes)


class AdaptiveChunker(ChunkingStrategy):
    """
    Adaptive chunking that adjusts based on content characteristics.
    """
    
    def __init__(
        self,
        base_size: int = 1000,
        density_threshold: float = 0.8,
        complexity_threshold: float = 0.6,
        min_size: int = 200,
        max_size: int = 2500
    ):
        """
        Initialize adaptive chunker.
        
        Args:
            base_size: Base chunk size
            density_threshold: Information density threshold
            complexity_threshold: Text complexity threshold
            min_size: Minimum chunk size
            max_size: Maximum chunk size
        """
        self.base_size = base_size
        self.density_threshold = density_threshold
        self.complexity_threshold = complexity_threshold
        self.min_size = min_size
        self.max_size = max_size
    
    def chunk_document(self, document: Document) -> List[ProcessedChunk]:
        """
        Chunk document adaptively.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunks
        """
        content = document.content
        
        # Analyze content characteristics
        density = self._calculate_information_density(content)
        complexity = self._calculate_text_complexity(content)
        
        # Adjust chunk size based on characteristics
        adjusted_size = self._adjust_chunk_size(density, complexity)
        
        # Use semantic chunking with adjusted size
        chunker = SemanticChunker(
            target_size=adjusted_size,
            min_size=self.min_size,
            max_size=self.max_size
        )
        
        chunks = chunker.chunk_document(document)
        
        # Add adaptive metadata
        for chunk in chunks:
            chunk.metadata.update({
                'chunk_strategy': 'adaptive',
                'adapted_size': adjusted_size,
                'content_density': density,
                'content_complexity': complexity
            })
        
        return chunks
    
    def _calculate_information_density(self, text: str) -> float:
        """
        Calculate information density of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Density score between 0 and 1
        """
        words = text.split()
        if not words:
            return 0.0
        
        # Simple metrics for information density
        unique_words = len(set(word.lower() for word in words))
        word_ratio = unique_words / len(words)
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        char_ratio = unique_chars / max(len(text), 1)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        length_factor = min(avg_word_length / 6.0, 1.0)  # Normalize around 6 chars
        
        # Combine metrics
        density = (word_ratio * 0.4 + char_ratio * 0.3 + length_factor * 0.3)
        return min(density, 1.0)
    
    def _calculate_text_complexity(self, text: str) -> float:
        """
        Calculate text complexity.
        
        Args:
            text: Text to analyze
            
        Returns:
            Complexity score between 0 and 1
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Average sentence length
        words = text.split()
        avg_sentence_length = len(words) / len(sentences)
        length_factor = min(avg_sentence_length / 20.0, 1.0)  # Normalize around 20 words
        
        # Punctuation density
        punctuation = sum(1 for char in text if char in '.,;:!?()[]{}')
        punct_density = min(punctuation / max(len(text), 1) * 100, 1.0)
        
        # Uppercase ratio (might indicate technical content)
        uppercase = sum(1 for char in text if char.isupper())
        upper_ratio = min(uppercase / max(len(text), 1) * 20, 1.0)
        
        # Combine metrics
        complexity = (length_factor * 0.5 + punct_density * 0.3 + upper_ratio * 0.2)
        return min(complexity, 1.0)
    
    def _adjust_chunk_size(self, density: float, complexity: float) -> int:
        """
        Adjust chunk size based on content characteristics.
        
        Args:
            density: Information density
            complexity: Text complexity
            
        Returns:
            Adjusted chunk size
        """
        # High density or complexity -> smaller chunks
        # Low density or complexity -> larger chunks
        
        density_factor = 1.0 - (density - 0.5) * 0.5
        complexity_factor = 1.0 - (complexity - 0.5) * 0.5
        
        adjustment = (density_factor + complexity_factor) / 2
        adjusted_size = int(self.base_size * adjustment)
        
        # Apply bounds
        adjusted_size = max(self.min_size, min(adjusted_size, self.max_size))
        
        return adjusted_size
    
    def get_chunk_size(self) -> int:
        """Get base chunk size."""
        return self.base_size


class ChunkingManager:
    """
    Manager for different chunking strategies.
    """
    
    def __init__(self):
        """Initialize chunking manager."""
        self.strategies = {
            'fixed': FixedSizeChunker,
            'semantic': SemanticChunker,
            'hierarchical': HierarchicalChunker,
            'adaptive': AdaptiveChunker
        }
    
    def register_strategy(self, name: str, strategy_class: type):
        """
        Register a new chunking strategy.
        
        Args:
            name: Strategy name
            strategy_class: Strategy class
        """
        self.strategies[name] = strategy_class
    
    def create_chunker(self, strategy: str, **kwargs) -> ChunkingStrategy:
        """
        Create chunker instance.
        
        Args:
            strategy: Strategy name
            **kwargs: Strategy parameters
            
        Returns:
            Chunker instance
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategies.keys())}")
        
        return self.strategies[strategy](**kwargs)
    
    def chunk_documents(
        self,
        documents: List[Document],
        strategy: str,
        **strategy_kwargs
    ) -> List[ProcessedChunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: Documents to chunk
            strategy: Chunking strategy
            **strategy_kwargs: Strategy parameters
            
        Returns:
            List of all chunks
        """
        chunker = self.create_chunker(strategy, **strategy_kwargs)
        all_chunks = []
        
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks