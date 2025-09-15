"""
Generation service for LLM-based text generation and RAG responses.
"""

import logging
import tiktoken
from typing import List, Dict, Any, Optional, Union, Generator
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GenerationModel(Enum):
    """Available generation models."""
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_5_MINI = "gpt-5-mini"


@dataclass
class GenerationResponse:
    """Container for generation response."""
    content: str
    model: str
    usage: Dict[str, int] = None
    cost: float = 0.0
    metadata: Dict[str, Any] = None


class GenerationService:
    """
    Service for text generation using OpenAI models.
    
    Features:
    - Multiple model support
    - Token counting and management
    - Cost tracking
    - Prompt templates
    - Streaming support
    - Context window optimization
    """
    
    # Model context windows
    MODEL_CONTEXT_WINDOWS = {
        "gpt-3.5-turbo": 16385,
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-5-mini": 400000  # 400k context window
    }
    
    # Model costs per 1K tokens (as of 2024)
    MODEL_COSTS = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-5-mini": {"input": 0.00015, "output": 0.0006}  # TBD - estimated pricing
    }
    
    def __init__(
        self,
        model: str = "gpt-5-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        track_costs: bool = True
    ):
        """
        Initialize generation service.
        
        Args:
            model: Model to use for generation
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            system_prompt: Default system prompt
            track_costs: Whether to track generation costs
        """
        try:
            from openai import OpenAI
            self.client = OpenAI()
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or "You are a helpful AI assistant specializing in information retrieval and question answering."
        self.track_costs = track_costs
        
        # Initialize tokenizer for counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except:
            # Fallback to cl100k_base for newer models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"Initialized generation service with model: {model}")
    
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[GenerationResponse, Generator[str, None, None]]:
        """
        Generate text based on prompt and context.
        
        Args:
            prompt: User prompt/question
            context: Optional context for RAG
            system_prompt: Override system prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            stream: Whether to stream the response
            
        Returns:
            GenerationResponse or generator for streaming
        """
        # Build messages
        messages = self._build_messages(prompt, context, system_prompt)
        
        # Check token limits
        messages = self._optimize_for_context_window(messages, max_tokens or self.max_tokens)
        
        # Generate response
        if stream:
            return self._generate_stream(
                messages,
                temperature or self.temperature,
                max_tokens or self.max_tokens
            )
        else:
            return self._generate_complete(
                messages,
                temperature or self.temperature,
                max_tokens or self.max_tokens
            )
    
    def generate_rag_response(
        self,
        query: str,
        search_results: List[Any],
        result_formatter: Optional[callable] = None,
        max_context_length: int = 8000,
        **kwargs
    ) -> GenerationResponse:
        """
        Generate RAG response from search results.
        
        Args:
            query: User query
            search_results: Search results to use as context
            result_formatter: Optional function to format results
            max_context_length: Maximum context length in characters
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        # Format search results as context
        if result_formatter:
            context = result_formatter(search_results)
        else:
            context = self._default_result_formatter(search_results, max_context_length)
        
        # Create RAG prompt
        rag_prompt = self._create_rag_prompt(query)
        
        # Generate response
        return self.generate(
            prompt=rag_prompt,
            context=context,
            **kwargs
        )
    
    def _build_messages(
        self,
        prompt: str,
        context: Optional[str],
        system_prompt: Optional[str]
    ) -> List[Dict[str, str]]:
        """
        Build messages for the chat completion.
        
        Args:
            prompt: User prompt
            context: Optional context
            system_prompt: System prompt
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system prompt
        system = system_prompt or self.system_prompt
        if context:
            system += f"\n\nUse the following context to answer the question:\n{context}"
        messages.append({"role": "system", "content": system})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _optimize_for_context_window(
        self,
        messages: List[Dict[str, str]],
        max_response_tokens: int
    ) -> List[Dict[str, str]]:
        """
        Optimize messages to fit within context window.
        
        Args:
            messages: Messages to optimize
            max_response_tokens: Reserved tokens for response
            
        Returns:
            Optimized messages
        """
        # Get context window for model
        context_window = self.MODEL_CONTEXT_WINDOWS.get(self.model, 8192)
        
        # Count current tokens
        total_tokens = sum(len(self.tokenizer.encode(msg["content"])) for msg in messages)
        
        # Check if within limits
        available_tokens = context_window - max_response_tokens - 100  # Safety margin
        
        if total_tokens <= available_tokens:
            return messages
        
        # Need to truncate context
        logger.warning(f"Context too long ({total_tokens} tokens), truncating to fit {available_tokens} tokens")
        
        # Truncate system message if it has context
        if len(messages) > 0 and "context" in messages[0]["content"].lower():
            system_msg = messages[0]["content"]
            # Find context portion and truncate it
            if "\n\nUse the following context" in system_msg:
                prefix, context_part = system_msg.split("\n\nUse the following context", 1)
                
                # Calculate how many tokens we need to remove
                tokens_to_remove = total_tokens - available_tokens
                
                # Truncate context proportionally
                context_tokens = len(self.tokenizer.encode(context_part))
                keep_ratio = max(0.3, 1 - (tokens_to_remove / context_tokens))
                keep_chars = int(len(context_part) * keep_ratio)
                
                truncated_context = context_part[:keep_chars] + "... [truncated]"
                messages[0]["content"] = prefix + "\n\nUse the following context" + truncated_context
        
        return messages
    
    def _generate_complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> GenerationResponse:
        """
        Generate complete response.
        
        Args:
            messages: Chat messages
            temperature: Temperature setting
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generation response
        """
        try:
            # GPT-5 mini has specific requirements:
            # - Uses max_completion_tokens instead of max_tokens
            # - Only supports temperature=1
            if self.model == "gpt-5-mini":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=1,  # GPT-5 mini only supports temperature=1
                    max_completion_tokens=max_tokens
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            # Extract content
            content = response.choices[0].message.content
            
            # Extract usage
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Calculate cost
            cost = 0.0
            if self.track_costs and self.model in self.MODEL_COSTS:
                costs = self.MODEL_COSTS[self.model]
                cost = (
                    (usage["prompt_tokens"] / 1000) * costs["input"] +
                    (usage["completion_tokens"] / 1000) * costs["output"]
                )
            
            # For GPT-5 mini, temperature is always 1
            actual_temperature = 1 if self.model == "gpt-5-mini" else temperature

            return GenerationResponse(
                content=content,
                model=self.model,
                usage=usage,
                cost=cost,
                metadata={"temperature": actual_temperature, "max_tokens": max_tokens}
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> Generator[str, None, None]:
        """
        Generate streaming response.
        
        Args:
            messages: Chat messages
            temperature: Temperature setting
            max_tokens: Maximum tokens to generate
            
        Yields:
            Content chunks
        """
        try:
            # GPT-5 mini has specific requirements:
            # - Uses max_completion_tokens instead of max_tokens
            # - Only supports temperature=1
            if self.model == "gpt-5-mini":
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=1,  # GPT-5 mini only supports temperature=1
                    max_completion_tokens=max_tokens,
                    stream=True
                )
            else:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True
                )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise
    
    def _create_rag_prompt(self, query: str) -> str:
        """
        Create RAG-specific prompt.
        
        Args:
            query: User query
            
        Returns:
            RAG prompt
        """
        return f"""Answer the following question based on the provided context. 
If the context doesn't contain enough information to answer the question completely, 
say so and provide what information is available.

Question: {query}

Please provide a comprehensive and accurate answer."""
    
    def _default_result_formatter(
        self,
        search_results: List[Any],
        max_length: int = 8000
    ) -> str:
        """
        Default formatter for search results.
        
        Args:
            search_results: Search results to format
            max_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(search_results, 1):
            # Extract content based on result type
            if hasattr(result, 'content'):
                content = result.content
            elif isinstance(result, dict) and 'content' in result:
                content = result['content']
            else:
                content = str(result)
            
            # Add result to context
            part = f"[{i}] {content}\n"
            part_length = len(part)
            
            if current_length + part_length > max_length:
                # Truncate if needed
                remaining = max_length - current_length
                if remaining > 100:  # Only add if meaningful amount remains
                    part = part[:remaining] + "..."
                    context_parts.append(part)
                break
            
            context_parts.append(part)
            current_length += part_length
        
        return "\n".join(context_parts)
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: Optional[str] = None
    ) -> float:
        """
        Estimate generation cost.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model to estimate for (default: current model)
            
        Returns:
            Estimated cost in dollars
        """
        model = model or self.model
        
        if model not in self.MODEL_COSTS:
            logger.warning(f"Cost data not available for model: {model}")
            return 0.0
        
        costs = self.MODEL_COSTS[model]
        return (
            (prompt_tokens / 1000) * costs["input"] +
            (completion_tokens / 1000) * costs["output"]
        )
    
    def create_sql_generation_prompt(
        self,
        query: str,
        schema_info: str
    ) -> str:
        """
        Create prompt for SQL generation.
        
        Args:
            query: Natural language query
            schema_info: Database schema information
            
        Returns:
            SQL generation prompt
        """
        return f"""Convert the following natural language query to SQL.

Database Schema:
{schema_info}

Query: {query}

Instructions:
1. Generate only the SQL query, no explanations
2. Use proper JOIN conditions where needed
3. Include appropriate WHERE clauses
4. Use LIMIT for result size control
5. Ensure the query is syntactically correct for PostgreSQL

SQL Query:"""
    
    def create_summary_prompt(
        self,
        text: str,
        max_length: int = 200
    ) -> str:
        """
        Create prompt for text summarization.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in words
            
        Returns:
            Summarization prompt
        """
        return f"""Summarize the following text in no more than {max_length} words. 
Focus on the key points and main ideas.

Text:
{text}

Summary:"""