#!/usr/bin/env python3
"""
Mart planning agent service.
Uses LLM to generate data mart plans based on RAG search results.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import openai
import logging
from dataclasses import dataclass

# Add the models directory to the path
script_dir = Path(__file__).parent
lab_dir = script_dir.parent
models_dir = lab_dir / 'models'
sys.path.insert(0, str(models_dir))

from mart_plan import MartPlan, FactDefinition, DimensionDefinition, MeasureDefinition, PlanValidationResult
from metadata_search_service import MetadataSearchService, SearchResult, QueryType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for the mart planning agent with GPT-5 support."""
    primary_model: str = "gpt-5"
    fast_model: str = "gpt-5-mini"
    fallback_model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 3000
    max_retries: int = 3

    # GPT-5 specific configurations
    gpt5_config: Dict[str, Any] = None
    gpt5_mini_config: Dict[str, Any] = None
    task_routing: Dict[str, str] = None

    def __post_init__(self):
        if self.gpt5_config is None:
            self.gpt5_config = {
                "temperature": 0.1,
                "max_tokens": 3000,
                "reasoning_depth": "thorough",
                "structured_output": True
            }
        if self.gpt5_mini_config is None:
            self.gpt5_mini_config = {
                "temperature": 0.05,
                "max_tokens": 1500,
                "optimization": "speed"
            }
        if self.task_routing is None:
            self.task_routing = {
                "complex_planning": "gpt-5",
                "plan_validation": "gpt-5-mini",
                "plan_explanation": "gpt-5-mini",
                "error_analysis": "gpt-5",
                "optimization_suggestions": "gpt-5"
            }

class MartPlanningAgent:
    """
    LLM-powered agent for generating data mart plans.
    Takes user questions and metadata search results to generate optimized mart schemas.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the mart planning agent with GPT-5 support."""
        self.config = config or AgentConfig()

        # Initialize OpenAI client
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = openai.OpenAI(api_key=api_key)
        self.search_service = MetadataSearchService()

        # Test model availability and set up routing
        self._test_model_availability()

        logger.info(f"Initialized mart planning agent with primary model: {self.config.primary_model}, fast model: {self.config.fast_model}")

    def _test_model_availability(self):
        """Test availability of GPT-5 models and adjust configuration if needed."""
        try:
            # Test GPT-5 availability with a simple request
            response = self.client.chat.completions.create(
                model=self.config.primary_model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
            logger.info(f"✓ {self.config.primary_model} is available")
        except Exception as e:
            logger.warning(f"⚠ {self.config.primary_model} not available, falling back to {self.config.fallback_model}: {e}")
            self.config.primary_model = self.config.fallback_model
            # Update task routing to use fallback model
            for task in self.config.task_routing:
                if self.config.task_routing[task] == "gpt-5":
                    self.config.task_routing[task] = self.config.fallback_model

        try:
            # Test GPT-5-mini availability
            response = self.client.chat.completions.create(
                model=self.config.fast_model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
            logger.info(f"✓ {self.config.fast_model} is available")
        except Exception as e:
            logger.warning(f"⚠ {self.config.fast_model} not available, using {self.config.fallback_model}: {e}")
            self.config.fast_model = self.config.fallback_model
            # Update task routing to use fallback model
            for task in self.config.task_routing:
                if self.config.task_routing[task] == "gpt-5-mini":
                    self.config.task_routing[task] = self.config.fallback_model

    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if model_name == "gpt-5":
            return self.config.gpt5_config
        elif model_name == "gpt-5-mini":
            return self.config.gpt5_mini_config
        else:
            # Fallback configuration
            return {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }

    def _call_llm(self, task_type: str, messages: List[Dict[str, str]]) -> str:
        """
        Call the appropriate LLM model based on task type.

        Args:
            task_type: Type of task (complex_planning, plan_validation, etc.)
            messages: Chat messages for the LLM

        Returns:
            Response text from the LLM
        """
        # Determine which model to use for this task
        model_name = self.config.task_routing.get(task_type, self.config.primary_model)
        model_config = self._get_model_config(model_name)

        logger.info(f"Using {model_name} for {task_type}")

        for attempt in range(self.config.max_retries):
            try:
                # Prepare request parameters
                request_params = {
                    "model": model_name,
                    "messages": messages,
                    **model_config
                }

                # Remove any parameters that the model doesn't support
                if model_name.startswith("gpt-4"):
                    # Remove GPT-5 specific parameters
                    request_params.pop("reasoning_depth", None)
                    request_params.pop("structured_output", None)
                    request_params.pop("optimization", None)

                response = self.client.chat.completions.create(**request_params)
                return response.choices[0].message.content.strip()

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed with {model_name}: {e}")
                if attempt == self.config.max_retries - 1:
                    # Last attempt, try fallback model if not already using it
                    if model_name != self.config.fallback_model:
                        logger.info(f"Trying fallback model {self.config.fallback_model}")
                        model_name = self.config.fallback_model
                        model_config = self._get_model_config(model_name)
                        continue
                    else:
                        raise
                continue

        raise RuntimeError(f"Failed to get response from LLM after all retries")

    def create_mart_planning_prompt(
        self,
        user_question: str,
        query_type: QueryType,
        search_results: List[SearchResult]
    ) -> str:
        """Create a detailed prompt for the LLM to generate a mart plan."""

        prompt_parts = [
            "You are a data warehouse architect specializing in dimensional modeling.",
            "Your task is to design an optimal data mart schema based on a user's business question",
            "and relevant database metadata discovered through semantic search.\n",

            "## Context",
            f"User Question: {user_question}",
            f"Query Type: {query_type.value}",
            f"Available Metadata: {len(search_results)} relevant elements found\n",

            "## Available Metadata",
            "The following database elements were identified as relevant to the user's question:\n"
        ]

        # Add metadata details
        tables_found = set()
        columns_by_table = {}
        relationships = []
        kpis = []

        for result in search_results:
            if result.metadata_type == 'table':
                tables_found.add(result.table_name)
                prompt_parts.append(
                    f"TABLE: {result.table_name}\n"
                    f"  - Description: {result.description}\n"
                    f"  - Rows: {result.additional_info.get('row_count', 'unknown')}\n"
                )

            elif result.metadata_type == 'column':
                table = result.table_name
                if table not in columns_by_table:
                    columns_by_table[table] = []

                column_info = {
                    'name': result.column_name,
                    'type': result.additional_info.get('data_type'),
                    'is_pk': result.additional_info.get('is_primary_key'),
                    'is_fk': result.additional_info.get('is_foreign_key'),
                    'referenced_table': result.additional_info.get('referenced_table'),
                    'description': result.description
                }
                columns_by_table[table].append(column_info)

            elif result.metadata_type == 'relationship':
                relationships.append({
                    'source': f"{result.additional_info.get('source_table')}.{result.additional_info.get('source_column')}",
                    'target': f"{result.additional_info.get('target_table')}.{result.additional_info.get('target_column')}",
                    'type': result.additional_info.get('relationship_type')
                })

            elif result.metadata_type == 'kpi':
                kpis.append({
                    'name': result.additional_info.get('kpi_name'),
                    'category': result.additional_info.get('kpi_category'),
                    'expression': result.additional_info.get('measure_expression'),
                    'tables': result.additional_info.get('required_tables'),
                    'description': result.description
                })

        # Add column details by table
        if columns_by_table:
            prompt_parts.append("\nCOLUMNS BY TABLE:")
            for table, columns in columns_by_table.items():
                prompt_parts.append(f"\n{table}:")
                for col in columns:
                    flags = []
                    if col['is_pk']:
                        flags.append("PK")
                    if col['is_fk']:
                        flags.append(f"FK -> {col['referenced_table']}")
                    flag_str = f" [{', '.join(flags)}]" if flags else ""

                    prompt_parts.append(f"  - {col['name']} ({col['type']}){flag_str}: {col['description']}")

        # Add relationships
        if relationships:
            prompt_parts.append("\nRELATIONSHIPS:")
            for rel in relationships:
                prompt_parts.append(f"  - {rel['source']} -> {rel['target']} ({rel['type']})")

        # Add suggested KPIs
        if kpis:
            prompt_parts.append("\nSUGGESTED KPIs:")
            for kpi in kpis:
                prompt_parts.append(f"  - {kpi['name']} ({kpi['category']}): {kpi['description']}")
                if kpi['expression']:
                    prompt_parts.append(f"    Expression: {kpi['expression']}")

        # Add design guidelines
        prompt_parts.extend([
            "\n## Design Guidelines",
            "1. Follow dimensional modeling best practices (Kimball methodology)",
            "2. Create fact tables for metrics/measures and dimension tables for descriptive attributes",
            "3. Use appropriate grain for fact tables (atomic level preferred)",
            "4. Include all necessary foreign keys for joins",
            "5. Optimize for the specific business question asked",
            "6. Consider performance - add appropriate indexes",
            "7. Use clear, business-friendly naming conventions\n",

            "## Required Output Format",
            "Generate a JSON response with the following structure:",
            "```json",
            "{",
            '  "source_schema": "src_northwind",',
            '  "target_schema": "mart_[business_area]",',
            '  "facts": [',
            '    {',
            '      "name": "fact_[name]",',
            '      "grain": ["key1", "key2"],',
            '      "measures": [',
            '        {',
            '          "name": "measure_name",',
            '          "expression": "sql_expression",',
            '          "aggregation": "sum|count|avg|max|min",',
            '          "description": "business description"',
            '        }',
            '      ],',
            '      "dimension_keys": ["dim_key1", "dim_key2"],',
            '      "source_tables": ["table1", "table2"],',
            '      "join_conditions": ["table1.id = table2.table1_id"],',
            '      "description": "business description"',
            '    }',
            '  ],',
            '  "dimensions": [',
            '    {',
            '      "name": "dim_[name]",',
            '      "source_table": "source_table",',
            '      "key_column": "primary_key",',
            '      "attributes": ["attr1", "attr2", "attr3"],',
            '      "description": "business description"',
            '    }',
            '  ]',
            '}',
            "```\n",

            "Focus specifically on answering the user's business question with an optimal mart design.",
            "Ensure all table and column names referenced in the plan actually exist in the metadata provided.",
            "Return ONLY the JSON response, no additional text or explanation."
        ])

        return "\n".join(prompt_parts)

    def generate_mart_plan(
        self,
        user_question: str,
        query_type: QueryType,
        search_results: List[SearchResult]
    ) -> MartPlan:
        """Generate a mart plan using GPT-5 for complex planning."""

        # Create the planning prompt
        prompt = self.create_mart_planning_prompt(user_question, query_type, search_results)

        logger.info("Generating mart plan with GPT-5...")

        # Enhanced system prompt for GPT-5
        system_prompt = """You are an expert data warehouse architect with deep expertise in dimensional modeling and modern BI practices.

Your task is to generate optimal data mart schemas that follow Kimball methodology while leveraging GPT-5's enhanced reasoning capabilities.

Key capabilities to utilize:
1. Deep analytical reasoning about business requirements
2. Advanced pattern recognition for optimal schema design
3. Sophisticated understanding of performance implications
4. Enhanced JSON structure generation with validation

Generate schemas that are:
- Optimized for the specific business question
- Performant for analytical workloads
- Maintainable and extensible
- Business-user friendly

Return ONLY valid JSON with no additional text."""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Use complex_planning task routing for GPT-5
        response_text = self._call_llm("complex_planning", messages)

        # Parse and validate the response
        return self._parse_mart_plan_response(response_text)

    def _parse_mart_plan_response(self, response_text: str) -> MartPlan:
        """Parse and validate the mart plan response from LLM."""
        # Try to extract JSON from response (handle code blocks)
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        else:
            json_text = response_text

        # Parse JSON
        try:
            plan_data = json.loads(json_text)
            mart_plan = MartPlan(**plan_data)
            logger.info("✓ Successfully generated mart plan with GPT-5")
            return mart_plan
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}\nResponse: {json_text}")
            raise ValueError(f"Failed to parse JSON response: {e}")
        except Exception as e:
            logger.error(f"Plan validation failed: {e}")
            raise ValueError(f"Failed to validate mart plan: {e}")

    def validate_mart_plan(self, plan: MartPlan) -> PlanValidationResult:
        """Validate a mart plan using GPT-5-mini for fast and thorough validation."""
        # First do basic structural validation
        errors = []
        warnings = []
        suggestions = []

        # Basic structural checks
        if not plan.facts:
            errors.append("Mart plan must have at least one fact table")

        # Check for naming conflicts
        all_table_names = [f.name for f in plan.facts] + [d.name for d in plan.dimensions]
        if len(all_table_names) != len(set(all_table_names)):
            errors.append("Duplicate table names found in mart plan")

        # If basic validation fails, return immediately
        if errors:
            return PlanValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions
            )

        # Use GPT-5-mini for advanced validation
        try:
            advanced_validation = self._validate_with_llm(plan)
            errors.extend(advanced_validation.get('errors', []))
            warnings.extend(advanced_validation.get('warnings', []))
            suggestions.extend(advanced_validation.get('suggestions', []))
        except Exception as e:
            warnings.append(f"Advanced validation failed: {str(e)}")

        return PlanValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )

    def _validate_with_llm(self, plan: MartPlan) -> Dict[str, List[str]]:
        """Use GPT-5-mini for advanced plan validation."""
        validation_prompt = f"""Analyze this data mart plan for potential issues and improvements:

{json.dumps(plan.dict(), indent=2)}

Check for:
1. Dimensional modeling best practices
2. Naming consistency and conventions
3. Fact table grain appropriateness
4. Measure expression validity
5. Dimension design quality
6. Performance considerations
7. Business usability

Return a JSON object with three arrays:
{{
  "errors": ["critical issues that must be fixed"],
  "warnings": ["potential issues to consider"],
  "suggestions": ["improvement recommendations"]
}}

Focus on practical, actionable feedback. Be concise but thorough."""

        messages = [
            {
                "role": "system",
                "content": "You are a data warehouse validation expert. Analyze mart plans and provide structured feedback."
            },
            {
                "role": "user",
                "content": validation_prompt
            }
        ]

        response_text = self._call_llm("plan_validation", messages)

        try:
            # Parse the validation response
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_text = response_text[start:end].strip()
            else:
                json_text = response_text.strip()

            return json.loads(json_text)
        except Exception as e:
            logger.warning(f"Failed to parse validation response: {e}")
            return {"errors": [], "warnings": [], "suggestions": []}

    def plan_mart_from_question(self, user_question: str) -> Tuple[MartPlan, List[SearchResult]]:
        """
        Complete workflow: search metadata and generate mart plan from user question.

        Args:
            user_question: Natural language business question

        Returns:
            Tuple of (mart_plan, search_results_used)
        """
        logger.info(f"Planning mart for question: {user_question}")

        # Search for relevant metadata
        query_type, search_results = self.search_service.search_metadata(user_question)

        if not search_results:
            raise ValueError("No relevant metadata found for the question")

        logger.info(f"Found {len(search_results)} relevant metadata elements")

        # Generate mart plan
        mart_plan = self.generate_mart_plan(user_question, query_type, search_results)

        # Validate plan
        validation = self.validate_mart_plan(mart_plan)
        if not validation.is_valid:
            logger.warning(f"Plan validation found errors: {validation.errors}")
            # Could try to regenerate or fix plan here

        if validation.warnings:
            logger.warning(f"Plan validation warnings: {validation.warnings}")

        return mart_plan, search_results

    def explain_mart_plan(self, plan: MartPlan, user_question: str) -> str:
        """Generate a human-readable explanation using GPT-5-mini for enhanced clarity."""

        # Create prompt for GPT-5-mini to generate explanation
        explanation_prompt = f"""Create a clear, business-friendly explanation of this data mart plan:

**Business Question:** {user_question}

**Mart Plan:**
{json.dumps(plan.dict(), indent=2)}

Generate an explanation that covers:
1. Overview of the mart design
2. How it answers the business question
3. Key fact tables and their purpose
4. Important dimensions for analysis
5. Business benefits and use cases

Write in markdown format, use business-friendly language, and focus on value to business users.
Keep it concise but comprehensive."""

        messages = [
            {
                "role": "system",
                "content": "You are a BI consultant explaining data mart designs to business users. Make technical concepts accessible and focus on business value."
            },
            {
                "role": "user",
                "content": explanation_prompt
            }
        ]

        try:
            # Use GPT-5-mini for fast, high-quality explanations
            explanation = self._call_llm("plan_explanation", messages)
            return explanation
        except Exception as e:
            logger.warning(f"Failed to generate LLM explanation, using fallback: {e}")
            # Fallback to static explanation
            return self._generate_static_explanation(plan, user_question)

    def _generate_static_explanation(self, plan: MartPlan, user_question: str) -> str:
        """Fallback static explanation generator."""
        explanation_parts = [
            f"## Data Mart Plan for: '{user_question}'\n",
            f"**Target Schema:** {plan.target_schema}",
            f"**Source Schema:** {plan.source_schema}\n"
        ]

        # Explain fact tables
        if plan.facts:
            explanation_parts.append("### Fact Tables")
            for fact in plan.facts:
                explanation_parts.append(f"\n**{fact.name}**")
                explanation_parts.append(f"- *Purpose:* {fact.description or 'Main fact table for analysis'}")
                explanation_parts.append(f"- *Grain:* {', '.join(fact.grain)}")

                if fact.measures:
                    explanation_parts.append("- *Measures:*")
                    for measure in fact.measures:
                        explanation_parts.append(f"  - **{measure.name}**: {measure.description or measure.expression}")

                explanation_parts.append(f"- *Source Tables:* {', '.join(fact.source_tables)}")

        # Explain dimension tables
        if plan.dimensions:
            explanation_parts.append("\n### Dimension Tables")
            for dim in plan.dimensions:
                explanation_parts.append(f"\n**{dim.name}**")
                explanation_parts.append(f"- *Purpose:* {dim.description or 'Descriptive attributes for analysis'}")
                explanation_parts.append(f"- *Key:* {dim.key_column}")
                explanation_parts.append(f"- *Attributes:* {', '.join(dim.attributes)}")
                explanation_parts.append(f"- *Source:* {dim.source_table}")

        # Add benefits explanation
        explanation_parts.extend([
            "\n### Benefits of This Design",
            "- **Performance**: Optimized for analytical queries with pre-aggregated measures",
            "- **Usability**: Business-friendly column names and structure",
            "- **Scalability**: Dimensional model scales well for reporting and BI tools",
            "- **Flexibility**: Supports ad-hoc analysis and drilling down/up"
        ])

        return "\n".join(explanation_parts)