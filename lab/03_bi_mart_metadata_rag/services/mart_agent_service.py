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
    """Configuration for the mart planning agent."""
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2000
    max_retries: int = 3

class MartPlanningAgent:
    """
    LLM-powered agent for generating data mart plans.
    Takes user questions and metadata search results to generate optimized mart schemas.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the mart planning agent."""
        self.config = config or AgentConfig()

        # Initialize OpenAI client
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = openai.OpenAI(api_key=api_key)
        self.search_service = MetadataSearchService()

        logger.info(f"Initialized mart planning agent with model: {self.config.model}")

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
        """Generate a mart plan using the LLM."""

        # Create the planning prompt
        prompt = self.create_mart_planning_prompt(user_question, query_type, search_results)

        logger.info("Generating mart plan with LLM...")

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a data warehouse architect. Generate optimal data mart schemas in JSON format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )

                # Extract JSON from response
                response_text = response.choices[0].message.content.strip()

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
                    logger.info("✓ Successfully generated mart plan")
                    return mart_plan
                except json.JSONDecodeError as e:
                    logger.warning(f"Attempt {attempt + 1}: JSON parsing failed: {e}")
                    if attempt == self.config.max_retries - 1:
                        raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {json_text}")
                    continue
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}: Plan validation failed: {e}")
                    if attempt == self.config.max_retries - 1:
                        raise ValueError(f"Failed to validate mart plan: {e}")
                    continue

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: LLM request failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                continue

        raise RuntimeError("Failed to generate mart plan after all retries")

    def validate_mart_plan(self, plan: MartPlan) -> PlanValidationResult:
        """Validate a mart plan for consistency and completeness."""
        errors = []
        warnings = []
        suggestions = []

        # Check if plan has at least one fact table
        if not plan.facts:
            errors.append("Mart plan must have at least one fact table")

        # Validate fact tables
        for fact in plan.facts:
            # Check if grain is defined
            if not fact.grain:
                errors.append(f"Fact table {fact.name} must have a defined grain")

            # Check if measures are defined
            if not fact.measures:
                warnings.append(f"Fact table {fact.name} has no measures defined")

            # Check if source tables are specified
            if not fact.source_tables:
                errors.append(f"Fact table {fact.name} must specify source tables")

            # Validate measure expressions
            for measure in fact.measures:
                if not measure.expression:
                    errors.append(f"Measure {measure.name} in {fact.name} must have an expression")

        # Validate dimension tables
        for dimension in plan.dimensions:
            if not dimension.key_column:
                errors.append(f"Dimension {dimension.name} must have a key column")

            if not dimension.attributes:
                warnings.append(f"Dimension {dimension.name} has no attributes defined")

        # Check for naming conflicts
        all_table_names = [f.name for f in plan.facts] + [d.name for d in plan.dimensions]
        if len(all_table_names) != len(set(all_table_names)):
            errors.append("Duplicate table names found in mart plan")

        # Add suggestions
        if len(plan.facts) > 3:
            suggestions.append("Consider if all fact tables are necessary - simpler marts are often more effective")

        if not plan.dimensions:
            suggestions.append("Consider adding dimension tables for better query performance and usability")

        return PlanValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )

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
        """Generate a human-readable explanation of the mart plan."""
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