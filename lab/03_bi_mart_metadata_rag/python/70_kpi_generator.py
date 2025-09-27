#!/usr/bin/env python3
"""
KPI Query Generator - Creates optimized KPI queries for the generated data mart
Uses GPT-5-mini for fast query generation and optimization
"""

import os
import sys
import json
import psycopg2
from pathlib import Path
from typing import List, Dict, Optional, Any
import openai
import logging

# Add models directory to path
script_dir = Path(__file__).parent
lab_dir = script_dir.parent
models_dir = lab_dir / 'models'
services_dir = lab_dir / 'services'
sys.path.insert(0, str(models_dir))
sys.path.insert(0, str(services_dir))

from mart_plan import MartPlan
from mart_agent_service import AgentConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KPIGenerator:
    """Generates KPI queries for data marts using AI assistance"""

    def __init__(self):
        """Initialize the KPI generator"""
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-5-mini"  # Use fast model for query generation

        # Test if GPT-5-mini is available, fallback to GPT-4 if not
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
            logger.info(f"Using {self.model} for KPI generation")
        except:
            self.model = "gpt-4"
            logger.info(f"GPT-5-mini not available, using {self.model}")

    def generate_kpi_queries(
        self,
        mart_plan: MartPlan,
        business_requirements: str,
        num_queries: int = 5
    ) -> List[Dict[str, str]]:
        """Generate KPI queries based on mart plan and business requirements"""

        prompt = self.create_kpi_generation_prompt(mart_plan, business_requirements, num_queries)

        messages = [
            {
                "role": "system",
                "content": """You are a BI analyst expert who creates optimized SQL queries for KPI dashboards.
Generate practical, performant queries that business users can understand and use.
Focus on common patterns: time series, top-N, comparisons, and aggregations."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )

            response_text = response.choices[0].message.content.strip()
            return self.parse_kpi_response(response_text)

        except Exception as e:
            logger.error(f"Failed to generate KPI queries: {e}")
            return self.generate_fallback_queries(mart_plan)

    def create_kpi_generation_prompt(
        self,
        mart_plan: MartPlan,
        business_requirements: str,
        num_queries: int
    ) -> str:
        """Create prompt for KPI query generation"""

        # Extract schema information
        fact_info = []
        for fact in mart_plan.facts:
            measures = [f"{m.name} ({m.aggregation})" for m in fact.measures]
            fact_info.append(f"- {fact.name}: {', '.join(measures)}")

        dimension_info = []
        for dim in mart_plan.dimensions:
            dimension_info.append(f"- {dim.name}: {', '.join(dim.attributes[:5])}")

        prompt = f"""Generate {num_queries} KPI queries for the following data mart:

**Business Requirements:** {business_requirements}

**Target Schema:** {mart_plan.target_schema}

**Fact Tables:**
{chr(10).join(fact_info)}

**Dimension Tables:**
{chr(10).join(dimension_info)}

Generate queries that cover:
1. Time-based trends (daily/monthly/yearly)
2. Top performers (products, customers, etc.)
3. Comparative analysis (period-over-period)
4. Key metrics with drill-down capability
5. Performance indicators with thresholds

Return the queries in this JSON format:
[
  {{
    "name": "Query Name",
    "description": "What this KPI shows",
    "category": "Sales/Inventory/Customer/etc",
    "query": "SELECT...",
    "visualization": "line_chart/bar_chart/table/kpi_card"
  }}
]

Focus on queries that directly address the business requirements.
Use proper JOIN syntax and include appropriate WHERE clauses for performance."""

        return prompt

    def parse_kpi_response(self, response_text: str) -> List[Dict[str, str]]:
        """Parse the KPI query response from the LLM"""
        try:
            # Extract JSON from response
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_text = response_text[start:end].strip()
            elif "[" in response_text:
                # Find the JSON array
                start = response_text.find("[")
                end = response_text.rfind("]") + 1
                json_text = response_text[start:end]
            else:
                json_text = response_text

            queries = json.loads(json_text)
            return queries

        except Exception as e:
            logger.warning(f"Failed to parse KPI response: {e}")
            return []

    def generate_fallback_queries(self, mart_plan: MartPlan) -> List[Dict[str, str]]:
        """Generate basic fallback queries if AI generation fails"""
        queries = []

        if mart_plan.facts:
            fact = mart_plan.facts[0]
            schema = mart_plan.target_schema

            # Query 1: Total metrics
            if fact.measures:
                measure_list = ', '.join([f"SUM({m.name}) as total_{m.name}" for m in fact.measures[:3]])
                queries.append({
                    'name': 'Overall Metrics Summary',
                    'description': 'High-level summary of key metrics',
                    'category': 'Overview',
                    'query': f"SELECT {measure_list} FROM {schema}.{fact.name};",
                    'visualization': 'kpi_card'
                })

            # Query 2: Time series
            if any('date' in col for col in fact.dimension_keys):
                date_col = next((col for col in fact.dimension_keys if 'date' in col), 'order_date')
                queries.append({
                    'name': 'Monthly Trend',
                    'description': 'Monthly trend of key metrics',
                    'category': 'Trends',
                    'query': f"""SELECT
    DATE_TRUNC('month', {date_col}) as month,
    SUM({fact.measures[0].name if fact.measures else '1'}) as total
FROM {schema}.{fact.name}
WHERE {date_col} >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY DATE_TRUNC('month', {date_col})
ORDER BY month;""",
                    'visualization': 'line_chart'
                })

            # Query 3: Top N
            if mart_plan.dimensions:
                dim = mart_plan.dimensions[0]
                queries.append({
                    'name': f'Top 10 by Revenue',
                    'description': f'Top performing items',
                    'category': 'Rankings',
                    'query': f"""SELECT
    d.{dim.attributes[0] if dim.attributes else dim.key_column},
    SUM(f.{fact.measures[0].name if fact.measures else '1'}) as total
FROM {schema}.{fact.name} f
JOIN {schema}.{dim.name} d ON f.{dim.key_column} = d.{dim.key_column}
GROUP BY d.{dim.attributes[0] if dim.attributes else dim.key_column}
ORDER BY total DESC
LIMIT 10;""",
                    'visualization': 'bar_chart'
                })

        return queries

    def validate_query(self, query: str, conn) -> Tuple[bool, Optional[str]]:
        """Validate a query by running EXPLAIN"""
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"EXPLAIN {query}")
                return True, None
        except Exception as e:
            return False, str(e)

    def optimize_query(self, query: str, mart_plan: MartPlan) -> str:
        """Optimize a query using GPT-5-mini"""
        prompt = f"""Optimize this SQL query for performance:

{query}

Schema: {mart_plan.target_schema}
Available indexes: {[idx.name for idx in (mart_plan.indexes or [])]}

Provide only the optimized query, no explanation."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a SQL optimization expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            optimized = response.choices[0].message.content.strip()

            # Clean up the response
            if "```sql" in optimized:
                start = optimized.find("```sql") + 6
                end = optimized.find("```", start)
                optimized = optimized[start:end].strip()
            elif "```" in optimized:
                start = optimized.find("```") + 3
                end = optimized.find("```", start)
                optimized = optimized[start:end].strip()

            return optimized

        except Exception as e:
            logger.warning(f"Failed to optimize query: {e}")
            return query

    def generate_dashboard_layout(self, queries: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate a suggested dashboard layout for the KPIs"""
        layout = {
            'title': 'KPI Dashboard',
            'rows': []
        }

        # Group queries by category
        categories = {}
        for query in queries:
            category = query.get('category', 'Other')
            if category not in categories:
                categories[category] = []
            categories[category].append(query)

        # Create layout rows
        for category, category_queries in categories.items():
            row = {
                'title': category,
                'widgets': []
            }

            for q in category_queries:
                widget = {
                    'name': q['name'],
                    'type': q.get('visualization', 'table'),
                    'query': q['query'],
                    'width': 6 if q.get('visualization') in ['line_chart', 'bar_chart'] else 3
                }
                row['widgets'].append(widget)

            layout['rows'].append(row)

        return layout

def main():
    """Main execution for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate KPI queries for a mart')
    parser.add_argument('plan_file', help='Path to mart plan JSON file')
    parser.add_argument('--requirements', '-r', help='Business requirements text', default='Generate comprehensive KPIs')
    parser.add_argument('--count', '-n', type=int, default=5, help='Number of queries to generate')
    parser.add_argument('--output', '-o', help='Output file for queries')
    parser.add_argument('--validate', action='store_true', help='Validate queries against database')
    parser.add_argument('--optimize', action='store_true', help='Optimize queries for performance')
    args = parser.parse_args()

    # Load mart plan
    with open(args.plan_file, 'r') as f:
        plan_data = json.load(f)
        mart_plan = MartPlan(**plan_data)

    # Create generator
    generator = KPIGenerator()

    print(f"🎯 Generating {args.count} KPI queries...")
    queries = generator.generate_kpi_queries(mart_plan, args.requirements, args.count)

    if not queries:
        print("❌ Failed to generate queries")
        sys.exit(1)

    print(f"✅ Generated {len(queries)} KPI queries")

    # Validate queries if requested
    if args.validate:
        print("\n🔍 Validating queries...")
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            print("⚠️  DATABASE_URL not set, skipping validation")
        else:
            conn = psycopg2.connect(db_url)
            for i, query in enumerate(queries):
                valid, error = generator.validate_query(query['query'], conn)
                if valid:
                    print(f"  ✅ {query['name']}: Valid")
                else:
                    print(f"  ❌ {query['name']}: {error}")
            conn.close()

    # Optimize queries if requested
    if args.optimize:
        print("\n⚡ Optimizing queries...")
        for query in queries:
            query['query'] = generator.optimize_query(query['query'], mart_plan)
        print("✅ Queries optimized")

    # Display queries
    print("\n" + "=" * 60)
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. {query['name']}")
        print(f"   Category: {query.get('category', 'General')}")
        print(f"   Description: {query.get('description', '')}")
        print(f"   Visualization: {query.get('visualization', 'table')}")
        print(f"\n{query['query']}\n")
        print("-" * 60)

    # Generate dashboard layout
    layout = generator.generate_dashboard_layout(queries)
    print(f"\n📊 Suggested Dashboard Layout:")
    for row in layout['rows']:
        print(f"  Row: {row['title']}")
        for widget in row['widgets']:
            print(f"    - {widget['name']} ({widget['type']})")

    # Save to file if requested
    if args.output:
        output_data = {
            'queries': queries,
            'dashboard_layout': layout
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Queries saved to {args.output}")

if __name__ == "__main__":
    main()