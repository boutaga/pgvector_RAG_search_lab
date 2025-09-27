#!/usr/bin/env python3
"""
Interactive mart planning agent.
Uses RAG + LLM to generate data mart plans from natural language business questions.
"""

import os
import sys
import json
from pathlib import Path

# Add the services directory to the path
script_dir = Path(__file__).parent
lab_dir = script_dir.parent
services_dir = lab_dir / 'services'
models_dir = lab_dir / 'models'
sys.path.insert(0, str(services_dir))
sys.path.insert(0, str(models_dir))

from mart_agent_service import MartPlanningAgent, AgentConfig

def print_mart_plan(plan, user_question):
    """Print a formatted mart plan."""
    print(f"\n📊 MART PLAN FOR: '{user_question}'")
    print("=" * 80)

    print(f"\n🎯 Target Schema: {plan.target_schema}")
    print(f"📁 Source Schema: {plan.source_schema}")

    # Print fact tables
    if plan.facts:
        print(f"\n📈 FACT TABLES ({len(plan.facts)}):")
        print("-" * 50)

        for i, fact in enumerate(plan.facts, 1):
            print(f"\n{i}. {fact.name}")
            print(f"   Description: {fact.description or 'Main fact table'}")
            print(f"   Grain: {', '.join(fact.grain)}")

            if fact.measures:
                print(f"   Measures ({len(fact.measures)}):")
                for measure in fact.measures:
                    print(f"     • {measure.name} ({measure.aggregation})")
                    print(f"       Expression: {measure.expression}")
                    if measure.description:
                        print(f"       Description: {measure.description}")

            print(f"   Dimension Keys: {', '.join(fact.dimension_keys)}")
            print(f"   Source Tables: {', '.join(fact.source_tables)}")

            if fact.join_conditions:
                print(f"   Join Conditions:")
                for join in fact.join_conditions:
                    print(f"     • {join}")

    # Print dimension tables
    if plan.dimensions:
        print(f"\n🏷️  DIMENSION TABLES ({len(plan.dimensions)}):")
        print("-" * 50)

        for i, dim in enumerate(plan.dimensions, 1):
            print(f"\n{i}. {dim.name}")
            print(f"   Description: {dim.description or 'Dimension for descriptive attributes'}")
            print(f"   Source Table: {dim.source_table}")
            print(f"   Key Column: {dim.key_column}")
            print(f"   Attributes: {', '.join(dim.attributes)}")

def save_mart_plan(plan, filename_prefix="mart_plan"):
    """Save mart plan to JSON file."""
    # Create samples directory if it doesn't exist
    samples_dir = lab_dir / 'samples'
    samples_dir.mkdir(exist_ok=True)

    # Generate filename
    timestamp = plan.plan_id or "generated"
    filename = f"{filename_prefix}_{timestamp}.json"
    filepath = samples_dir / filename

    # Save to file
    with open(filepath, 'w') as f:
        json.dump(plan.dict(), f, indent=2)

    print(f"\n💾 Mart plan saved to: {filepath}")
    return filepath

def get_sample_business_questions():
    """Return sample business questions for demonstration."""
    return [
        "What are the fastest-selling products and their revenue contribution?",
        "How can I analyze customer purchasing patterns and lifetime value?",
        "What metrics should I track for inventory turnover by category?",
        "How can I compare sales performance across different regions and time periods?",
        "What are the most profitable product categories and their trends?",
        "How can I analyze employee sales performance and commission calculations?",
        "What shipping costs and delivery performance metrics should I track?",
        "How can I identify seasonal trends in product sales?",
        "What are the top customers by revenue and order frequency?",
        "How can I analyze discount effectiveness and pricing strategies?"
    ]

def interactive_planning_loop():
    """Main interactive loop for mart planning."""
    print("=" * 80)
    print("🏗️  Interactive Data Mart Planning Agent")
    print("=" * 80)

    # Initialize agent with GPT-5 configuration
    try:
        config = AgentConfig(
            primary_model="gpt-5",
            fast_model="gpt-5-mini",
            fallback_model="gpt-4",
            temperature=0.1,
            max_tokens=3000,
            max_retries=3
        )
        agent = MartPlanningAgent(config)
        print("✓ Mart planning agent initialized with GPT-5 and GPT-5-mini")
    except Exception as e:
        print(f"✗ Error initializing agent: {e}")
        return

    # Show sample questions
    print("\n💡 Sample business questions:")
    for i, question in enumerate(get_sample_business_questions()[:5], 1):
        print(f"  {i}. {question}")

    print("\n" + "=" * 80)
    print("Ask me a business question and I'll design an optimal data mart!")
    print("Type 'samples' to see more sample questions.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 80)

    while True:
        try:
            # Get user input
            question = input("\n💼 Your business question: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 🏗️")
                break

            if question.lower() == 'samples':
                print("\n💡 Sample business questions:")
                for i, sample in enumerate(get_sample_business_questions(), 1):
                    print(f"  {i:2d}. {sample}")
                continue

            # Generate mart plan
            print(f"\n🤔 Analyzing question: '{question}'")
            print("🔍 Searching relevant metadata...")
            print("🧠 Generating optimal mart design...")
            print("..." * 20)

            mart_plan, search_results = agent.plan_mart_from_question(question)

            # Display results
            print_mart_plan(mart_plan, question)

            # Show explanation
            show_explanation = input("\nShow detailed explanation? (y/n): ").lower()
            if show_explanation in ['y', 'yes']:
                explanation = agent.explain_mart_plan(mart_plan, question)
                print(f"\n📖 EXPLANATION:\n{explanation}")

            # Show metadata used
            show_metadata = input("\nShow metadata sources used? (y/n): ").lower()
            if show_metadata in ['y', 'yes']:
                print(f"\n🔍 METADATA SOURCES ({len(search_results)}):")
                print("-" * 50)
                for i, result in enumerate(search_results[:10], 1):  # Show top 10
                    print(f"{i:2d}. [{result.metadata_type.upper()}] {result.table_name}")
                    if result.column_name:
                        print(f"     Column: {result.column_name}")
                    print(f"     Score: {result.similarity_score:.3f}")
                    print(f"     Description: {result.description}")
                    print()

            # Save plan
            save_plan = input("\nSave mart plan to file? (y/n): ").lower()
            if save_plan in ['y', 'yes']:
                # Generate plan ID
                import uuid
                mart_plan.plan_id = str(uuid.uuid4())[:8]
                save_mart_plan(mart_plan)

        except KeyboardInterrupt:
            print("\n\nGoodbye! 🏗️")
            break
        except Exception as e:
            print(f"\n✗ Error during planning: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function."""
    # Check for required environment variables
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("✗ Error: OPENAI_API_KEY environment variable is required")
        print("\nPlease set your OpenAI API key:")
        print('  export OPENAI_API_KEY="your_api_key_here"')
        sys.exit(1)

    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("✗ Error: DATABASE_URL environment variable is required")
        print("\nPlease set your database connection:")
        print('  export DATABASE_URL="postgresql://user:password@localhost/dbname"')
        sys.exit(1)

    # Check if we're running in non-interactive mode
    if len(sys.argv) > 1:
        # Non-interactive mode - plan for provided question
        question = ' '.join(sys.argv[1:])
        print(f"🏗️  Planning mart for: '{question}'")

        try:
            agent = MartPlanningAgent()
            mart_plan, search_results = agent.plan_mart_from_question(question)

            print_mart_plan(mart_plan, question)

            # Auto-save plan
            import uuid
            mart_plan.plan_id = str(uuid.uuid4())[:8]
            filepath = save_mart_plan(mart_plan)

            print(f"\n📖 Explanation:")
            explanation = agent.explain_mart_plan(mart_plan, question)
            print(explanation)

            print(f"\nNext steps:")
            print(f"1. Review the saved plan: {filepath}")
            print(f"2. Run 60_mart_executor.py to create the mart")
            print(f"3. Run 70_kpi_generator.py to generate KPI queries")

        except Exception as e:
            print(f"✗ Error during planning: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Interactive mode
        interactive_planning_loop()

if __name__ == "__main__":
    main()