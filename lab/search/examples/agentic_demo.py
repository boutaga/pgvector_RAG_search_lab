#!/usr/bin/env python3
"""
Agentic RAG Demonstration Script

This script demonstrates the Agentic RAG approach where an LLM agent
autonomously decides when to retrieve information from the database.

The script compares different query types to show how the agent makes decisions:
1. Simple queries - Agent may answer directly
2. Complex queries - Agent uses search tool
3. Specific Wikipedia queries - Agent definitely needs search

Run this to see Agentic RAG in action!
"""

import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from lab.search.agentic_search import AgenticSearchEngine
from lab.core.database import DatabaseService
from lab.core.config import ConfigService


def print_separator(char='=', length=80):
    """Print a separator line."""
    print(char * length)


def print_header(title):
    """Print a formatted header."""
    print_separator()
    print(f"  {title}")
    print_separator()


def print_result(result, show_sources=True):
    """Print formatted search result."""
    # Decision indicator
    if result['decision'] == 'search':
        decision_icon = '🔄'
        decision_text = 'AGENT USED SEARCH TOOL'
    elif result['decision'] == 'direct':
        decision_icon = '⚡'
        decision_text = 'AGENT ANSWERED DIRECTLY'
    else:
        decision_icon = '❓'
        decision_text = f"DECISION: {result['decision'].upper()}"

    print(f"\n{decision_icon} {decision_text}")
    print(f"   Tool Used: {result['tool_used']}")
    print(f"   Search Count: {result['search_count']}")
    print(f"   Cost: ${result['cost']:.4f}")
    print(f"   Sources Retrieved: {result['num_results']}")

    print(f"\n{'─' * 80}")
    print("ANSWER:")
    print('─' * 80)
    print(result['answer'])

    if show_sources and result['sources']:
        print(f"\n{'─' * 80}")
        print(f"SOURCES ({len(result['sources'])} retrieved):")
        print('─' * 80)
        for i, source in enumerate(result['sources'], 1):
            title = source.get('metadata', {}).get('title', 'Unknown')
            print(f"\n[{i}] {title}")
            print(f"    {source['content']}")
            print(f"    Similarity Score: {source['score']:.4f}")


def main():
    """Run Agentic RAG demonstration."""
    print_header("🤖 AGENTIC RAG DEMONSTRATION")
    print("\nThis demo shows how an LLM agent decides when to retrieve information.")
    print("Watch how the agent handles different types of queries:\n")
    print("  ⚡ Direct Answer - Agent knows the answer without searching")
    print("  🔄 Search Tool - Agent retrieves information from database\n")

    # Initialize
    print("Initializing Agentic RAG system...")
    try:
        config = ConfigService()
        db_service = DatabaseService(
            config.database.connection_string,
            config.database.min_connections,
            config.database.max_connections
        )
        engine = AgenticSearchEngine(db_service, config, source='wikipedia')
        print("✅ System initialized successfully!\n")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return

    # Test queries demonstrating different agent behaviors
    test_cases = [
        {
            'category': 'Simple Math (Likely Direct Answer)',
            'query': 'What is 2 + 2?',
            'expected': 'Agent should answer directly without searching'
        },
        {
            'category': 'General Knowledge (May Answer Directly)',
            'query': 'What is the capital of France?',
            'expected': 'Agent might answer directly or search for confirmation'
        },
        {
            'category': 'Wikipedia-Specific Query (Should Search)',
            'query': 'Explain PostgreSQL MVCC and how it handles concurrent transactions',
            'expected': 'Agent should use search tool for detailed information'
        },
        {
            'category': 'Complex Technical Query (Should Search)',
            'query': 'How does PostgreSQL replication work and what are the different methods?',
            'expected': 'Agent should definitely search the database'
        },
        {
            'category': 'Specific Historical Query (Should Search)',
            'query': 'What major events happened in 2004?',
            'expected': 'Agent should search for specific Wikipedia content'
        }
    ]

    results = []
    total_cost = 0.0

    # Run each test case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n{'#' * 80}")
        print(f"TEST CASE {i}/{len(test_cases)}: {test_case['category']}")
        print('#' * 80)
        print(f"\nQuery: \"{test_case['query']}\"")
        print(f"Expected Behavior: {test_case['expected']}\n")

        try:
            result = engine.search_and_answer_agentic(test_case['query'], top_k=5)
            results.append(result)
            total_cost += result['cost']

            print_result(result, show_sources=(i > 2))  # Show sources for later queries

        except Exception as e:
            print(f"❌ Error: {e}")

        input("\n[Press Enter to continue to next test case...]")

    # Summary
    print_header("📊 DEMONSTRATION SUMMARY")

    direct_count = sum(1 for r in results if r['decision'] == 'direct')
    search_count = sum(1 for r in results if r['decision'] == 'search')

    print(f"\nTotal Test Cases: {len(results)}")
    print(f"Direct Answers: {direct_count} ({direct_count/len(results)*100:.0f}%)")
    print(f"Search Tool Used: {search_count} ({search_count/len(results)*100:.0f}%)")
    print(f"Total Cost: ${total_cost:.4f}")
    print(f"Average Cost per Query: ${total_cost/len(results):.4f}")

    print("\n\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("""
1. The agent autonomously decides when to retrieve information
2. Simple queries can be answered directly, saving cost and latency
3. Complex or specific queries trigger the search tool automatically
4. The agent grounds its answers in retrieved information when searching
5. Source citations are provided for transparency

This demonstrates the power of Agentic RAG:
- More efficient than always searching (Naive RAG)
- More flexible than fixed pipelines (Hybrid/Adaptive RAG)
- Autonomous decision-making by the LLM itself
""")

    print("=" * 80)
    print("🎉 Demonstration Complete!")
    print("=" * 80)

    # Cleanup
    db_service.close()


if __name__ == "__main__":
    main()
