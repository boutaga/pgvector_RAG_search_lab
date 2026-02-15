#!/usr/bin/env python3
"""
Lab 5 — Demo: Change Significance Detection

Applies known changes to articles and shows the change detector's
EMBED vs SKIP decisions with similarity ratios and structural analysis.

Usage:
    python examples/demo_change_significance.py
    python examples/demo_change_significance.py --article-id 42
"""

import argparse
import json
import logging
import os
import sys

import psycopg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from change_detector import should_reembed, analyze_structural_change, text_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DEMO] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/wikipedia",
)


def demo_change_significance(db_url: str, article_id: int = None):
    """Demonstrate change significance analysis on real article content."""
    with psycopg.connect(db_url) as conn:
        # Pick an article
        if article_id is None:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, title, content FROM articles
                    WHERE length(content) > 500
                    ORDER BY random() LIMIT 1
                """)
                row = cur.fetchone()
        else:
            with conn.cursor() as cur:
                cur.execute("SELECT id, title, content FROM articles WHERE id = %s", (article_id,))
                row = cur.fetchone()

        if not row:
            print("No article found")
            return

        article_id, title, original_content = row

        print(f"\n{'=' * 70}")
        print(f"  Demo: Change Significance Detection")
        print(f"  Article: [{article_id}] {title[:50]}")
        print(f"  Original length: {len(original_content)} chars")
        print(f"{'=' * 70}")

        # Define test mutations
        mutations = [
            (
                "Typo Fix (1 char change)",
                original_content[:100] + "X" + original_content[101:],
            ),
            (
                "Whitespace Cleanup",
                original_content.replace("  ", " "),
            ),
            (
                "Append Sentence",
                original_content + " This article has been reviewed for accuracy.",
            ),
            (
                "Add New Paragraph",
                original_content + "\n\nA recent comprehensive study has revealed "
                "significant new findings related to this topic. The research, "
                "conducted by an international team, provides compelling evidence "
                "that challenges some of the previously held assumptions.",
            ),
            (
                "Rewrite Middle Section (~20%)",
                rewrite_middle(original_content),
            ),
            (
                "Major Rewrite (keep intro only)",
                major_rewrite(original_content),
            ),
        ]

        print(f"\n{'  Change Type':<35} {'Decision':>8} {'Similarity':>11} {'Para Sim':>9} {'Len Ratio':>10}")
        print(f"  {'-' * 73}")

        for label, modified_content in mutations:
            decision, similarity, structural = should_reembed(original_content, modified_content)
            para_sim = structural.get("paragraph_similarity", "-")
            len_ratio = structural.get("length_ratio", "-")

            print(f"  {label:<35} {decision:>8} {similarity:>10.4f} "
                  f"{para_sim if isinstance(para_sim, str) else f'{para_sim:.4f}':>9} "
                  f"{len_ratio if isinstance(len_ratio, str) else f'{len_ratio:.4f}':>10}")

        # Detailed analysis for the major rewrite
        print(f"\n{'=' * 70}")
        print(f"  Detailed Structural Analysis: Major Rewrite")
        print(f"{'=' * 70}")
        _, _, last_modified = mutations[-1]
        structural = analyze_structural_change(original_content, major_rewrite(original_content))
        for key, val in structural.items():
            print(f"  {key:<25}: {val}")

    print(f"\n{'=' * 70}\n")


def rewrite_middle(content: str) -> str:
    """Replace the middle ~20% of content."""
    lines = content.split("\n")
    if len(lines) < 5:
        return content + "\n\nRewritten section with significant changes."
    start = len(lines) // 3
    end = start + max(1, len(lines) // 5)
    replacement = [
        "This section contains substantially revised content.",
        "The information presented here has been updated based on new sources.",
        "Multiple aspects of the previous version have been corrected.",
        "Additional nuance has been added to better represent the complexity.",
    ]
    lines[start:end] = replacement
    return "\n".join(lines)


def major_rewrite(content: str) -> str:
    """Keep only the first paragraph, replace the rest."""
    paragraphs = content.split("\n\n")
    first = paragraphs[0] if paragraphs else ""
    return (
        f"{first}\n\n"
        "This article has been completely rewritten to provide a modern, "
        "comprehensive treatment of the subject. The previous version contained "
        "outdated information and has been replaced with current research findings.\n\n"
        "Key topics now covered include recent developments, revised historical "
        "context, updated statistical data, and a broader international perspective. "
        "The article now reflects the scholarly consensus as of the current year."
    )


def main():
    parser = argparse.ArgumentParser(description="Change significance demo")
    parser.add_argument("--article-id", type=int, help="Specific article to analyze")
    parser.add_argument("--db-url", type=str, default=DB_URL, help="Database URL")
    args = parser.parse_args()

    demo_change_significance(args.db_url, args.article_id)


if __name__ == "__main__":
    main()
