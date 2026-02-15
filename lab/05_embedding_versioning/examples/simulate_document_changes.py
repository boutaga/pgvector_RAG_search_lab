#!/usr/bin/env python3
"""
Lab 5 — Simulate Document Changes

Applies random mutations to Wikipedia articles to trigger the
embedding versioning pipeline. Mutation types mirror real-world changes:

- typo_fix (40%):       Minor character-level changes → should SKIP
- metadata_only (20%):  Title change only → should SKIP (content unchanged)
- paragraph_add (25%):  New paragraph appended → should EMBED
- section_rewrite (10%): Replace a section → should EMBED
- major_rewrite (5%):   Complete content replacement → should EMBED

Usage:
    python examples/simulate_document_changes.py --count 50
    python examples/simulate_document_changes.py --count 10 --type typo_fix
    python examples/simulate_document_changes.py --count 20 --dry-run
"""

import argparse
import logging
import os
import random
import string
import sys

import psycopg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SIMULATE] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/wikipedia",
)

MUTATION_WEIGHTS = {
    "typo_fix": 40,
    "metadata_only": 20,
    "paragraph_add": 25,
    "section_rewrite": 10,
    "major_rewrite": 5,
}


# ---------------------------------------------------------------------------
# Mutation functions
# ---------------------------------------------------------------------------

def mutate_typo_fix(content: str) -> str:
    """Introduce 1-3 small typo fixes (swap/add/remove a character)."""
    chars = list(content)
    num_changes = random.randint(1, 3)
    for _ in range(num_changes):
        if len(chars) < 10:
            break
        pos = random.randint(10, len(chars) - 1)
        action = random.choice(["swap", "insert", "delete"])
        if action == "swap" and pos < len(chars):
            chars[pos] = random.choice(string.ascii_lowercase)
        elif action == "insert":
            chars.insert(pos, random.choice(string.ascii_lowercase))
        elif action == "delete":
            chars.pop(pos)
    return "".join(chars)


def mutate_metadata_only(title: str) -> str:
    """Change only the title (metadata), not the content."""
    suffixes = [" (updated)", " (revised)", " (v2)", " (corrected)"]
    return title + random.choice(suffixes)


def mutate_paragraph_add(content: str) -> str:
    """Add a new paragraph at the end."""
    new_paragraphs = [
        "\n\nRecent developments in this area have led to significant advances. "
        "Researchers continue to explore new approaches and methodologies that "
        "could reshape our understanding of the subject.",

        "\n\nFurther analysis reveals additional complexity in the topic. "
        "Multiple studies have confirmed these findings through independent "
        "verification and peer review processes.",

        "\n\nThe implications of these findings extend beyond the immediate "
        "scope of the original research. Interdisciplinary collaboration has "
        "opened new avenues for investigation and application.",
    ]
    return content + random.choice(new_paragraphs)


def mutate_section_rewrite(content: str) -> str:
    """Replace a middle section of the content (roughly 20%)."""
    lines = content.split("\n")
    if len(lines) < 5:
        return mutate_paragraph_add(content)

    # Replace ~20% of lines in the middle
    start = len(lines) // 3
    end = start + max(1, len(lines) // 5)

    replacement = [
        "This section has been updated with new information.",
        "The previous content has been revised to reflect current understanding.",
        "Additional context and recent findings have been incorporated.",
        "Several key points have been clarified and expanded upon.",
    ]

    lines[start:end] = replacement
    return "\n".join(lines)


def mutate_major_rewrite(content: str) -> str:
    """Replace most of the content (keep only first paragraph)."""
    paragraphs = content.split("\n\n")
    first = paragraphs[0] if paragraphs else ""
    new_content = (
        f"{first}\n\n"
        "This article has undergone a major revision. The previous content "
        "has been substantially rewritten to provide a more comprehensive "
        "and accurate treatment of the subject matter.\n\n"
        "The scope of coverage has been expanded to include recent developments, "
        "alternative perspectives, and updated references. Key sections have been "
        "reorganized for clarity and logical flow.\n\n"
        "Notable changes include updated statistical data, revised historical "
        "timelines, and the inclusion of recent scholarly publications that "
        "shed new light on the topic."
    )
    return new_content


# ---------------------------------------------------------------------------
# Apply mutations
# ---------------------------------------------------------------------------

def select_random_articles(conn, count: int) -> list[tuple]:
    """Select random articles from the database."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, title, content
            FROM articles
            ORDER BY random()
            LIMIT %s
        """, (count,))
        return cur.fetchall()


def choose_mutation_type(forced_type: str = None) -> str:
    """Choose a mutation type based on weights (or force a specific type)."""
    if forced_type:
        return forced_type
    types = list(MUTATION_WEIGHTS.keys())
    weights = list(MUTATION_WEIGHTS.values())
    return random.choices(types, weights=weights, k=1)[0]


def apply_mutation(conn, article_id: int, title: str, content: str,
                   mutation_type: str, dry_run: bool) -> dict:
    """Apply a mutation to an article and return details."""
    result = {
        "article_id": article_id,
        "title": title[:50],
        "mutation_type": mutation_type,
        "applied": False,
    }

    if mutation_type == "typo_fix":
        new_content = mutate_typo_fix(content)
        new_title = title
    elif mutation_type == "metadata_only":
        new_content = content  # content unchanged
        new_title = mutate_metadata_only(title)
    elif mutation_type == "paragraph_add":
        new_content = mutate_paragraph_add(content)
        new_title = title
    elif mutation_type == "section_rewrite":
        new_content = mutate_section_rewrite(content)
        new_title = title
    elif mutation_type == "major_rewrite":
        new_content = mutate_major_rewrite(content)
        new_title = title
    else:
        log.warning("Unknown mutation type: %s", mutation_type)
        return result

    result["content_changed"] = new_content != content
    result["title_changed"] = new_title != title

    if dry_run:
        log.info("[DRY RUN] Article %d (%s): %s (content_changed=%s)",
                 article_id, title[:30], mutation_type, result["content_changed"])
        return result

    with conn.cursor() as cur:
        if new_content != content and new_title != title:
            cur.execute(
                "UPDATE articles SET title = %s, content = %s WHERE id = %s",
                (new_title, new_content, article_id),
            )
        elif new_content != content:
            cur.execute(
                "UPDATE articles SET content = %s WHERE id = %s",
                (new_content, article_id),
            )
        elif new_title != title:
            cur.execute(
                "UPDATE articles SET title = %s WHERE id = %s",
                (new_title, article_id),
            )

    conn.commit()
    result["applied"] = True
    log.info("Article %d (%s): %s applied", article_id, title[:30], mutation_type)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Simulate document changes")
    parser.add_argument("--count", type=int, default=10, help="Number of articles to mutate")
    parser.add_argument("--type", type=str, choices=list(MUTATION_WEIGHTS.keys()),
                        help="Force a specific mutation type")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--db-url", type=str, default=DB_URL, help="Database URL")
    args = parser.parse_args()

    with psycopg.connect(args.db_url) as conn:
        articles = select_random_articles(conn, args.count)
        log.info("Selected %d articles for mutation", len(articles))

        stats = {}
        for article_id, title, content in articles:
            mutation_type = choose_mutation_type(args.type)
            result = apply_mutation(conn, article_id, title, content,
                                   mutation_type, args.dry_run)
            stats[mutation_type] = stats.get(mutation_type, 0) + 1

        print("\nMutation Summary:")
        print("-" * 40)
        for mtype, count in sorted(stats.items()):
            print(f"  {mtype:<20} {count:>4}")
        print(f"  {'TOTAL':<20} {sum(stats.values()):>4}")

        if not args.dry_run:
            print("\nCheck the queue:")
            print("  SELECT status, count(*) FROM embedding_queue GROUP BY status;")


if __name__ == "__main__":
    main()
