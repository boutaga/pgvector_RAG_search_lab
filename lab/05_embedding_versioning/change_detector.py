#!/usr/bin/env python3
"""
Lab 5 — Change Significance Detector

Analyzes whether a document change is significant enough to warrant
re-embedding, or if it can be skipped (e.g., typo fixes, whitespace).

Uses difflib for text similarity and structural analysis to make
EMBED vs SKIP decisions.

Usage:
    python change_detector.py --analyze-queue        # process pending queue items
    python change_detector.py --article-id 42        # analyze a specific article
    python change_detector.py --threshold 0.92       # custom similarity threshold
"""

import argparse
import difflib
import json
import logging
import os
import re
import sys

import psycopg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DETECTOR] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/wikipedia",
)

# Above this similarity ratio, changes are considered trivial → SKIP
DEFAULT_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Text Analysis
# ---------------------------------------------------------------------------

def text_similarity(old_text: str, new_text: str) -> float:
    """Compute similarity ratio between two texts using SequenceMatcher."""
    if not old_text and not new_text:
        return 1.0
    if not old_text or not new_text:
        return 0.0
    return difflib.SequenceMatcher(None, old_text, new_text).ratio()


def analyze_structural_change(old_text: str, new_text: str) -> dict:
    """Analyze changes at paragraph and heading level.

    Returns a dict with structural change metrics.
    """
    def extract_paragraphs(text: str) -> list[str]:
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    def extract_headings(text: str) -> list[str]:
        """Extract lines that look like headings (== Heading == or # Heading)."""
        patterns = [
            r'^={2,}\s*(.+?)\s*={2,}$',   # MediaWiki: == Heading ==
            r'^#{1,6}\s+(.+)$',             # Markdown: # Heading
        ]
        headings = []
        for line in text.split("\n"):
            for pat in patterns:
                m = re.match(pat, line.strip())
                if m:
                    headings.append(m.group(1).strip())
                    break
        return headings

    old_paragraphs = extract_paragraphs(old_text)
    new_paragraphs = extract_paragraphs(new_text)
    old_headings = extract_headings(old_text)
    new_headings = extract_headings(new_text)

    # Paragraph-level diff
    para_matcher = difflib.SequenceMatcher(None, old_paragraphs, new_paragraphs)
    para_similarity = para_matcher.ratio()

    # Heading changes
    headings_added = set(new_headings) - set(old_headings)
    headings_removed = set(old_headings) - set(new_headings)

    # Length change
    len_old = len(old_text)
    len_new = len(new_text)
    length_ratio = min(len_old, len_new) / max(len_old, len_new) if max(len_old, len_new) > 0 else 1.0

    return {
        "paragraph_similarity": round(para_similarity, 4),
        "paragraphs_old": len(old_paragraphs),
        "paragraphs_new": len(new_paragraphs),
        "headings_added": list(headings_added),
        "headings_removed": list(headings_removed),
        "heading_changes": len(headings_added) + len(headings_removed),
        "length_ratio": round(length_ratio, 4),
        "char_diff": len_new - len_old,
    }


def should_reembed(old_text: str, new_text: str, threshold: float = DEFAULT_THRESHOLD) -> tuple[str, float, dict]:
    """Decide whether to EMBED or SKIP based on change significance.

    Returns:
        (decision, similarity, structural_details)
    """
    similarity = text_similarity(old_text, new_text)
    structural = analyze_structural_change(old_text, new_text)

    # Decision logic
    # 1. If text is very similar (above threshold), SKIP
    if similarity >= threshold:
        return "SKIP", similarity, structural

    # 2. If headings changed, always EMBED (structural change)
    if structural["heading_changes"] > 0:
        return "EMBED", similarity, structural

    # 3. If length changed dramatically (>20%), EMBED
    if structural["length_ratio"] < 0.80:
        return "EMBED", similarity, structural

    # 4. If paragraph structure changed significantly, EMBED
    if structural["paragraph_similarity"] < 0.85:
        return "EMBED", similarity, structural

    # 5. Moderate change — EMBED to be safe
    if similarity < threshold:
        return "EMBED", similarity, structural

    return "SKIP", similarity, structural


# ---------------------------------------------------------------------------
# Old content reconstruction
# ---------------------------------------------------------------------------

def get_old_content(conn, article_id: int) -> str | None:
    """Reconstruct the previous content from stored chunk_text in
    article_embeddings_versioned.

    Returns concatenated text or None if no embeddings exist.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_text FROM article_embeddings_versioned
            WHERE article_id = %s AND is_current = true
            ORDER BY chunk_index
        """, (article_id,))
        rows = cur.fetchall()

    if not rows:
        return None

    return "\n".join(text for (text,) in rows if text)


# ---------------------------------------------------------------------------
# Queue analysis
# ---------------------------------------------------------------------------

def analyze_queue_item(conn, article_id: int, threshold: float) -> dict:
    """Analyze a single queued article and log the decision."""
    # Get current content
    with conn.cursor() as cur:
        cur.execute("SELECT content, content_hash FROM articles WHERE id = %s", (article_id,))
        row = cur.fetchone()

    if not row:
        log.warning("Article %d not found", article_id)
        return {"article_id": article_id, "decision": "SKIP", "reason": "not_found"}

    new_content, new_hash = row

    # Get old content from stored embeddings
    old_content = get_old_content(conn, article_id)

    if old_content is None:
        # No previous embeddings — must embed
        decision, similarity, structural = "EMBED", 0.0, {"reason": "no_previous_embeddings"}
    else:
        decision, similarity, structural = should_reembed(old_content, new_content, threshold)

    # Log the decision
    with conn.cursor() as cur:
        # Get old hash from current embeddings
        cur.execute("""
            SELECT source_hash FROM article_embeddings_versioned
            WHERE article_id = %s AND is_current = true
            LIMIT 1
        """, (article_id,))
        old_hash_row = cur.fetchone()
        old_hash = old_hash_row[0] if old_hash_row else None

        cur.execute("""
            INSERT INTO embedding_change_log
                (article_id, old_hash, new_hash, change_type, decision, similarity, details)
            VALUES (%s, %s, %s, 'content_update', %s, %s, %s)
        """, (article_id, old_hash, new_hash, decision, similarity, json.dumps(structural)))

    conn.commit()

    log.info("Article %d: %s (similarity=%.4f)", article_id, decision, similarity)
    return {
        "article_id": article_id,
        "decision": decision,
        "similarity": similarity,
        "structural": structural,
    }


def analyze_pending_queue(db_url: str, threshold: float, limit: int = 100):
    """Process pending queue items: analyze significance, mark SKIPs as completed."""
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, article_id FROM embedding_queue
                WHERE status = 'pending'
                ORDER BY priority, queued_at
                LIMIT %s
            """, (limit,))
            items = cur.fetchall()

        if not items:
            log.info("No pending items in queue")
            return

        log.info("Analyzing %d pending queue items (threshold=%.2f)", len(items), threshold)

        stats = {"EMBED": 0, "SKIP": 0}

        for queue_id, article_id in items:
            result = analyze_queue_item(conn, article_id, threshold)
            decision = result["decision"]
            stats[decision] = stats.get(decision, 0) + 1

            if decision == "SKIP":
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE embedding_queue
                        SET status = 'skipped', completed_at = now()
                        WHERE id = %s
                    """, (queue_id,))
                conn.commit()

        log.info("Results: %d EMBED, %d SKIP", stats["EMBED"], stats["SKIP"])


# ---------------------------------------------------------------------------
# Single article analysis
# ---------------------------------------------------------------------------

def analyze_single(db_url: str, article_id: int, threshold: float):
    """Analyze a single article (for debugging/demo)."""
    with psycopg.connect(db_url) as conn:
        result = analyze_queue_item(conn, article_id, threshold)
        print(json.dumps(result, indent=2, default=str))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Change significance detector")
    parser.add_argument("--analyze-queue", action="store_true", help="Analyze all pending queue items")
    parser.add_argument("--article-id", type=int, help="Analyze a specific article")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Similarity threshold (default 0.95)")
    parser.add_argument("--limit", type=int, default=100, help="Max queue items to analyze")
    parser.add_argument("--db-url", type=str, default=DB_URL, help="Database URL")
    args = parser.parse_args()

    if args.article_id:
        analyze_single(args.db_url, args.article_id, args.threshold)
    elif args.analyze_queue:
        analyze_pending_queue(args.db_url, args.threshold, args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
