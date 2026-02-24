#!/usr/bin/env python3
"""
Lab 06 — 01_load_wikipedia.py
Load Wikipedia articles from the CSV zip into PostgreSQL.
Populates text columns, category, and word_count. Resume-safe.
"""

import csv
import io
import sys
import zipfile

import psycopg2

import config


def extract_category(url: str) -> str:
    """Derive a simple category from the Wikipedia URL."""
    if not url:
        return "Unknown"
    parts = url.strip("/").split("/")
    # URLs look like https://en.wikipedia.org/wiki/Category:...
    # or just https://en.wikipedia.org/wiki/Article_Name
    # We use a simplistic heuristic on the title words
    return "General"


def word_count(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def load_articles(conn):
    """Load articles from the Wikipedia CSV zip into the articles table."""
    zip_path = config.WIKIPEDIA_ZIP
    if not zip_path.exists():
        print(f"ERROR: Wikipedia zip not found at {zip_path}")
        print("Download it from: https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip")
        sys.exit(1)

    cur = conn.cursor()

    # Check how many rows already exist (resume support)
    cur.execute("SELECT COUNT(*) FROM articles;")
    existing = cur.fetchone()[0]
    if existing > 0:
        print(f"Table already has {existing} rows. Skipping load (resume-safe).")
        print("To reload, TRUNCATE articles first.")
        cur.close()
        return existing

    print(f"Opening {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = config.WIKIPEDIA_CSV
        if csv_name not in zf.namelist():
            # Try to find a CSV in the zip
            csvs = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csvs:
                print("ERROR: No CSV file found in the zip archive.")
                sys.exit(1)
            csv_name = csvs[0]
            print(f"Using CSV: {csv_name}")

        with zf.open(csv_name) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))

            batch = []
            total = 0

            for row in reader:
                article_id = int(row.get("id", 0))
                url = row.get("url", "")
                title = row.get("title", "")
                content = row.get("text", "")
                category = extract_category(url)
                wc = word_count(content)

                batch.append((article_id, url, title, content, category, wc))

                if len(batch) >= 500:
                    _insert_batch(cur, batch)
                    conn.commit()
                    total += len(batch)
                    print(f"  Loaded {total} articles...")
                    batch = []

            if batch:
                _insert_batch(cur, batch)
                conn.commit()
                total += len(batch)

    cur.close()
    print(f"Loaded {total} articles into the articles table.")
    return total


def _insert_batch(cur, batch):
    """Insert a batch of articles."""
    args_str = ",".join(
        cur.mogrify("(%s, %s, %s, %s, %s, %s)", row).decode("utf-8")
        for row in batch
    )
    cur.execute(
        f"INSERT INTO articles (id, url, title, content, category, word_count) "
        f"VALUES {args_str} "
        f"ON CONFLICT (id) DO NOTHING;"
    )


def assign_categories(conn):
    """
    Assign more meaningful categories based on content keywords.
    This runs after initial load to populate the category column
    with values useful for filtered search demos.
    """
    cur = conn.cursor()

    categories = [
        ("Science",     ["physics", "chemistry", "biology", "scientific", "experiment", "theory", "molecule"]),
        ("Technology",  ["computer", "software", "technology", "internet", "digital", "algorithm", "programming"]),
        ("History",     ["war", "century", "empire", "kingdom", "ancient", "historical", "dynasty"]),
        ("Geography",   ["country", "city", "river", "mountain", "island", "population", "continent"]),
        ("Arts",        ["music", "film", "painting", "artist", "novel", "literature", "theater"]),
        ("Sports",      ["football", "olympic", "championship", "player", "team", "league", "tournament"]),
        ("Politics",    ["government", "president", "election", "political", "parliament", "democracy"]),
        ("Mathematics", ["mathematical", "equation", "theorem", "algebra", "geometry", "calculus"]),
    ]

    for cat, keywords in categories:
        pattern = "|".join(keywords)
        cur.execute(
            "UPDATE articles SET category = %s "
            "WHERE category = 'General' "
            "AND content ~* %s;",
            (cat, pattern)
        )
        updated = cur.rowcount
        if updated > 0:
            print(f"  Assigned '{cat}' to {updated} articles")

    conn.commit()
    cur.close()


def main():
    print(f"Connecting to {config.DATABASE_URL}...")
    conn = psycopg2.connect(config.DATABASE_URL)

    total = load_articles(conn)

    if total > 0:
        print("\nAssigning categories based on content analysis...")
        assign_categories(conn)

    # Final stats
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM articles;")
    count = cur.fetchone()[0]
    cur.execute("SELECT category, COUNT(*) FROM articles GROUP BY category ORDER BY COUNT(*) DESC;")
    cats = cur.fetchall()
    cur.close()

    print(f"\nFinal: {count} articles loaded")
    print("Category distribution:")
    for cat, cnt in cats:
        print(f"  {cat}: {cnt}")

    conn.close()


if __name__ == "__main__":
    main()
