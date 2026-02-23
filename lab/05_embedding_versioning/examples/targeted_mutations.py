#!/usr/bin/env python3
"""Apply targeted mutations to the 10 already-embedded articles."""
import psycopg

conn = psycopg.connect('postgresql://postgres@localhost:5432/wikipedia')
cur = conn.cursor()

cur.execute('SELECT DISTINCT article_id FROM article_embeddings_versioned WHERE is_current = true ORDER BY article_id')
ids = [r[0] for r in cur.fetchall()]
print(f'Embedded articles: {ids}')

# Tiny typo on first 5 (should SKIP)
for aid in ids[:5]:
    cur.execute('UPDATE articles SET content = content || %s WHERE id = %s', ('.', aid))
    print(f'  Article {aid}: appended period (typo fix)')

# Major paragraph on next 3 (should EMBED)
big_text = (
    '\n\nThis section has been completely revised with substantial new '
    'research findings, updated methodologies, and comprehensive analysis '
    'spanning multiple disciplines. The previous content was outdated '
    'and has been replaced.'
)
for aid in ids[5:8]:
    cur.execute('UPDATE articles SET content = content || %s WHERE id = %s', (big_text, aid))
    print(f'  Article {aid}: appended major paragraph')

# Rewrite second half on last 2 (should EMBED)
rewrite = (
    '\n\nThe remainder of this article has been entirely rewritten to '
    'reflect modern understanding and current scholarly consensus on the topic.'
)
for aid in ids[8:]:
    cur.execute('UPDATE articles SET content = left(content, length(content)/2) || %s WHERE id = %s', (rewrite, aid))
    print(f'  Article {aid}: rewrote second half')

conn.commit()
conn.close()
print('Done - 10 targeted mutations applied')
