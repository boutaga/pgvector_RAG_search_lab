#!/usr/bin/env python3
"""
Scan PostgreSQL database metadata and populate the catalog schema.
Extracts information from information_schema, pg_stats, and system catalogs.
"""

import os
import sys
import psycopg2
from psycopg2 import sql
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any

def get_db_connection():
    """Create a database connection using environment variables."""
    db_url = os.environ.get('DATABASE_URL')
    if db_url:
        return psycopg2.connect(db_url)
    else:
        return psycopg2.connect(
            host=os.environ.get('DB_HOST', 'localhost'),
            port=os.environ.get('DB_PORT', '5432'),
            database=os.environ.get('DB_NAME', 'postgres'),
            user=os.environ.get('DB_USER', 'postgres'),
            password=os.environ.get('DB_PASSWORD', '')
        )

def scan_table_metadata(conn, schema_name: str = 'src_northwind') -> List[Dict]:
    """Scan table-level metadata from the database."""
    query = """
    WITH table_stats AS (
        SELECT
            s.schemaname,
            s.tablename,
            COALESCE(s.n_tup_ins, 0) + COALESCE(s.n_tup_upd, 0) + COALESCE(s.n_tup_del, 0) as total_activity,
            COALESCE(s.n_live_tup, 0) as row_count,
            pg_relation_size(c.oid) as table_size
        FROM pg_stat_user_tables s
        JOIN pg_class c ON c.relname = s.tablename
        JOIN pg_namespace n ON n.nspname = s.schemaname AND n.oid = c.relnamespace
        WHERE s.schemaname = %s
    ),
    table_info AS (
        SELECT
            t.table_schema,
            t.table_name,
            t.table_type,
            obj_description(c.oid) as table_comment,
            COUNT(DISTINCT col.column_name) as column_count
        FROM information_schema.tables t
        JOIN pg_class c ON c.relname = t.table_name
        JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = t.table_schema
        LEFT JOIN information_schema.columns col
            ON col.table_schema = t.table_schema
            AND col.table_name = t.table_name
        WHERE t.table_schema = %s
        GROUP BY t.table_schema, t.table_name, t.table_type, c.oid
    ),
    pk_info AS (
        SELECT
            tc.table_schema,
            tc.table_name,
            COUNT(*) as pk_count
        FROM information_schema.table_constraints tc
        WHERE tc.table_schema = %s
            AND tc.constraint_type = 'PRIMARY KEY'
        GROUP BY tc.table_schema, tc.table_name
    ),
    fk_info AS (
        SELECT
            tc.table_schema,
            tc.table_name,
            COUNT(*) as fk_count
        FROM information_schema.table_constraints tc
        WHERE tc.table_schema = %s
            AND tc.constraint_type = 'FOREIGN KEY'
        GROUP BY tc.table_schema, tc.table_name
    ),
    index_info AS (
        SELECT
            n.nspname as schemaname,
            c.relname as tablename,
            COUNT(i.indexrelid) as index_count
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_index i ON i.indrelid = c.oid
        WHERE n.nspname = %s
            AND c.relkind = 'r'
        GROUP BY n.nspname, c.relname
    )
    SELECT
        ti.table_schema,
        ti.table_name,
        ti.table_type,
        COALESCE(ts.row_count, 0) as row_count,
        COALESCE(ts.table_size, 0) as table_size_bytes,
        ti.table_comment,
        COALESCE(pk.pk_count, 0) > 0 as has_primary_key,
        ti.column_count,
        COALESCE(fk.fk_count, 0) as foreign_key_count,
        COALESCE(idx.index_count, 0) as index_count
    FROM table_info ti
    LEFT JOIN table_stats ts
        ON ti.table_schema = ts.schemaname
        AND ti.table_name = ts.tablename
    LEFT JOIN pk_info pk
        ON ti.table_schema = pk.table_schema
        AND ti.table_name = pk.table_name
    LEFT JOIN fk_info fk
        ON ti.table_schema = fk.table_schema
        AND ti.table_name = fk.table_name
    LEFT JOIN index_info idx
        ON ti.table_schema = idx.schemaname
        AND ti.table_name = idx.tablename
    ORDER BY ti.table_name
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (schema_name, schema_name, schema_name, schema_name, schema_name))
        columns = [desc[0] for desc in cursor.description]
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
    return results

def scan_column_metadata(conn, schema_name: str = 'src_northwind') -> List[Dict]:
    """Scan column-level metadata including statistics."""
    query = """
    WITH column_info AS (
        SELECT
            c.table_schema,
            c.table_name,
            c.column_name,
            c.ordinal_position,
            c.data_type,
            c.character_maximum_length,
            c.numeric_precision,
            c.numeric_scale,
            c.is_nullable = 'YES' as is_nullable,
            c.column_default,
            col_description(pgc.oid, c.ordinal_position) as column_comment
        FROM information_schema.columns c
        JOIN pg_class pgc ON pgc.relname = c.table_name
        JOIN pg_namespace n ON n.oid = pgc.relnamespace AND n.nspname = c.table_schema
        WHERE c.table_schema = %s
    ),
    pk_columns AS (
        SELECT
            kcu.table_schema,
            kcu.table_name,
            kcu.column_name
        FROM information_schema.key_column_usage kcu
        JOIN information_schema.table_constraints tc
            ON kcu.constraint_name = tc.constraint_name
            AND kcu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
            AND kcu.table_schema = %s
    ),
    fk_columns AS (
        SELECT
            kcu.table_schema,
            kcu.table_name,
            kcu.column_name,
            ccu.table_schema as ref_schema,
            ccu.table_name as ref_table,
            ccu.column_name as ref_column
        FROM information_schema.key_column_usage kcu
        JOIN information_schema.constraint_column_usage ccu
            ON kcu.constraint_name = ccu.constraint_name
        JOIN information_schema.table_constraints tc
            ON kcu.constraint_name = tc.constraint_name
            AND kcu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
            AND kcu.table_schema = %s
    ),
    unique_columns AS (
        SELECT DISTINCT
            kcu.table_schema,
            kcu.table_name,
            kcu.column_name
        FROM information_schema.key_column_usage kcu
        JOIN information_schema.table_constraints tc
            ON kcu.constraint_name = tc.constraint_name
            AND kcu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'UNIQUE'
            AND kcu.table_schema = %s
    ),
    stats AS (
        -- Try to get stats, but make it optional
        SELECT
            schemaname::text as schemaname,
            tablename::text as tablename,
            attname::text as attname,
            n_distinct,
            null_frac,
            avg_width,
            correlation,
            COALESCE(most_common_vals::text, '') as most_common_vals,
            COALESCE(most_common_freqs::text, '') as most_common_freqs
        FROM pg_stats
        WHERE schemaname = %s
        UNION ALL
        -- Fallback if pg_stats doesn't exist or has no data
        SELECT
            %s::text as schemaname,
            ''::text as tablename,
            ''::text as attname,
            NULL::real as n_distinct,
            NULL::real as null_frac,
            NULL::integer as avg_width,
            NULL::real as correlation,
            ''::text as most_common_vals,
            ''::text as most_common_freqs
        WHERE NOT EXISTS (
            SELECT 1 FROM pg_stats WHERE schemaname = %s LIMIT 1
        )
    )
    SELECT
        ci.*,
        pk.column_name IS NOT NULL as is_primary_key,
        fk.column_name IS NOT NULL as is_foreign_key,
        uc.column_name IS NOT NULL as is_unique,
        fk.ref_schema as referenced_schema,
        fk.ref_table as referenced_table,
        fk.ref_column as referenced_column,
        s.n_distinct,
        s.null_frac as null_fraction,
        s.avg_width,
        s.correlation,
        s.most_common_vals,
        s.most_common_freqs
    FROM column_info ci
    LEFT JOIN pk_columns pk
        ON ci.table_schema = pk.table_schema
        AND ci.table_name = pk.table_name
        AND ci.column_name = pk.column_name
    LEFT JOIN fk_columns fk
        ON ci.table_schema = fk.table_schema
        AND ci.table_name = fk.table_name
        AND ci.column_name = fk.column_name
    LEFT JOIN unique_columns uc
        ON ci.table_schema = uc.table_schema
        AND ci.table_name = uc.table_name
        AND ci.column_name = uc.column_name
    LEFT JOIN stats s
        ON ci.table_schema = s.schemaname
        AND ci.table_name = s.tablename
        AND ci.column_name = s.attname
    ORDER BY ci.table_name, ci.ordinal_position
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (schema_name, schema_name, schema_name, schema_name, schema_name, schema_name, schema_name, schema_name))
        columns = [desc[0] for desc in cursor.description]
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
    return results

def scan_relationships(conn, schema_name: str = 'src_northwind') -> List[Dict]:
    """Scan foreign key relationships."""
    query = """
    SELECT
        tc.constraint_name,
        tc.table_schema as source_schema,
        tc.table_name as source_table,
        kcu.column_name as source_column,
        ccu.table_schema as target_schema,
        ccu.table_name as target_table,
        ccu.column_name as target_column,
        rc.delete_rule,
        rc.update_rule
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
        AND tc.table_schema = kcu.table_schema
    JOIN information_schema.constraint_column_usage ccu
        ON tc.constraint_name = ccu.constraint_name
    JOIN information_schema.referential_constraints rc
        ON tc.constraint_name = rc.constraint_name
    WHERE tc.constraint_type = 'FOREIGN KEY'
        AND tc.table_schema = %s
    ORDER BY tc.table_name, kcu.column_name
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (schema_name,))
        columns = [desc[0] for desc in cursor.description]
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
    return results

def generate_metadata_text(metadata_type: str, metadata: Dict) -> str:
    """Generate descriptive text for embedding from metadata."""
    if metadata_type == 'table':
        text_parts = [
            f"Table: {metadata['table_schema']}.{metadata['table_name']}"
        ]
        if metadata.get('table_comment'):
            text_parts.append(f"Description: {metadata['table_comment']}")
        text_parts.append(f"Type: {metadata['table_type']}")
        text_parts.append(f"Rows: {metadata['row_count']}")
        text_parts.append(f"Columns: {metadata['column_count']}")
        if metadata['foreign_key_count'] > 0:
            text_parts.append(f"Foreign Keys: {metadata['foreign_key_count']}")
        if metadata['has_primary_key']:
            text_parts.append("Has Primary Key")
        return " | ".join(text_parts)

    elif metadata_type == 'column':
        text_parts = [
            f"Column: {metadata['table_schema']}.{metadata['table_name']}.{metadata['column_name']}"
        ]
        text_parts.append(f"Type: {metadata['data_type']}")

        if metadata['is_primary_key']:
            text_parts.append("PRIMARY KEY")
        if metadata['is_foreign_key']:
            text_parts.append(f"FOREIGN KEY -> {metadata['referenced_table']}.{metadata['referenced_column']}")
        if metadata['is_unique']:
            text_parts.append("UNIQUE")
        if metadata['is_nullable']:
            text_parts.append("NULLABLE")

        if metadata.get('column_comment'):
            text_parts.append(f"Description: {metadata['column_comment']}")

        if metadata.get('n_distinct'):
            text_parts.append(f"Distinct Values: {metadata['n_distinct']}")
        if metadata.get('null_fraction') is not None:
            null_pct = metadata['null_fraction'] * 100
            text_parts.append(f"Null: {null_pct:.1f}%")

        return " | ".join(text_parts)

    elif metadata_type == 'relationship':
        return (f"Relationship: {metadata['source_table']}.{metadata['source_column']} "
                f"-> {metadata['target_table']}.{metadata['target_column']} "
                f"(Delete: {metadata['delete_rule']}, Update: {metadata['update_rule']})")

    return ""

def populate_catalog(conn):
    """Main function to populate the catalog schema with metadata."""
    print("=" * 60)
    print("Scanning Database Metadata")
    print("=" * 60)

    try:
        # Clear existing catalog data
        print("\nClearing existing catalog data...")
        with conn.cursor() as cursor:
            cursor.execute("TRUNCATE catalog.table_metadata CASCADE")
            cursor.execute("TRUNCATE catalog.column_metadata CASCADE")
            cursor.execute("TRUNCATE catalog.relationship_metadata CASCADE")
            conn.commit()

        # Scan and insert table metadata
        print("\nScanning table metadata...")
        tables = scan_table_metadata(conn)
        print(f"Found {len(tables)} tables")

        with conn.cursor() as cursor:
            for table in tables:
                metadata_text = generate_metadata_text('table', table)
                cursor.execute("""
                    INSERT INTO catalog.table_metadata (
                        schema_name, table_name, table_type, row_count,
                        table_size_bytes, description, table_comment,
                        has_primary_key, column_count, foreign_key_count,
                        index_count, metadata_text
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    table['table_schema'], table['table_name'], table['table_type'],
                    table['row_count'], table['table_size_bytes'], table.get('description'),
                    table.get('table_comment'), table['has_primary_key'],
                    table['column_count'], table['foreign_key_count'],
                    table['index_count'], metadata_text
                ))
            conn.commit()
            print(f"  ✓ Inserted {len(tables)} table records")

        # Scan and insert column metadata
        print("\nScanning column metadata...")
        columns = scan_column_metadata(conn)
        print(f"Found {len(columns)} columns")

        with conn.cursor() as cursor:
            for col in columns:
                metadata_text = generate_metadata_text('column', col)

                # Parse array strings for most_common_vals
                common_vals = None
                if col.get('most_common_vals'):
                    try:
                        # Remove curly braces and split
                        vals_str = col['most_common_vals'].strip('{}')
                        if vals_str:
                            common_vals = vals_str.split(',')[:5]  # Limit to 5 values
                    except:
                        pass

                cursor.execute("""
                    INSERT INTO catalog.column_metadata (
                        schema_name, table_name, column_name, ordinal_position,
                        data_type, character_maximum_length, numeric_precision,
                        numeric_scale, is_nullable, column_default, is_primary_key,
                        is_foreign_key, is_unique, referenced_schema, referenced_table,
                        referenced_column, column_comment, n_distinct, null_fraction,
                        avg_width, correlation, most_common_values, metadata_text
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                             %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    col['table_schema'], col['table_name'], col['column_name'],
                    col['ordinal_position'], col['data_type'],
                    col.get('character_maximum_length'), col.get('numeric_precision'),
                    col.get('numeric_scale'), col['is_nullable'], col.get('column_default'),
                    col['is_primary_key'], col['is_foreign_key'], col['is_unique'],
                    col.get('referenced_schema'), col.get('referenced_table'),
                    col.get('referenced_column'), col.get('column_comment'),
                    col.get('n_distinct'), col.get('null_fraction'),
                    col.get('avg_width'), col.get('correlation'),
                    common_vals, metadata_text
                ))
            conn.commit()
            print(f"  ✓ Inserted {len(columns)} column records")

        # Scan and insert relationship metadata
        print("\nScanning relationship metadata...")
        relationships = scan_relationships(conn)
        print(f"Found {len(relationships)} foreign key relationships")

        with conn.cursor() as cursor:
            for rel in relationships:
                metadata_text = generate_metadata_text('relationship', rel)

                # Determine relationship type
                rel_type = 'ONE_TO_MANY'  # Default assumption

                cursor.execute("""
                    INSERT INTO catalog.relationship_metadata (
                        constraint_name, source_schema, source_table, source_column,
                        target_schema, target_table, target_column, relationship_type,
                        delete_rule, update_rule, metadata_text
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    rel['constraint_name'], rel['source_schema'], rel['source_table'],
                    rel['source_column'], rel['target_schema'], rel['target_table'],
                    rel['target_column'], rel_type, rel['delete_rule'],
                    rel['update_rule'], metadata_text
                ))
            conn.commit()
            print(f"  ✓ Inserted {len(relationships)} relationship records")

        # Generate suggested KPIs
        print("\nGenerating suggested KPIs...")
        generate_suggested_kpis(conn)

        print("\n" + "=" * 60)
        print("✓ Metadata scan completed successfully!")
        print("=" * 60)

        # Show summary statistics
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT
                    (SELECT COUNT(*) FROM catalog.table_metadata) as tables,
                    (SELECT COUNT(*) FROM catalog.column_metadata) as columns,
                    (SELECT COUNT(*) FROM catalog.relationship_metadata) as relationships,
                    (SELECT COUNT(*) FROM catalog.suggested_kpis) as kpis
            """)
            stats = cursor.fetchone()
            print(f"\nCatalog Statistics:")
            print(f"  Tables: {stats[0]}")
            print(f"  Columns: {stats[1]}")
            print(f"  Relationships: {stats[2]}")
            print(f"  Suggested KPIs: {stats[3]}")

    except Exception as e:
        print(f"\n✗ Error during metadata scan: {e}")
        conn.rollback()
        raise

def generate_suggested_kpis(conn):
    """Generate suggested KPIs based on the scanned metadata."""
    kpis = [
        {
            'name': 'Total Sales Revenue',
            'description': 'Total revenue from all orders',
            'category': 'Sales',
            'measure': 'SUM(unit_price * quantity * (1 - discount))',
            'dimensions': ['order_date', 'product_id', 'customer_id'],
            'tables': ['orders', 'order_details'],
            'query_template': """
                SELECT SUM(unit_price * quantity * (1 - discount)) as total_revenue
                FROM order_details
            """
        },
        {
            'name': 'Product Sales Velocity',
            'description': 'Rate of product sales over time',
            'category': 'Sales',
            'measure': 'SUM(quantity) / DATEDIFF(days)',
            'dimensions': ['product_id', 'category_id', 'time_period'],
            'tables': ['orders', 'order_details', 'products'],
            'query_template': """
                SELECT product_id, SUM(quantity) / 30.0 as daily_velocity
                FROM order_details od
                JOIN orders o ON od.order_id = o.order_id
                WHERE o.order_date >= CURRENT_DATE - 30
                GROUP BY product_id
            """
        },
        {
            'name': 'Customer Lifetime Value',
            'description': 'Total revenue per customer over their lifetime',
            'category': 'Customer',
            'measure': 'SUM(order_value) per customer',
            'dimensions': ['customer_id', 'customer_segment', 'geography'],
            'tables': ['customers', 'orders', 'order_details'],
            'query_template': """
                SELECT customer_id,
                       SUM(unit_price * quantity * (1 - discount)) as lifetime_value
                FROM orders o
                JOIN order_details od ON o.order_id = od.order_id
                GROUP BY customer_id
            """
        },
        {
            'name': 'Inventory Turnover',
            'description': 'Rate at which inventory is sold and replaced',
            'category': 'Inventory',
            'measure': 'Cost of Goods Sold / Average Inventory',
            'dimensions': ['product_id', 'category_id', 'supplier_id'],
            'tables': ['products', 'order_details'],
            'query_template': """
                SELECT category_id,
                       SUM(quantity * unit_price) / AVG(units_in_stock * unit_price) as turnover
                FROM products p
                JOIN order_details od ON p.product_id = od.product_id
                GROUP BY category_id
            """
        }
    ]

    with conn.cursor() as cursor:
        for kpi in kpis:
            metadata_text = (f"KPI: {kpi['name']} | Category: {kpi['category']} | "
                           f"Description: {kpi['description']} | "
                           f"Tables: {', '.join(kpi['tables'])}")

            cursor.execute("""
                INSERT INTO catalog.suggested_kpis (
                    kpi_name, kpi_description, kpi_category,
                    measure_expression, dimension_columns,
                    required_tables, query_template, metadata_text
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                kpi['name'], kpi['description'], kpi['category'],
                kpi['measure'], kpi['dimensions'], kpi['tables'],
                kpi['query_template'], metadata_text
            ))
        conn.commit()
        print(f"  ✓ Generated {len(kpis)} suggested KPIs")

def main():
    """Main execution function."""
    try:
        conn = get_db_connection()
        print("✓ Connected to database")

        populate_catalog(conn)

        print("\nNext steps:")
        print("1. Run 30_embed_metadata.py to generate embeddings")
        print("2. Run 40_metadata_rag_search.py to test RAG search")
        print("3. Run 50_mart_planning_agent.py to generate mart plans")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()