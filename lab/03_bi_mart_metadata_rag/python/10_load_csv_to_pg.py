#!/usr/bin/env python3
"""
Load Northwind CSV data into PostgreSQL source schema.
Creates tables and loads data using COPY command for efficiency.
"""

import os
import sys
import psycopg2
from psycopg2 import sql
from pathlib import Path
import csv
from datetime import datetime

def get_db_connection():
    """Create a database connection using environment variables."""
    db_url = os.environ.get('DATABASE_URL')
    if db_url:
        return psycopg2.connect(db_url)
    else:
        # Default connection parameters
        return psycopg2.connect(
            host=os.environ.get('DB_HOST', 'localhost'),
            port=os.environ.get('DB_PORT', '5432'),
            database=os.environ.get('DB_NAME', 'postgres'),
            user=os.environ.get('DB_USER', 'postgres'),
            password=os.environ.get('DB_PASSWORD', '')
        )

def execute_sql_file(conn, sql_file_path):
    """Execute SQL commands from a file."""
    with open(sql_file_path, 'r') as f:
        sql_content = f.read()

    with conn.cursor() as cursor:
        try:
            cursor.execute(sql_content)
            conn.commit()
            print(f"✓ Executed {sql_file_path.name}")
        except Exception as e:
            conn.rollback()
            print(f"✗ Error executing {sql_file_path.name}: {e}")
            raise

def clean_csv_value(value, column_name=None):
    """Clean CSV values for PostgreSQL."""
    if value == '' or value == 'NULL':
        return None

    # Skip picture/photo columns (they contain hex data)
    if column_name and ('picture' in column_name.lower() or 'photo' in column_name.lower()):
        return None  # Skip binary data for now

    # Define numeric columns that should not be converted to boolean
    numeric_columns = {'categoryID', 'supplierID', 'productID', 'orderID', 'customerID',
                      'employeeID', 'regionID', 'territoryID', 'shipperID',
                      'unitPrice', 'quantity', 'discount', 'unitsInStock', 'unitsOnOrder',
                      'reorderLevel', 'freight', 'shipVia', 'reportsTo'}

    # Don't convert numeric values to boolean
    if column_name and column_name in numeric_columns:
        return value

    # Handle boolean values only for non-numeric fields
    if value.lower() in ('true', 'yes'):
        return True
    elif value.lower() in ('false', 'no'):
        return False
    return value

def load_csv_to_table(conn, schema_name, table_name, csv_file_path, column_mappings=None, skip_column=None):
    """Load CSV data into a PostgreSQL table.

    Args:
        skip_column: Column name to skip during initial load (for self-referential FKs)
    """
    with conn.cursor() as cursor:
        # Read CSV to determine columns
        with open(csv_file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            # Map CSV columns to table columns if mapping provided
            if column_mappings:
                table_columns = [column_mappings.get(h, h) for h in headers if not (skip_column and column_mappings.get(h, h) == skip_column)]
                headers_to_use = [h for h in headers if not (skip_column and column_mappings.get(h, h) == skip_column)]
            else:
                table_columns = [h for h in headers if h != skip_column]
                headers_to_use = [h for h in headers if h != skip_column]

            # Prepare data for insertion
            rows = []
            for row in reader:
                cleaned_row = [clean_csv_value(row[h], h) for h in headers_to_use]
                rows.append(cleaned_row)

        if not rows:
            print(f"  ⚠ No data found in {csv_file_path.name}")
            return

        # Create INSERT query with ON CONFLICT DO NOTHING for safety
        insert_query = sql.SQL("""
            INSERT INTO {}.{} ({})
            VALUES ({})
            ON CONFLICT DO NOTHING
        """).format(
            sql.Identifier(schema_name),
            sql.Identifier(table_name),
            sql.SQL(', ').join([sql.Identifier(col) for col in table_columns]),
            sql.SQL(', ').join([sql.Placeholder() for _ in table_columns])
        )

        # Execute batch insert
        try:
            cursor.executemany(insert_query, rows)
            rows_inserted = cursor.rowcount
            conn.commit()
            if skip_column:
                print(f"  ✓ Loaded {rows_inserted} rows into {schema_name}.{table_name} (without {skip_column})")
            else:
                print(f"  ✓ Loaded {rows_inserted} rows into {schema_name}.{table_name}")
        except Exception as e:
            conn.rollback()
            print(f"  ✗ Error loading {table_name}: {e}")
            raise

def update_employee_reports_to(conn, csv_file_path, column_mappings):
    """Update the reports_to column in employees table after initial load."""
    with conn.cursor() as cursor:
        # Read CSV to get reports_to data
        with open(csv_file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)

            for row in reader:
                employee_id = clean_csv_value(row['employeeID'], 'employeeID')
                reports_to = clean_csv_value(row.get('reportsTo'), 'reportsTo')

                if reports_to:
                    update_query = sql.SQL("""
                        UPDATE src_northwind.employees
                        SET reports_to = %s
                        WHERE employee_id = %s
                    """)
                    try:
                        cursor.execute(update_query, (reports_to, employee_id))
                    except Exception as e:
                        print(f"  ⚠ Warning: Could not update reports_to for employee {employee_id}: {e}")

        conn.commit()
        print(f"  ✓ Updated reports_to relationships for employees")

def load_northwind_data():
    """Main function to load all Northwind data."""
    script_dir = Path(__file__).parent
    lab_dir = script_dir.parent
    data_dir = lab_dir / 'data'
    sql_dir = lab_dir / 'sql'

    print("=" * 60)
    print("Loading Northwind Data to PostgreSQL")
    print("=" * 60)

    # Connect to database
    try:
        conn = get_db_connection()
        print("✓ Connected to database")
    except Exception as e:
        print(f"✗ Failed to connect to database: {e}")
        print("\nMake sure PostgreSQL is running and DATABASE_URL is set:")
        print('  export DATABASE_URL="postgresql://user:password@localhost/dbname"')
        sys.exit(1)

    try:
        # Execute setup SQL files
        print("\nSetting up database schema...")
        sql_files = [
            '00_setup_extensions.sql',
            '01_create_source_schema.sql',
            '02_create_metadata_schema.sql'
        ]

        for sql_file in sql_files:
            sql_path = sql_dir / sql_file
            if sql_path.exists():
                execute_sql_file(conn, sql_path)

        # Define table loading order (respecting foreign key constraints)
        table_load_order = [
            ('categories', 'categories.csv', {
                'categoryID': 'category_id',
                'categoryName': 'category_name'
            }),
            ('regions', 'regions.csv', {
                'regionID': 'region_id',
                'regionDescription': 'region_description'
            }),
            ('suppliers', 'suppliers.csv', {
                'supplierID': 'supplier_id',
                'companyName': 'company_name',
                'contactName': 'contact_name',
                'contactTitle': 'contact_title',
                'postalCode': 'postal_code',
                'homePage': 'homepage'  # Map camelCase to lowercase
            }),
            ('shippers', 'shippers.csv', {
                'shipperID': 'shipper_id',
                'companyName': 'company_name'
            }),
            ('customers', 'customers.csv', {
                'customerID': 'customer_id',
                'companyName': 'company_name',
                'contactName': 'contact_name',
                'contactTitle': 'contact_title',
                'postalCode': 'postal_code'
            }),
            ('employees', 'employees.csv', {
                'employeeID': 'employee_id',
                'lastName': 'last_name',
                'firstName': 'first_name',
                'titleOfCourtesy': 'title_of_courtesy',
                'birthDate': 'birth_date',
                'hireDate': 'hire_date',
                'postalCode': 'postal_code',
                'homePhone': 'home_phone',
                'reportsTo': 'reports_to',
                'photoPath': 'photo_path'
            }),
            ('products', 'products.csv', {
                'productID': 'product_id',
                'productName': 'product_name',
                'supplierID': 'supplier_id',
                'categoryID': 'category_id',
                'quantityPerUnit': 'quantity_per_unit',
                'unitPrice': 'unit_price',
                'unitsInStock': 'units_in_stock',
                'unitsOnOrder': 'units_on_order',
                'reorderLevel': 'reorder_level'
            }),
            ('territories', 'territories.csv', {
                'territoryID': 'territory_id',
                'territoryDescription': 'territory_description',
                'regionID': 'region_id'
            }),
            ('orders', 'orders.csv', {
                'orderID': 'order_id',
                'customerID': 'customer_id',
                'employeeID': 'employee_id',
                'orderDate': 'order_date',
                'requiredDate': 'required_date',
                'shippedDate': 'shipped_date',
                'shipVia': 'ship_via',
                'shipName': 'ship_name',
                'shipAddress': 'ship_address',
                'shipCity': 'ship_city',
                'shipRegion': 'ship_region',
                'shipPostalCode': 'ship_postal_code',
                'shipCountry': 'ship_country'
            }),
            ('order_details', 'order_details.csv', {
                'orderID': 'order_id',
                'productID': 'product_id',
                'unitPrice': 'unit_price'
            }),
            ('employee_territories', 'employee_territories.csv', {
                'employeeID': 'employee_id',
                'territoryID': 'territory_id'
            })
        ]

        print("\nLoading CSV data...")
        employees_data = None  # Store employees data for second pass
        for table_name, csv_filename, column_mapping in table_load_order:
            csv_path = data_dir / csv_filename
            if csv_path.exists():
                # Special handling for employees table - load in two passes
                if table_name == 'employees':
                    # First pass: load without reports_to column
                    load_csv_to_table(conn, 'src_northwind', table_name, csv_path, column_mapping, skip_column='reports_to')
                    employees_data = (csv_path, column_mapping)  # Save for second pass
                else:
                    load_csv_to_table(conn, 'src_northwind', table_name, csv_path, column_mapping)
            else:
                print(f"  ⚠ Skipping {table_name}: {csv_filename} not found")

        # Second pass for employees: update reports_to relationships
        if employees_data:
            csv_path, column_mapping = employees_data
            update_employee_reports_to(conn, csv_path, column_mapping)

        # Verify data was loaded
        print("\nVerifying data load...")
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT
                    table_name,
                    COUNT(*) OVER() as table_count
                FROM information_schema.tables
                WHERE table_schema = 'src_northwind'
                    AND table_type = 'BASE TABLE'
                LIMIT 1
            """)
            result = cursor.fetchone()
            if result:
                table_count = result[1]
                print(f"✓ Created {table_count} tables in src_northwind schema")

            # Get row counts for key tables
            key_tables = ['orders', 'order_details', 'products', 'customers']
            for table in key_tables:
                cursor.execute(sql.SQL("""
                    SELECT COUNT(*) FROM src_northwind.{}
                """).format(sql.Identifier(table)))
                count = cursor.fetchone()[0]
                print(f"  - {table}: {count} rows")

        print("\n" + "=" * 60)
        print("✓ Northwind data loaded successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run 20_scan_metadata.py to extract schema metadata")
        print("2. Run 30_embed_metadata.py to generate embeddings")
        print("3. Run 40_metadata_rag_search.py to test RAG search")

    except Exception as e:
        print(f"\n✗ Error during data load: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    load_northwind_data()