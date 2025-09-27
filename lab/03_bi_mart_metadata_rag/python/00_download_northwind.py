#!/usr/bin/env python3
"""
Download Northwind CSV data files for the BI Mart Metadata RAG lab.
Uses the Northwind database in CSV format for demonstration purposes.
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

def download_northwind_data():
    """Download and extract Northwind CSV files."""

    # Get the data directory path
    script_dir = Path(__file__).parent
    lab_dir = script_dir.parent
    data_dir = lab_dir / 'data'

    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)

    print("Downloading Northwind CSV data...")

    # Download URL for Northwind CSV data
    # Using a reliable source with CSV format
    csv_urls = {
        'categories.csv': 'https://raw.githubusercontent.com/graphql-compose/graphql-compose-examples/master/examples/northwind/data/csv/categories.csv',
        'customers.csv': 'https://raw.githubusercontent.com/graphql-compose/graphql-compose-examples/master/examples/northwind/data/csv/customers.csv',
        'employees.csv': 'https://raw.githubusercontent.com/graphql-compose/graphql-compose-examples/master/examples/northwind/data/csv/employees.csv',
        'orders.csv': 'https://raw.githubusercontent.com/graphql-compose/graphql-compose-examples/master/examples/northwind/data/csv/orders.csv',
        'order_details.csv': 'https://raw.githubusercontent.com/graphql-compose/graphql-compose-examples/master/examples/northwind/data/csv/order_details.csv',
        'products.csv': 'https://raw.githubusercontent.com/graphql-compose/graphql-compose-examples/master/examples/northwind/data/csv/products.csv',
        'regions.csv': 'https://raw.githubusercontent.com/graphql-compose/graphql-compose-examples/master/examples/northwind/data/csv/regions.csv',
        'shippers.csv': 'https://raw.githubusercontent.com/graphql-compose/graphql-compose-examples/master/examples/northwind/data/csv/shippers.csv',
        'suppliers.csv': 'https://raw.githubusercontent.com/graphql-compose/graphql-compose-examples/master/examples/northwind/data/csv/suppliers.csv',
        'territories.csv': 'https://raw.githubusercontent.com/graphql-compose/graphql-compose-examples/master/examples/northwind/data/csv/territories.csv',
        'employee_territories.csv': 'https://raw.githubusercontent.com/graphql-compose/graphql-compose-examples/master/examples/northwind/data/csv/employee_territories.csv'
    }

    # Download each CSV file
    for filename, url in csv_urls.items():
        filepath = data_dir / filename
        try:
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"  ✓ Downloaded {filename}")
        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {e}")
            # Try alternative approach or skip
            continue

    print("\nNorthwind CSV data downloaded successfully!")
    print(f"Files saved in: {data_dir}")

    # List downloaded files
    csv_files = list(data_dir.glob('*.csv'))
    if csv_files:
        print(f"\nDownloaded {len(csv_files)} CSV files:")
        for csv_file in sorted(csv_files):
            size_kb = csv_file.stat().st_size / 1024
            print(f"  - {csv_file.name}: {size_kb:.1f} KB")

    return data_dir

def verify_data_integrity(data_dir):
    """Verify that all required tables are present."""
    required_tables = [
        'categories', 'customers', 'employees', 'orders',
        'order_details', 'products', 'suppliers'
    ]

    missing_tables = []
    for table in required_tables:
        csv_file = data_dir / f"{table}.csv"
        if not csv_file.exists():
            missing_tables.append(table)

    if missing_tables:
        print(f"\nWarning: Missing required tables: {', '.join(missing_tables)}")
        return False

    print("\n✓ All required tables are present")
    return True

def main():
    """Main execution function."""
    try:
        data_dir = download_northwind_data()
        verify_data_integrity(data_dir)

        print("\nNext steps:")
        print("1. Run 10_load_csv_to_pg.py to load data into PostgreSQL")
        print("2. Run 20_scan_metadata.py to extract schema metadata")
        print("3. Run 30_embed_metadata.py to generate embeddings")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()