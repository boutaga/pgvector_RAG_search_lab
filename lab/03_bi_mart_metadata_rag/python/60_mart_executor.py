#!/usr/bin/env python3
"""
Mart Executor - Creates and populates data mart based on generated plans
Handles DDL/DML generation and execution with proper error handling
"""

import os
import sys
import json
import psycopg2
from psycopg2 import sql
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

# Add models directory to path
script_dir = Path(__file__).parent
lab_dir = script_dir.parent
models_dir = lab_dir / 'models'
sys.path.insert(0, str(models_dir))

from mart_plan import MartPlan, ExecutionStep, ExecutionPlan, ExecutionResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MartExecutor:
    """Executes mart plans by creating database schemas and populating tables"""

    def __init__(self, conn=None):
        """Initialize the mart executor"""
        if conn:
            self.conn = conn
        else:
            self.conn = self.get_db_connection()

    def get_db_connection(self):
        """Create a database connection"""
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

    def generate_execution_plan(self, mart_plan: MartPlan) -> ExecutionPlan:
        """Generate a detailed execution plan from a mart plan"""
        steps = []
        step_number = 0

        # Step 1: Create schema
        steps.append(ExecutionStep(
            step_number=step_number,
            step_type="CREATE_SCHEMA",
            description=f"Create schema {mart_plan.target_schema}",
            sql_statement=f"CREATE SCHEMA IF NOT EXISTS {mart_plan.target_schema};",
            depends_on=None,
            estimated_duration_ms=100
        ))
        step_number += 1

        # Step 2: Create dimension tables
        dimension_step_numbers = {}
        for dim in mart_plan.dimensions:
            sql = self.generate_dimension_ddl(mart_plan.target_schema, dim)
            steps.append(ExecutionStep(
                step_number=step_number,
                step_type="CREATE_DIMENSION",
                description=f"Create dimension table {dim.name}",
                sql_statement=sql,
                depends_on=[0],
                estimated_duration_ms=200
            ))
            dimension_step_numbers[dim.name] = step_number
            step_number += 1

        # Step 3: Populate dimension tables
        for dim in mart_plan.dimensions:
            sql = self.generate_dimension_populate_sql(
                mart_plan.source_schema,
                mart_plan.target_schema,
                dim
            )
            steps.append(ExecutionStep(
                step_number=step_number,
                step_type="POPULATE_DIMENSION",
                description=f"Populate dimension {dim.name}",
                sql_statement=sql,
                depends_on=[dimension_step_numbers[dim.name]],
                estimated_duration_ms=5000
            ))
            step_number += 1

        # Step 4: Create fact tables
        fact_step_numbers = {}
        for fact in mart_plan.facts:
            sql = self.generate_fact_ddl(mart_plan.target_schema, fact)
            steps.append(ExecutionStep(
                step_number=step_number,
                step_type="CREATE_FACT",
                description=f"Create fact table {fact.name}",
                sql_statement=sql,
                depends_on=[0],
                estimated_duration_ms=200
            ))
            fact_step_numbers[fact.name] = step_number
            step_number += 1

        # Step 5: Populate fact tables
        for fact in mart_plan.facts:
            sql = self.generate_fact_populate_sql(
                mart_plan.source_schema,
                mart_plan.target_schema,
                fact
            )
            # Depends on all dimensions being populated
            dependencies = [fact_step_numbers[fact.name]]
            dependencies.extend([s.step_number for s in steps if s.step_type == "POPULATE_DIMENSION"])

            steps.append(ExecutionStep(
                step_number=step_number,
                step_type="POPULATE_FACT",
                description=f"Populate fact table {fact.name}",
                sql_statement=sql,
                depends_on=dependencies,
                estimated_duration_ms=30000
            ))
            step_number += 1

        # Step 6: Create indexes
        if mart_plan.indexes:
            for index in mart_plan.indexes:
                sql = self.generate_index_ddl(mart_plan.target_schema, index)
                steps.append(ExecutionStep(
                    step_number=step_number,
                    step_type="CREATE_INDEX",
                    description=f"Create index {index.name}",
                    sql_statement=sql,
                    depends_on=[s.step_number for s in steps if "POPULATE" in s.step_type],
                    estimated_duration_ms=1000
                ))
                step_number += 1

        # Step 7: Analyze tables
        all_tables = [f.name for f in mart_plan.facts] + [d.name for d in mart_plan.dimensions]
        for table_name in all_tables:
            steps.append(ExecutionStep(
                step_number=step_number,
                step_type="ANALYZE_TABLE",
                description=f"Analyze table {table_name}",
                sql_statement=f"ANALYZE {mart_plan.target_schema}.{table_name};",
                depends_on=[s.step_number for s in steps if "POPULATE" in s.step_type],
                estimated_duration_ms=500
            ))
            step_number += 1

        total_duration = sum(s.estimated_duration_ms or 0 for s in steps)

        return ExecutionPlan(
            mart_plan=mart_plan,
            steps=steps,
            total_estimated_duration_ms=total_duration
        )

    def generate_dimension_ddl(self, schema: str, dim) -> str:
        """Generate DDL for a dimension table"""
        columns = [f"{dim.key_column} SERIAL PRIMARY KEY"]

        # Add attribute columns
        for attr in dim.attributes:
            # Infer data type from attribute name
            if 'date' in attr.lower():
                col_type = 'DATE'
            elif 'id' in attr.lower() or 'count' in attr.lower():
                col_type = 'INTEGER'
            elif 'price' in attr.lower() or 'amount' in attr.lower():
                col_type = 'NUMERIC(10,2)'
            else:
                col_type = 'VARCHAR(255)'

            columns.append(f"{attr} {col_type}")

        ddl = f"""
CREATE TABLE IF NOT EXISTS {schema}.{dim.name} (
    {',\n    '.join(columns)}
);

COMMENT ON TABLE {schema}.{dim.name} IS '{dim.description or "Dimension table"}';
"""
        return ddl

    def generate_fact_ddl(self, schema: str, fact) -> str:
        """Generate DDL for a fact table"""
        columns = []

        # Add grain columns (composite key)
        for grain_col in fact.grain:
            if 'date' in grain_col.lower():
                columns.append(f"{grain_col} DATE NOT NULL")
            else:
                columns.append(f"{grain_col} INTEGER NOT NULL")

        # Add dimension keys
        for dim_key in fact.dimension_keys:
            if dim_key not in fact.grain:
                if 'date' in dim_key.lower():
                    columns.append(f"{dim_key} DATE")
                else:
                    columns.append(f"{dim_key} INTEGER")

        # Add measure columns
        for measure in fact.measures:
            if measure.data_type:
                columns.append(f"{measure.name} {measure.data_type}")
            else:
                columns.append(f"{measure.name} NUMERIC")

        # Add composite primary key
        pk_columns = ', '.join(fact.grain)

        ddl = f"""
CREATE TABLE IF NOT EXISTS {schema}.{fact.name} (
    {',\n    '.join(columns)},
    PRIMARY KEY ({pk_columns})
);

COMMENT ON TABLE {schema}.{fact.name} IS '{fact.description or "Fact table"}';
"""

        # Add column comments for measures
        for measure in fact.measures:
            if measure.description:
                ddl += f"\nCOMMENT ON COLUMN {schema}.{fact.name}.{measure.name} IS '{measure.description}';"

        return ddl

    def generate_dimension_populate_sql(self, source_schema: str, target_schema: str, dim) -> str:
        """Generate SQL to populate a dimension table"""
        # Build column list
        columns = [dim.key_column] + dim.attributes

        # Handle special cases for date dimensions
        if dim.name == 'dim_date':
            return f"""
INSERT INTO {target_schema}.{dim.name} ({', '.join(columns)})
SELECT DISTINCT
    {', '.join(columns)}
FROM (
    SELECT DISTINCT
        order_date,
        EXTRACT(year FROM order_date) as year,
        EXTRACT(quarter FROM order_date) as quarter,
        EXTRACT(month FROM order_date) as month,
        EXTRACT(dow FROM order_date) as day_of_week,
        TO_CHAR(order_date, 'Month') as month_name,
        TO_CHAR(order_date, 'Day') as day_name
    FROM {source_schema}.orders
    WHERE order_date IS NOT NULL
) AS date_data
ON CONFLICT DO NOTHING;
"""

        # Standard dimension population
        # Map attributes to source columns
        select_list = []
        for col in columns:
            # Try to find matching column in source table
            if col == dim.key_column or col in dim.attributes:
                select_list.append(col)

        sql = f"""
INSERT INTO {target_schema}.{dim.name} ({', '.join(columns)})
SELECT DISTINCT
    {', '.join(select_list)}
FROM {source_schema}.{dim.source_table}
ON CONFLICT DO NOTHING;
"""
        return sql

    def generate_fact_populate_sql(self, source_schema: str, target_schema: str, fact) -> str:
        """Generate SQL to populate a fact table"""
        # Build select list with measures
        select_list = []

        # Add grain columns
        select_list.extend(fact.grain)

        # Add dimension keys
        for dim_key in fact.dimension_keys:
            if dim_key not in fact.grain:
                select_list.append(dim_key)

        # Add measure expressions
        for measure in fact.measures:
            if measure.aggregation == 'sum':
                select_list.append(f"SUM({measure.expression}) AS {measure.name}")
            elif measure.aggregation == 'count':
                select_list.append(f"COUNT({measure.expression}) AS {measure.name}")
            elif measure.aggregation == 'avg':
                select_list.append(f"AVG({measure.expression}) AS {measure.name}")
            elif measure.aggregation == 'max':
                select_list.append(f"MAX({measure.expression}) AS {measure.name}")
            elif measure.aggregation == 'min':
                select_list.append(f"MIN({measure.expression}) AS {measure.name}")
            else:
                select_list.append(f"{measure.expression} AS {measure.name}")

        # Build FROM clause with joins
        from_clause = f"{source_schema}.{fact.source_tables[0]}"
        for i, join_condition in enumerate(fact.join_conditions):
            if i < len(fact.source_tables) - 1:
                from_clause += f"\nJOIN {source_schema}.{fact.source_tables[i+1]} ON {join_condition}"

        # Build WHERE clause
        where_clause = ""
        if fact.where_conditions:
            where_clause = "\nWHERE " + " AND ".join(fact.where_conditions)

        # Build GROUP BY clause (for grain)
        group_by = ', '.join([col for col in fact.grain + fact.dimension_keys if col not in [m.name for m in fact.measures]])

        sql = f"""
INSERT INTO {target_schema}.{fact.name}
SELECT
    {',\n    '.join(select_list)}
FROM {from_clause}
{where_clause}
GROUP BY {group_by};
"""
        return sql

    def generate_index_ddl(self, schema: str, index) -> str:
        """Generate DDL for an index"""
        columns = ', '.join(index.columns)
        unique = "UNIQUE " if index.is_unique else ""

        sql = f"""
CREATE {unique}INDEX IF NOT EXISTS {index.name}
ON {schema}.{index.table_name} USING {index.index_type} ({columns});
"""
        return sql

    def execute_plan(
        self,
        execution_plan: ExecutionPlan,
        dry_run: bool = False,
        progress_callback=None
    ) -> ExecutionResult:
        """Execute a mart execution plan"""
        import uuid
        execution_id = str(uuid.uuid4())

        result = ExecutionResult(
            execution_id=execution_id,
            mart_plan=execution_plan.mart_plan,
            status="EXECUTING",
            steps_completed=0,
            total_steps=len(execution_plan.steps),
            tables_created=0,
            rows_processed=0
        )

        start_time = datetime.now()

        try:
            with self.conn.cursor() as cursor:
                for step in execution_plan.steps:
                    logger.info(f"Executing step {step.step_number}: {step.description}")

                    if progress_callback:
                        progress_callback(step.step_number, len(execution_plan.steps), step.description)

                    if not dry_run:
                        try:
                            cursor.execute(step.sql_statement)

                            # Track rows affected for DML statements
                            if "INSERT" in step.sql_statement or "UPDATE" in step.sql_statement:
                                result.rows_processed += cursor.rowcount

                            # Track table creation
                            if step.step_type in ["CREATE_FACT", "CREATE_DIMENSION"]:
                                result.tables_created += 1

                            self.conn.commit()
                            result.steps_completed += 1

                        except Exception as e:
                            logger.error(f"Step {step.step_number} failed: {e}")
                            self.conn.rollback()
                            result.status = "FAILED"
                            result.error_message = str(e)
                            break
                    else:
                        # Dry run - just log the SQL
                        logger.info(f"[DRY RUN] Would execute: {step.sql_statement[:100]}...")
                        result.steps_completed += 1

                if result.status != "FAILED":
                    result.status = "COMPLETED"

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            result.status = "FAILED"
            result.error_message = str(e)
            self.conn.rollback()

        # Calculate execution time
        end_time = datetime.now()
        result.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return result

    def rollback_mart(self, schema_name: str):
        """Rollback a mart by dropping the schema"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")
                self.conn.commit()
                logger.info(f"Successfully rolled back schema {schema_name}")
        except Exception as e:
            logger.error(f"Failed to rollback schema {schema_name}: {e}")
            raise

    def validate_source_tables(self, mart_plan: MartPlan) -> Tuple[bool, List[str]]:
        """Validate that all source tables exist"""
        errors = []

        with self.conn.cursor() as cursor:
            # Get all source tables
            all_source_tables = set()
            for fact in mart_plan.facts:
                all_source_tables.update(fact.source_tables)
            for dim in mart_plan.dimensions:
                all_source_tables.add(dim.source_table)

            # Check each table exists
            for table in all_source_tables:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = %s
                        AND table_name = %s
                    )
                """, (mart_plan.source_schema, table))

                exists = cursor.fetchone()[0]
                if not exists:
                    errors.append(f"Source table {mart_plan.source_schema}.{table} does not exist")

        return len(errors) == 0, errors

    def estimate_mart_size(self, mart_plan: MartPlan) -> Dict[str, int]:
        """Estimate the size of the mart tables"""
        estimates = {}

        with self.conn.cursor() as cursor:
            # Estimate fact table sizes
            for fact in mart_plan.facts:
                # Get row count from source tables
                if fact.source_tables:
                    cursor.execute(f"""
                        SELECT COUNT(*)
                        FROM {mart_plan.source_schema}.{fact.source_tables[0]}
                    """)
                    base_count = cursor.fetchone()[0]

                    # Adjust based on grain
                    if len(fact.grain) > 1:
                        # Multiple grain columns likely reduce row count
                        estimates[fact.name] = int(base_count * 0.7)
                    else:
                        estimates[fact.name] = base_count

            # Estimate dimension table sizes
            for dim in mart_plan.dimensions:
                cursor.execute(f"""
                    SELECT COUNT(DISTINCT {dim.key_column})
                    FROM {mart_plan.source_schema}.{dim.source_table}
                """)
                estimates[dim.name] = cursor.fetchone()[0]

        return estimates

def main():
    """Main execution for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Execute a mart plan')
    parser.add_argument('plan_file', help='Path to mart plan JSON file')
    parser.add_argument('--dry-run', action='store_true', help='Show SQL without executing')
    parser.add_argument('--rollback', action='store_true', help='Rollback the mart schema')
    args = parser.parse_args()

    # Load mart plan
    with open(args.plan_file, 'r') as f:
        plan_data = json.load(f)
        mart_plan = MartPlan(**plan_data)

    # Create executor
    executor = MartExecutor()

    if args.rollback:
        # Rollback mode
        executor.rollback_mart(mart_plan.target_schema)
    else:
        # Validate source tables
        valid, errors = executor.validate_source_tables(mart_plan)
        if not valid:
            print("❌ Validation errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)

        # Generate execution plan
        execution_plan = executor.generate_execution_plan(mart_plan)

        print(f"📋 Execution plan has {len(execution_plan.steps)} steps")
        print(f"⏱️  Estimated time: {execution_plan.total_estimated_duration_ms / 1000:.1f} seconds")

        if not args.dry_run:
            response = input("\nProceed with execution? (y/n): ")
            if response.lower() != 'y':
                print("Execution cancelled")
                sys.exit(0)

        # Execute plan
        def progress_callback(current, total, description):
            print(f"[{current}/{total}] {description}")

        result = executor.execute_plan(execution_plan, dry_run=args.dry_run, progress_callback=progress_callback)

        # Print results
        print("\n" + "=" * 60)
        if result.status == "COMPLETED":
            print(f"✅ Mart execution completed successfully!")
            print(f"  - Tables created: {result.tables_created}")
            print(f"  - Rows processed: {result.rows_processed:,}")
            print(f"  - Execution time: {result.execution_time_ms / 1000:.1f} seconds")
        else:
            print(f"❌ Mart execution failed!")
            print(f"  - Error: {result.error_message}")
            print(f"  - Steps completed: {result.steps_completed}/{result.total_steps}")

if __name__ == "__main__":
    main()