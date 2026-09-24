"""Generate configurable Athena examples without creating AWS resources."""

from __future__ import annotations

import argparse
import os
import re

from src.data.schema import COLUMN_NAMES, NUMERIC_COLUMNS, TARGET

DEFAULT_DATABASE = os.getenv("FINSIGHT_ATHENA_DATABASE", "finsight")
DEFAULT_TABLE = os.getenv("FINSIGHT_ATHENA_TABLE", "german_credit_analytics")


def identifier(value: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError("Use lowercase SQL identifiers: letters, digits, underscores; start with a letter")
    return value


def example_queries(database: str = DEFAULT_DATABASE, table: str = DEFAULT_TABLE) -> dict[str, str]:
    relation = f'"{identifier(database)}"."{identifier(table)}"'
    return {
        "applicant_count": f"SELECT COUNT(*) AS applicant_count FROM {relation};",
        "risk_counts": f"SELECT credit_risk, credit_risk_label, COUNT(*) AS applicants FROM {relation} GROUP BY credit_risk, credit_risk_label ORDER BY credit_risk;",
        "average_credit_amount": f"SELECT AVG(CAST(credit_amount AS DOUBLE)) AS average_credit_amount_dm FROM {relation};",
        "average_loan_duration": f"SELECT AVG(CAST(duration_months AS DOUBLE)) AS average_duration_months FROM {relation};",
        "grouped_credit_risk": f"SELECT savings_status, employment_duration, COUNT(*) AS applicants, SUM(is_bad_risk) AS bad_risk_count, ROUND(100.0 * AVG(CAST(is_bad_risk AS DOUBLE)), 2) AS bad_risk_percent FROM {relation} GROUP BY savings_status, employment_duration ORDER BY applicants DESC;",
    }


def create_table_sql(s3_location: str, database: str = DEFAULT_DATABASE,
                     table: str = DEFAULT_TABLE) -> str:
    """Explicit schema alternative/correction to crawler inference; CSV order matters."""
    database, table = identifier(database), identifier(table)
    if not re.fullmatch(r"s3://[a-z0-9][a-z0-9.-]+/[A-Za-z0-9_./-]+/", s3_location):
        raise ValueError("Use an S3 prefix ending in /, with no quotes or whitespace")
    columns = [("applicant_id", "bigint")]
    columns += [(name, "bigint" if name in NUMERIC_COLUMNS + [TARGET] else "string") for name in COLUMN_NAMES]
    columns += [("credit_risk_label", "string"), ("is_bad_risk", "bigint")]
    definitions = ",\n  ".join(f"`{name}` {kind}" for name, kind in columns)
    return (
        f"CREATE EXTERNAL TABLE IF NOT EXISTS `{database}`.`{table}` (\n  {definitions}\n)\n"
        "ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'\n"
        "WITH SERDEPROPERTIES ('separatorChar'=',', 'quoteChar'='\"')\n"
        "STORED AS TEXTFILE\n"
        f"LOCATION '{s3_location}'\n"
        "TBLPROPERTIES ('skip.header.line.count'='1');"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--s3-location", help="Optionally print a CREATE TABLE statement too")
    args = parser.parse_args()
    if args.s3_location:
        print(create_table_sql(args.s3_location, args.database, args.table))
    for name, query in example_queries(args.database, args.table).items():
        print(f"\n-- {name}\n{query}")


if __name__ == "__main__":
    main()
