"""Upload processed tables to Snowflake.

Credentials come from .env (never hard-coded, never committed). See .env.example.
If the required variables are missing, the upload is skipped with a message so
the rest of the pipeline still works.

Least privilege: the pipeline logs in as a dedicated service user whose role can
only create tables in CHURN_INSIGHTS.CORE and use warehouse CHURN_WH. The
database and schema are created once by an admin, not by this code.

Why write_pandas (and not the Spark-Snowflake connector)? write_pandas ships with
snowflake-connector-python: it saves the DataFrame as Parquet, uploads it to a
temporary Snowflake stage, and runs COPY INTO, which is fast and needs no extra
Java JARs. The Spark connector only pays off for data too big for one machine.
"""
import os
from pathlib import Path

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from snowflake.connector.pandas_tools import write_pandas

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_connection():
    """Return an open Snowflake connection, or None if credentials are missing.

    Two ways to authenticate:
      - key pair (recommended): SNOWFLAKE_PRIVATE_KEY_FILE = path to the private key.
        Snowflake stores only the public key; nothing secret is sent over the wire.
      - password: SNOWFLAKE_PASSWORD (a personal user; may require MFA).
    """
    load_dotenv(PROJECT_ROOT / ".env")
    missing = [name for name in ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER"] if not os.getenv(name)]
    if not (os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE") or os.getenv("SNOWFLAKE_PASSWORD")):
        missing.append("SNOWFLAKE_PRIVATE_KEY_FILE (or SNOWFLAKE_PASSWORD)")
    if missing:
        print(f"  Snowflake upload skipped: set {', '.join(missing)} in .env (see .env.example).")
        return None

    params = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "private_key_file": os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "CHURN_WH"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "CHURN_INSIGHTS"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "CORE"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "authenticator": os.getenv("SNOWFLAKE_AUTHENTICATOR"),
    }
    if params["private_key_file"]:
        params["authenticator"] = "SNOWFLAKE_JWT"  # Snowflake's name for key-pair auth
        params["password"] = None
    # Drop optional settings that weren't provided so Snowflake uses its defaults.
    return snowflake.connector.connect(**{k: val for k, val in params.items() if val})


def upload_tables(tables: dict[str, pd.DataFrame]) -> bool:
    conn = get_connection()
    if conn is None:
        return False

    database = os.getenv("SNOWFLAKE_DATABASE", "CHURN_INSIGHTS")
    schema = os.getenv("SNOWFLAKE_SCHEMA", "CORE")
    try:
        cur = conn.cursor()
        # The database and schema must already exist (created by an admin; see
        # README). The pipeline's role deliberately cannot create databases.
        cur.execute(f"USE SCHEMA {database}.{schema}")

        for name, df in tables.items():
            table = name.upper()
            out = df.copy()
            # Snowflake stores unquoted names in UPPERCASE. Uploading lowercase
            # column names would force quoting ("order_id") in every SQL query.
            out.columns = [c.upper() for c in out.columns]
            write_pandas(
                conn, out, table_name=table,
                auto_create_table=True,  # create the table from the DataFrame's types
                overwrite=True,          # replace on each run, so reruns are idempotent
                use_logical_type=True,   # store pandas datetimes as real TIMESTAMPs
            )
            # Verify the upload instead of trusting it.
            n_rows = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            status = "OK" if n_rows == len(out) else "ROW COUNT MISMATCH"
            print(f"  {database}.{schema}.{table}: {n_rows:,} rows [{status}]")
            if n_rows != len(out):
                raise RuntimeError(f"Snowflake table {table} has {n_rows} rows, expected {len(out)}")
    finally:
        conn.close()
    return True
