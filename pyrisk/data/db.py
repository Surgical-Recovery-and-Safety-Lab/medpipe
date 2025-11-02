"""
Database functions module.

This module provides functions to open, query, and save data from databases.

Functions:
-
-
"""

import duckdb

import pyrisk.utils.exceptions


def parquet_to_duckdb(
    parquet_path: str, duckdb_path: str, table_name: str = "main"
) -> None:
    """
    Converts a .parquet file to a .duckdb file.

    Parameters:
    ----------
    parquet_path
        File path to the .parquet file.
    duckdb_path
        File path to the .duckdb file.
    table_name, default: 'main'
        Name of the table to create in the duckdb database.

    Returns:
    -------
    None
        Nothing is returned.

    Raises:
    ------
    TypeError
        If parquet_path or duckdb_path are not str.
    FileNotFoundError
        If parquet_path or duckdb_path do not exist.
    ValueError
        If parquet_path extension is not a .parquet file.
    ValueError
        If duckdb_path extension is not a .duckdb file.

    """
    try:
        pyrisk.utils.exceptions.path_checks(parquet_path, ".parquet")
    except (FileNotFoundError, TypeError, ValueError):
        raise

    try:
        pyrisk.utils.exceptions.path_checks(duckdb_path, ".duckdb")
    except (TypeError, ValueError):
        raise
    except FileNotFoundError:
        # Duckdb will be created so pass this exception
        pass

    conn = duckdb.connect(duckdb_path)

    drop_table = f"DROP TABLE IF EXISTS {table_name}"
    query = f"CREATE TABLE {table_name} AS SELECT * FROM '{parquet_path}'"

    # Execute the query
    conn.execute(drop_table)
    conn.execute(query)
    conn.close()
