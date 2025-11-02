"""
Database functions module.

This module provides functions to open, query, and save data from databases.

Functions:
- parquet_to_duckdb: Converts a .parquet file to a duckdb.
- extract_data_from_duckdb: Queries a duckdb to extract data.
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

    # Execute the queries
    conn.execute(drop_table)
    conn.execute(query)
    conn.close()


def extract_data_from_duckdb(duckdb_path: str, query: str):
    """
    Extracts data from a duckdb and saves it to a .csv file.

    Parameters:
    ----------
    duckdb_path
        Path to the duckdb file.
    query
        Query to send to the duckdb to extract data.

    Returns:
    -------
    data : pd.DataFrame
        Extracted data from the duckdb database.

    Raises:
    ------
    TypeError
        If duckdb_path or query is not a str.
    FileNotFoundError
        If duckdb_path does not exist.
    ValueError
        If duckdb_path extension is not a .duckdb file.

    """
    try:
        pyrisk.utils.exceptions.path_checks(duckdb_path, ".duckdb")
    except (FileNotFoundError, TypeError, ValueError):
        raise

    if type(query) is not str:
        raise TypeError(f"{query} should be a string")

    conn = duckdb.connect(duckdb_path)

    df = conn.execute(query).df()  # Get the data
    conn.close()

    return df
