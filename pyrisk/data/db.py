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
    parquet_file: str, duckdb_file: str, table_name: str = "main"
) -> None:
    """
    Converts a .parquet file to a .duckdb file.

    Parameters
    ----------
    parquet_file
        File path to the .parquet file.
    duckdb_file
        File path to the .duckdb file.
    table_name, default: 'main'
        Name of the table to create in the duckdb database.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    TypeError
        If parquet_file or duckdb_file are not str.
    FileNotFoundError
        If parquet_file does not exist.
    IsADirectoryError
        If parquet_file or duckdb_file are not a file.
    ValueError
        If parquet_file extension is not a .parquet file.
    ValueError
        If duckdb_file extension is not a .duckdb file.

    """
    try:
        pyrisk.utils.exceptions.file_checks(parquet_file, ".parquet")
    except (FileNotFoundError, IsADirectoryError, TypeError, ValueError):
        raise

    try:
        pyrisk.utils.exceptions.file_checks(duckdb_file, ".duckdb")
    except (TypeError, ValueError, IsADirectoryError):
        raise
    except FileNotFoundError:
        # Duckdb will be created so pass this exception
        pass

    conn = duckdb.connect(duckdb_file)

    drop_table = f"DROP TABLE IF EXISTS {table_name}"
    query = f"CREATE TABLE {table_name} AS SELECT * FROM '{parquet_file}'"

    # Execute the queries
    conn.execute(drop_table)
    conn.execute(query)
    conn.close()


def extract_data_from_duckdb(duckdb_file: str, query: str):
    """
    Extracts data from a duckdb and saves it to a .csv file.

    Parameters
    ----------
    duckdb_file
        Path to the duckdb file.
    query
        Query to send to the duckdb to extract data.

    Returns
    -------
    data : pd.DataFrame
        Extracted data from the duckdb database.

    Raises
    ------
    TypeError
        If duckdb_file or query is not a str.
    FileNotFoundError
        If duckdb_file does not exist.
    IsADirectoryError
        If duckdb_file is not a file.
    ValueError
        If duckdb_file extension is not a .duckdb file.

    """
    try:
        pyrisk.utils.exceptions.file_checks(duckdb_file, ".duckdb")
    except (FileNotFoundError, IsADirectoryError, TypeError, ValueError):
        raise

    if type(query) is not str:
        raise TypeError(f"{query} should be a string")

    conn = duckdb.connect(duckdb_file)

    df = conn.execute(query).df()  # Get the data
    conn.close()

    return df
