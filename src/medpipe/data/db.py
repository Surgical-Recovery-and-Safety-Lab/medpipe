"""
Database functions module.

This module provides functions to open, query, and save data from databases.

Functions:
- parquet_to_db: Converts a .parquet file to a .db fil.
- extract_data_from_db: Queries a SQL .db to extract data.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from medpipe.utils.exceptions import file_checks


def parquet_to_db(parquet_file: str, db_file: str, table_name: str = "main") -> None:
    """
    Converts a .parquet file to a .db file.

    Parameters
    ----------
    parquet_file : str
        File path to the .parquet file.
    db_file : str
        File path to the .db file.
    table_name : str, default: 'main'
        Name of the table to create in the SQL database.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    TypeError
        If parquet_file or db_file are not str.
    FileNotFoundError
        If parquet_file does not exist.
    IsADirectoryError
        If parquet_file or db_file are not a file.
    ValueError
        If parquet_file extension is not a .parquet file.
    ValueError
        If db_file extension is not a .sqlite3 file.

    """
    try:
        file_checks(parquet_file, ".parquet")
    except (FileNotFoundError, IsADirectoryError, TypeError, ValueError):
        raise

    try:
        file_checks(db_file, ".db", exists=False)
    except (FileNotFoundError, TypeError, ValueError, IsADirectoryError):
        raise

    conn = sqlite3.connect(db_file)
    df = pd.read_parquet(parquet_file)

    drop_table = f"DROP TABLE IF EXISTS {table_name}"

    # Execute the queries
    conn.execute(drop_table)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()


def extract_data_from_db(db_file: str, query: str) -> pd.DataFrame:
    """
    Extracts data from a .db and saves it to a .csv file.

    The parquet file

    Parameters
    ----------
    db_file : str
        Path to the .db file.
    query : str
        Query to send to extract data.

    Returns
    -------
    data : pd.DataFrame
        Extracted data from the database.

    Raises
    ------
    TypeError
        If db_file or query is not a str.
    FileNotFoundError
        If db_file does not exist.
    IsADirectoryError
        If db_file is not a file.
    ValueError
        If db_file extension is not a .sqlite3 file.

    """
    try:
        file_checks(db_file, ".db")
    except (FileNotFoundError, IsADirectoryError, TypeError, ValueError):
        raise

    if type(query) is not str:
        raise TypeError(f"{query} should be a string")

    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df
