"""
SQL command utilities.
"""
import sqlite3
from typing import List, Tuple

def list_tables(conn: sqlite3.Connection) -> List[str]:
    """List all tables in a SQLite database."""
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [row[0] for row in cur.fetchall()]


def get_table_schema(conn: sqlite3.Connection, table: str) -> List[Tuple]:
    """Get schema info of a table."""
    cur = conn.execute(f"PRAGMA table_info({table});")
    return cur.fetchall()


def sample_rows_ignore_blob(conn: sqlite3.Connection, table: str, limit: int = 10) -> List[Tuple]:
    """Get sample rows from a table. Ignore BLOB columns."""
    # 1. 取列信息：cid, name, type, ...
    cur_result = get_table_schema(conn, table)
    keep_cols = []
    for _, name, typ, _, _, _ in cur_result:
        if "blob" not in typ.lower():          # 忽略大小写
            keep_cols.append(name)

    if not keep_cols:          # 整张表全是 BLOB
        return []

    placeholders = ",".join(f'"{c}"' for c in keep_cols)
    cur = conn.execute(f'SELECT {placeholders} FROM "{table}" LIMIT {limit}')
    return cur.fetchall()

