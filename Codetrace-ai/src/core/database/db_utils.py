import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def get_db_connection(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """
    Yield a configured SQLite connection and guarantee it is closed.

    This is a context manager (``with get_db_connection(path) as conn:``).
    The previous implementation returned a bare connection; because
    ``sqlite3.Connection.__exit__`` only commits/rolls back the transaction
    and never closes the connection, every call leaked a connection (and, in
    WAL mode, file handles). Closing in ``finally`` fixes that leak.

    Callers remain responsible for calling ``conn.commit()`` on writes; if the
    ``with`` body raises, the connection is closed without committing, which
    discards the uncommitted transaction (an effective rollback).
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    try:
        # Correctness
        conn.execute("PRAGMA journal_mode=WAL")        # Concurrent reads and writes
        conn.execute("PRAGMA foreign_keys=ON")         # Enforce foreign key constraints

        # Performance optimizations
        conn.execute("PRAGMA synchronous=NORMAL")      # Safe with WAL, faster writes
        conn.execute("PRAGMA cache_size=-64000")       # 64MB page cache in RAM
        conn.execute("PRAGMA temp_store=MEMORY")       # Temporary tables in RAM
        conn.execute("PRAGMA mmap_size=268435456")     # 256MB memory-mapped reads

        # Maintenance
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL") # Reclaim space from deletes
        conn.execute("PRAGMA optimize")                # Freshen query planner stats

        yield conn
    finally:
        conn.close()
