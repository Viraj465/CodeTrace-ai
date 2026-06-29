import sqlite3
from pathlib import Path

def get_db_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)

    # Keep existing constraints
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

    return conn
