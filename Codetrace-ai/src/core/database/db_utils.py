# import sqlite3
# from pathlib import Path

# def get_db_connection(db_path: str | Path) -> sqlite3.Connection:
#     """
#     Ensures the parent directory exists and returns a 
#     WAL-enabled, foreign-key enforced SQLite connection.
#     """
#     path = Path(db_path)
    
#     # Create the parent directory (.codetrace) if it doesn't exist
#     path.parent.mkdir(parents=True, exist_ok=True)
    
#     # Connect directly to the full database path
#     conn = sqlite3.connect(path)
#     conn.execute("PRAGMA journal_mode=WAL")
#     conn.execute("PRAGMA foreign_keys=ON")
    
#     return conn


import sqlite3
from pathlib import Path

def get_db_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)

    # --- Existing (keep) ---
    conn.execute("PRAGMA journal_mode=WAL")       # concurrent reads + writes
    conn.execute("PRAGMA foreign_keys=ON")         # enforce FK constraints

    # --- Performance ---
    conn.execute("PRAGMA synchronous=NORMAL")      # safe with WAL, faster writes
    conn.execute("PRAGMA cache_size=-64000")       # 64MB page cache in RAM
    conn.execute("PRAGMA temp_store=MEMORY")       # temp tables in RAM
    conn.execute("PRAGMA mmap_size=268435456")     # 256MB memory-mapped reads

    # --- Maintenance ---
    conn.execute("PRAGMA auto_vacuum=INCREMENTAL") # reclaim space from deletes
    conn.execute("PRAGMA optimize")                # freshen query planner stats

    return conn
