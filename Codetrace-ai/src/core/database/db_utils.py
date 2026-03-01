import sqlite3
from pathlib import Path

def get_db_connection(db_path: str | Path) -> sqlite3.Connection:
    """
    Ensures the parent directory exists and returns a 
    WAL-enabled, foreign-key enforced SQLite connection.
    """
    path = Path(db_path)
    
    # Create the parent directory (.codetrace) if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect directly to the full database path
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    
    return conn