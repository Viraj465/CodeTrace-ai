"""
SyncManager: Handles delta updates using SQLite to prevent redundant parsing and embedding.
Tracks file hashes and modification timestamps to identify changes.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, List, Any

from src.core.database.db_utils import get_db_connection

logger = logging.getLogger(__name__)


class SyncManager:

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS file_hashes (
        filepath      TEXT PRIMARY KEY,
        filehash      TEXT NOT NULL,
        last_updated  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS file_snapshots (
        filepath      TEXT PRIMARY KEY,
        filehash      TEXT NOT NULL,
        content       TEXT NOT NULL,
        line_count    INTEGER NOT NULL,
        size_bytes    INTEGER NOT NULL,
        is_truncated  INTEGER NOT NULL DEFAULT 0,
        indexed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS index_metadata (
        key           TEXT PRIMARY KEY,
        value         TEXT NOT NULL,
        updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_last_updated ON file_hashes(last_updated);
    CREATE INDEX IF NOT EXISTS idx_snapshots_indexed_at ON file_snapshots(indexed_at);
    """

    MAX_SNAPSHOT_BYTES = 1_000_000

    def __init__(self, db_dir: str = ".codetrace") -> None:
        db_path = Path(db_dir)
        db_path.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path / "sync_metadata.db"
        self._init_db()

    def _init_db(self) -> None:
        with get_db_connection(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
            conn.commit()

    def _compute_file_hash(
        self, filepath: str | Path, chunk_size: int = 8192
    ) -> Optional[str]:
        path = Path(filepath)
        if not path.exists() or not path.is_file():
            return None
        file_hash = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(chunk_size):
                    file_hash.update(chunk)
            return file_hash.hexdigest()
        except Exception as e:
            logger.error("Error hashing file %s: %s", filepath, e)
            return None

    def _is_probably_text(self, data: bytes) -> bool:
        if not data:
            return True
        if b"\x00" in data:
            return False
        control_bytes = sum(1 for b in data if b < 9 or (13 < b < 32))
        return (control_bytes / max(1, len(data))) < 0.30

    def has_file_changed(
        self, filepath: str | Path
    ) -> tuple[bool, Optional[str]]:
        """
        Returns (changed: bool, current_hash: str | None).
        Caller should pass the hash to mark_file_synced to avoid re-hashing.
        """
        current_hash = self._compute_file_hash(filepath)
        if not current_hash:
            return False, None

        with get_db_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT filehash FROM file_hashes WHERE filepath = ?",
                (str(filepath),)
            ).fetchone()

        changed = row is None or row[0] != current_hash
        return changed, current_hash

    def mark_file_synced(
        self, filepath: str | Path, file_hash: Optional[str] = None
    ) -> None:
        current_hash = file_hash or self._compute_file_hash(filepath)
        if not current_hash:
            return

        with get_db_connection(self.db_path) as conn:
            conn.execute("""
                INSERT INTO file_hashes (filepath, filehash, last_updated)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(filepath) DO UPDATE SET
                    filehash=excluded.filehash,
                    last_updated=CURRENT_TIMESTAMP
            """, (str(filepath), current_hash))
            conn.commit()

    def remove_file_record(self, filepath: str | Path) -> None:
        with get_db_connection(self.db_path) as conn:
            conn.execute(
                "DELETE FROM file_hashes WHERE filepath = ?", (str(filepath),)
            )
            conn.commit()

    def upsert_file_snapshot(
        self, filepath: str | Path, content: str, file_hash: Optional[str] = None
    ) -> None:
        current_hash = file_hash or self._compute_file_hash(filepath)
        if not current_hash:
            return

        encoded = content.encode("utf-8", errors="replace")
        is_truncated = 1 if len(encoded) > self.MAX_SNAPSHOT_BYTES else 0
        if is_truncated:
            encoded = encoded[: self.MAX_SNAPSHOT_BYTES]
            content = encoded.decode("utf-8", errors="replace")

        line_count = len(content.splitlines())
        with get_db_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO file_snapshots (filepath, filehash, content, line_count, size_bytes, is_truncated, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(filepath) DO UPDATE SET
                    filehash=excluded.filehash,
                    content=excluded.content,
                    line_count=excluded.line_count,
                    size_bytes=excluded.size_bytes,
                    is_truncated=excluded.is_truncated,
                    indexed_at=CURRENT_TIMESTAMP
                """,
                (str(filepath), current_hash, content, line_count, len(encoded), is_truncated),
            )
            conn.commit()

    def upsert_file_snapshot_from_disk(
        self, filepath: str | Path, file_hash: Optional[str] = None
    ) -> None:
        path = Path(filepath)
        if not path.exists() or not path.is_file():
            return
        try:
            raw = path.read_bytes()
        except Exception as e:
            logger.warning("Skipping snapshot for %s: %s", filepath, e)
            return

        if not self._is_probably_text(raw):
            return

        content = raw.decode("utf-8", errors="replace")
        self.upsert_file_snapshot(filepath, content, file_hash=file_hash)

    def remove_file_snapshot(self, filepath: str | Path) -> None:
        with get_db_connection(self.db_path) as conn:
            conn.execute(
                "DELETE FROM file_snapshots WHERE filepath = ?", (str(filepath),)
            )
            conn.commit()

    def get_file_snapshot(self, filepath: str | Path) -> Optional[dict[str, Any]]:
        fp = str(filepath)
        with get_db_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT filepath, filehash, content, line_count, size_bytes, is_truncated
                FROM file_snapshots
                WHERE filepath = ?
                """,
                (fp,),
            ).fetchone()

            if row is None:
                row = conn.execute(
                    """
                    SELECT filepath, filehash, content, line_count, size_bytes, is_truncated
                    FROM file_snapshots
                    WHERE filepath LIKE ?
                    ORDER BY LENGTH(filepath) ASC
                    LIMIT 1
                    """,
                    (f"%{fp}",),
                ).fetchone()

        if not row:
            return None
        return {
            "filepath": row[0],
            "filehash": row[1],
            "content": row[2],
            "line_count": row[3],
            "size_bytes": row[4],
            "is_truncated": bool(row[5]),
        }

    def list_indexed_files(self, query: str = "", limit: int = 200) -> list[str]:
        safe_limit = max(1, min(limit, 1000))
        with get_db_connection(self.db_path) as conn:
            if query:
                rows = conn.execute(
                    """
                    SELECT filepath
                    FROM file_snapshots
                    WHERE filepath LIKE ?
                    ORDER BY filepath ASC
                    LIMIT ?
                    """,
                    (f"%{query}%", safe_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT filepath
                    FROM file_snapshots
                    ORDER BY filepath ASC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
        return [row[0] for row in rows]

    def set_metadata(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        with get_db_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO index_metadata (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (key, payload),
            )
            conn.commit()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        with get_db_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM index_metadata WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return default
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return default

    def update_index_manifest(
        self,
        project_root: str | Path,
        file_hash_pairs: list[tuple[str, str]],
        supported_file_count: int,
    ) -> None:
        serial = "\n".join(f"{fp}:{h}" for fp, h in sorted(file_hash_pairs))
        manifest_hash = hashlib.sha256(serial.encode("utf-8")).hexdigest()
        self.set_metadata("project_root", str(Path(project_root).resolve()))
        self.set_metadata("supported_file_count", supported_file_count)
        self.set_metadata("tracked_snapshot_count", len(file_hash_pairs))
        self.set_metadata("manifest_hash", manifest_hash)

    def get_changed_files(self, filepaths: List[str]) -> List[tuple[str, str]]:
        """
        Check many files in ONE db query.
        Returns list of (filepath, hash) for files that are new or modified.
        """
        if not filepaths:
            return []

        placeholders = ",".join("?" * len(filepaths))
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT filepath, filehash FROM file_hashes WHERE filepath IN ({placeholders})",
                [str(p) for p in filepaths]
            ).fetchall()

        db_hashes = {r[0]: r[1] for r in rows}
        changed = []
        for fp in filepaths:
            current_hash = self._compute_file_hash(fp)
            if current_hash and current_hash != db_hashes.get(str(fp)):
                changed.append((str(fp), current_hash))
        return changed

    def mark_files_synced_batch(self, file_hash_pairs: List[tuple[str, str]]) -> None:
        """
        Upsert many records in a SINGLE transaction - 10x faster than looping.
        file_hash_pairs: list of (filepath, hash)
        """
        if not file_hash_pairs:
            return
        with get_db_connection(self.db_path) as conn:
            conn.executemany("""
                INSERT INTO file_hashes (filepath, filehash, last_updated)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(filepath) DO UPDATE SET
                    filehash=excluded.filehash,
                    last_updated=CURRENT_TIMESTAMP
            """, file_hash_pairs)
            conn.commit()
        logger.info("Batch synced %d files.", len(file_hash_pairs))

    def get_all_tracked_file_hashes(self) -> list[tuple[str, str]]:
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT filepath, filehash FROM file_hashes ORDER BY filepath ASC"
            ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def get_deleted_files(self, current_files: List[str]) -> List[str]:
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute("SELECT filepath FROM file_hashes").fetchall()

        db_files = {r[0] for r in rows}
        current_set = {str(Path(f)) for f in current_files}
        return list(db_files - current_set)
