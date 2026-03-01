"""
ChatStore: SQLite-backed persistent conversation history.

Stores chat sessions per-project in `.codetrace/chat_history.db`.
Each session is a sequence of user/assistant messages that can be
resumed, searched, and exported.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


class ChatStore:
    """Manages persistent chat sessions in SQLite."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                title       TEXT DEFAULT '',
                created_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now')),
                project     TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL REFERENCES sessions(id),
                role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content     TEXT NOT NULL,
                timestamp   TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id);
        """)
        self.conn.commit()

    
    # Session management
    

    def create_session(self, project: str = "") -> str:
        """Create a new chat session. Returns the session ID."""
        session_id = str(uuid.uuid4())[:8]
        self.conn.execute(
            "INSERT INTO sessions (id, project) VALUES (?, ?)",
            (session_id, project),
        )
        self.conn.commit()
        return session_id

    def get_latest_session_id(self) -> Optional[str]:
        """Return the most recent session ID, or None."""
        row = self.conn.execute(
            "SELECT id FROM sessions ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        return row["id"] if row else None

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """List recent sessions with metadata."""
        rows = self.conn.execute("""
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   COUNT(m.id) as message_count,
                   (SELECT content FROM messages
                    WHERE session_id = s.id AND role = 'user'
                    ORDER BY id ASC LIMIT 1) as first_query
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [
            {
                "id": r["id"],
                "title": r["title"] or (r["first_query"][:60] + "..." if r["first_query"] and len(r["first_query"]) > 60 else r["first_query"] or "Empty session"),
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "message_count": r["message_count"],
            }
            for r in rows
        ]

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        row = self.conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row is not None

    
    # Message management
    

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to a session."""
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        # Update session timestamp and auto-title from first user message
        self.conn.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?",
            (session_id,),
        )
        self.conn.commit()

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict]:
        """Get messages for a session, optionally limited to the last N."""
        if limit:
            # Get the last `limit` messages (in chronological order)
            rows = self.conn.execute("""
                SELECT role, content FROM (
                    SELECT role, content, id FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) sub ORDER BY id ASC
            """, (session_id, limit)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()

        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def get_history_for_llm(self, session_id: str, max_turns: int = 5) -> list[tuple[str, str]]:
        """Get chat history formatted for LangChain's MessagesPlaceholder.

        Returns a list of (role, content) tuples for the last `max_turns`
        exchanges (each exchange = 1 user msg + 1 assistant msg = 2 messages).
        """
        messages = self.get_messages(session_id, limit=max_turns * 2)
        return [(m["role"], m["content"]) for m in messages]

    
    # Search & Export
    

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search across all sessions for messages matching the query."""
        rows = self.conn.execute("""
            SELECT m.role, m.content, m.timestamp, s.id as session_id,
                   (SELECT content FROM messages
                    WHERE session_id = s.id AND role = 'user'
                    ORDER BY id ASC LIMIT 1) as session_title
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE m.content LIKE ?
            ORDER BY m.timestamp DESC
            LIMIT ?
        """, (f"%{query}%", limit)).fetchall()

        return [
            {
                "session_id": r["session_id"],
                "session_title": r["session_title"],
                "role": r["role"],
                "content": r["content"][:200],
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

    def export_session(self, session_id: str) -> str:
        """Export a session as a Markdown document."""
        messages = self.get_messages(session_id)
        if not messages:
            return "Empty session."

        session = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()

        lines = [
            f"# Codetrace Chat Session",
            f"**Session:** {session_id}",
            f"**Date:** {session['created_at']}",
            f"",
            "---",
            "",
        ]

        for msg in messages:
            if msg["role"] == "user":
                lines.append(f"## 🧑 User\n\n{msg['content']}\n")
            else:
                lines.append(f"## 🤖 Architect\n\n{msg['content']}\n")

        return "\n".join(lines)

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
