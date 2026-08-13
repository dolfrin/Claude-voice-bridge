"""SQLite-backed Store for Telegram message routing and per-project state."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import aiosqlite

from voice_bridge.config import ProjectConfig

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    message_id INTEGER PRIMARY KEY,
    project    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS projects (
    name       TEXT PRIMARY KEY,
    enabled    INTEGER NOT NULL DEFAULT 1,
    session_id TEXT
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS topics (
    project   TEXT PRIMARY KEY,
    chat_id   INTEGER NOT NULL,
    thread_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    key        TEXT PRIMARY KEY,
    project    TEXT NOT NULL,
    ordinal    INTEGER NOT NULL,
    chat_id    INTEGER,
    thread_id  INTEGER,
    session_id TEXT,
    closed     INTEGER NOT NULL DEFAULT 0,
    created    REAL NOT NULL DEFAULT 0
);
"""


@dataclass(frozen=True)
class Conversation:
    """One agent conversation: its own forum topic, its own Claude session."""

    key: str  # "paprika#1" — the id every other module routes on
    project: str
    ordinal: int
    chat_id: int | None
    thread_id: int | None
    session_id: str | None
    closed: bool
    created: float

    @property
    def label(self) -> str:
        return f"#{self.ordinal}"


def conversation_key(project: str, ordinal: int) -> str:
    return f"{project}#{ordinal}"


def project_of(key: str) -> str:
    """The project a conversation key belongs to (a plain name maps to itself)."""
    return key.split("#", 1)[0]


class Store:
    """Persistent routing/state store backed by SQLite via aiosqlite."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    async def init(self) -> None:
        """Create tables if they do not exist (idempotent). No seeding."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()
            await self._migrate_project_sessions(db)

    @staticmethod
    async def _migrate_project_sessions(db) -> None:
        """Turn a pre-conversations database into ``<project>#1`` rows.

        Sessions used to be one per project, stored on the project row. Without
        this, the first start after the upgrade would open empty conversations
        and every running project would silently lose its context.

        Runs once: any conversation row at all, open or closed, means the
        database has already moved over.
        """
        cur = await db.execute("SELECT COUNT(*) FROM conversations")
        if (await cur.fetchone())[0]:
            return
        cur = await db.execute("SELECT name, session_id FROM projects")
        rows = await cur.fetchall()
        if not rows:
            return
        await db.executemany(
            "INSERT INTO conversations "
            "(key, project, ordinal, session_id, closed, created) "
            "VALUES (?, ?, 1, ?, 0, 0)",
            [(f"{name}#1", name, session_id) for name, session_id in rows],
        )
        await db.commit()
        logger.info("migrated %d projects to conversations", len(rows))

    async def seed(self, projects: list[ProjectConfig]) -> None:
        """INSERT OR IGNORE a row per project using its enabled default.

        Never overwrites existing rows, so user toggles persist across restarts.
        """
        async with aiosqlite.connect(self.db_path) as db:
            for p in projects:
                await db.execute(
                    "INSERT OR IGNORE INTO projects (name, enabled) VALUES (?, ?)",
                    (p.name, 1 if p.enabled else 0),
                )
            await db.commit()

    # ------------------------------------------------------------------
    # Message <-> project mapping
    # ------------------------------------------------------------------

    async def map_message(self, message_id: int, project: str) -> None:
        """Map a Telegram message_id to a project name (upsert)."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO messages (message_id, project) VALUES (?, ?) "
                "ON CONFLICT(message_id) DO UPDATE SET project=excluded.project",
                (message_id, project),
            )
            await db.commit()

    async def project_for_message(self, message_id: int) -> str | None:
        """Return the project name for a message_id, or None if unknown."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT project FROM messages WHERE message_id = ?", (message_id,)
            )
            row = await cur.fetchone()
        return row[0] if row is not None else None

    # ------------------------------------------------------------------
    # last_active (meta table)
    # ------------------------------------------------------------------

    async def set_last_active(self, project: str) -> None:
        """Record the most-recently-active project name in the meta table."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO meta (key, value) VALUES ('last_active', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (project,),
            )
            await db.commit()

    async def get_last_active(self) -> str | None:
        """Return the most-recently-active project name, or None if unset."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT value FROM meta WHERE key = 'last_active'"
            )
            row = await cur.fetchone()
        return row[0] if row is not None else None

    async def set_meta(self, key: str, value: str) -> None:
        """Persist one small bridge-owned value (see :meth:`get_meta`)."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            await db.commit()

    async def get_meta(self, key: str) -> str | None:
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("SELECT value FROM meta WHERE key = ?", (key,))
            row = await cur.fetchone()
        return row[0] if row is not None else None

    # ------------------------------------------------------------------
    # enabled flag
    # ------------------------------------------------------------------

    async def set_enabled(self, project: str, enabled: bool) -> None:
        """Set the enabled flag for a project (upsert)."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO projects (name, enabled) VALUES (?, ?) "
                "ON CONFLICT(name) DO UPDATE SET enabled=excluded.enabled",
                (project, 1 if enabled else 0),
            )
            await db.commit()

    async def is_enabled(self, project: str) -> bool:
        """Return the stored enabled value; False for unknown/unseeded projects."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT enabled FROM projects WHERE name = ?", (project,)
            )
            row = await cur.fetchone()
        return bool(row[0]) if row is not None else False

    async def enabled_map(self) -> dict[str, bool]:
        """Return a dict mapping every known project name to its enabled flag."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("SELECT name, enabled FROM projects")
            rows = await cur.fetchall()
        return {name: bool(enabled) for name, enabled in rows}

    # ------------------------------------------------------------------
    # conversations (one per agent session / forum sub-topic)
    # ------------------------------------------------------------------

    async def next_ordinal(self, project: str) -> int:
        """The next ``#N`` for a project. Never reuses a number, so a closed
        ``#2`` does not come back as a different conversation with the same
        topic title."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT MAX(ordinal) FROM conversations WHERE project = ?", (project,)
            )
            row = await cur.fetchone()
        return (row[0] or 0) + 1

    async def add_conversation(
        self,
        key: str,
        project: str,
        ordinal: int,
        *,
        chat_id: int | None = None,
        thread_id: int | None = None,
        session_id: str | None = None,
        created: float = 0.0,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO conversations "
                "(key, project, ordinal, chat_id, thread_id, session_id, closed, created) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, ?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "chat_id=excluded.chat_id, thread_id=excluded.thread_id, "
                "session_id=excluded.session_id, closed=0",
                (key, project, ordinal, chat_id, thread_id, session_id, created),
            )
            await db.commit()

    async def set_conversation_topic(
        self, key: str, chat_id: int, thread_id: int
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE conversations SET chat_id = ?, thread_id = ? WHERE key = ?",
                (chat_id, thread_id, key),
            )
            await db.commit()

    async def set_conversation_session(self, key: str, session_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE conversations SET session_id = ? WHERE key = ?",
                (session_id, key),
            )
            await db.commit()

    async def close_conversation(self, key: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE conversations SET closed = 1 WHERE key = ?", (key,)
            )
            await db.commit()

    async def conversation(self, key: str) -> Conversation | None:
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT key, project, ordinal, chat_id, thread_id, session_id, "
                "closed, created FROM conversations WHERE key = ?",
                (key,),
            )
            row = await cur.fetchone()
        return _conversation(row) if row is not None else None

    async def conversations(
        self, project: str | None = None, include_closed: bool = False
    ) -> list[Conversation]:
        sql = (
            "SELECT key, project, ordinal, chat_id, thread_id, session_id, "
            "closed, created FROM conversations"
        )
        where: list[str] = []
        args: list = []
        if project is not None:
            where.append("project = ?")
            args.append(project)
        if not include_closed:
            where.append("closed = 0")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY project, ordinal"
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(sql, args)
            rows = await cur.fetchall()
        return [_conversation(row) for row in rows]

    async def conversation_for_session(self, session_id: str) -> Conversation | None:
        """Which conversation already owns a Claude session uuid.

        Resuming a session that is already open in a topic must land the user in
        that topic instead of spawning a second writer for the same .jsonl.
        """
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT key, project, ordinal, chat_id, thread_id, session_id, "
                "closed, created FROM conversations WHERE session_id = ? "
                "ORDER BY closed, ordinal DESC LIMIT 1",
                (session_id,),
            )
            row = await cur.fetchone()
        return _conversation(row) if row is not None else None

    # ------------------------------------------------------------------
    # forum topics
    # ------------------------------------------------------------------

    async def set_topic(self, project: str, chat_id: int, thread_id: int) -> None:
        """Remember which forum topic belongs to a project."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO topics (project, chat_id, thread_id) VALUES (?, ?, ?) "
                "ON CONFLICT(project) DO UPDATE SET "
                "chat_id=excluded.chat_id, thread_id=excluded.thread_id",
                (project, chat_id, thread_id),
            )
            await db.commit()

    async def topics_for_chat(self, chat_id: int) -> dict[str, int]:
        """Return ``{project: thread_id}`` for one chat.

        Scoped by chat so a recreated group does not leave the bot posting into
        thread ids that belong to a group it is no longer in.
        """
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT project, thread_id FROM topics WHERE chat_id = ?", (chat_id,)
            )
            rows = await cur.fetchall()
        return {row[0]: row[1] for row in rows}


def _conversation(row) -> Conversation:
    return Conversation(
        key=row[0],
        project=row[1],
        ordinal=row[2],
        chat_id=row[3],
        thread_id=row[4],
        session_id=row[5],
        closed=bool(row[6]),
        created=row[7] or 0.0,
    )
