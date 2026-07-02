"""SQLite-backed Store for Telegram message routing and per-project state."""
from __future__ import annotations

import aiosqlite

from voice_bridge.config import ProjectConfig

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
CREATE TABLE IF NOT EXISTS usage (
    project               TEXT PRIMARY KEY,
    turns                 INTEGER NOT NULL DEFAULT 0,
    input_tokens          INTEGER NOT NULL DEFAULT 0,
    output_tokens         INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens     INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd              REAL NOT NULL DEFAULT 0
);
"""


class Store:
    """Persistent routing/state store backed by SQLite via aiosqlite."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    async def init(self) -> None:
        """Create tables if they do not exist (idempotent). No seeding."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()

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
    # session_id
    # ------------------------------------------------------------------

    async def set_session_id(self, project: str, session_id: str) -> None:
        """Set the Claude Agent session_id for a project.

        Creates the project row lazily (enabled defaults to 1) if not yet seeded.
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO projects (name, session_id) VALUES (?, ?) "
                "ON CONFLICT(name) DO UPDATE SET session_id=excluded.session_id",
                (project, session_id),
            )
            await db.commit()

    async def get_session_id(self, project: str) -> str | None:
        """Return the stored session_id for a project, or None if unset/unknown."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT session_id FROM projects WHERE name = ?", (project,)
            )
            row = await cur.fetchone()
        return row[0] if row is not None else None

    # ------------------------------------------------------------------
    # per-project token & cost usage (B3c)
    # ------------------------------------------------------------------

    async def add_usage(
        self,
        project: str,
        *,
        cost_usd: float | None,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        """Accumulate one turn's usage into *project*'s running totals.

        Upsert-accumulate: creates the row on the first call, otherwise adds
        to the existing totals (never overwrites). ``cost_usd is None`` (e.g.
        Claude Code subscription auth reports no ``total_cost_usd``) adds 0.
        """
        cost = cost_usd if cost_usd is not None else 0.0
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO usage (project, turns, input_tokens, output_tokens, "
                "cache_read_tokens, cache_creation_tokens, cost_usd) "
                "VALUES (?, 1, ?, ?, ?, ?, ?) "
                "ON CONFLICT(project) DO UPDATE SET "
                "turns = turns + 1, "
                "input_tokens = input_tokens + excluded.input_tokens, "
                "output_tokens = output_tokens + excluded.output_tokens, "
                "cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens, "
                "cache_creation_tokens = cache_creation_tokens + excluded.cache_creation_tokens, "
                "cost_usd = cost_usd + excluded.cost_usd",
                (
                    project,
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_creation_tokens,
                    cost,
                ),
            )
            await db.commit()

    async def get_usage(self, project: str) -> dict:
        """Return the accumulated usage row for *project* (zeros if none)."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT turns, input_tokens, output_tokens, cache_read_tokens, "
                "cache_creation_tokens, cost_usd FROM usage WHERE project = ?",
                (project,),
            )
            row = await cur.fetchone()
        if row is None:
            return {
                "turns": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "cost_usd": 0.0,
            }
        return _usage_row_to_dict(row)

    async def all_usage(self) -> dict[str, dict]:
        """Return the accumulated usage row for every project with usage."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT project, turns, input_tokens, output_tokens, "
                "cache_read_tokens, cache_creation_tokens, cost_usd FROM usage"
            )
            rows = await cur.fetchall()
        return {row[0]: _usage_row_to_dict(row[1:]) for row in rows}


def _usage_row_to_dict(row) -> dict:
    turns, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, cost_usd = row
    return {
        "turns": turns,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cost_usd": cost_usd,
    }
