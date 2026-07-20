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
    name         TEXT PRIMARY KEY,
    enabled      INTEGER NOT NULL DEFAULT 1,
    session_id   TEXT,
    autonomy     TEXT,
    voice        TEXT,
    verbose      INTEGER,
    effort       TEXT,
    cwd          TEXT,
    display_name TEXT,
    created      INTEGER NOT NULL DEFAULT 0
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
CREATE TABLE IF NOT EXISTS approval_policy (
    project   TEXT NOT NULL,
    signature TEXT NOT NULL,
    PRIMARY KEY (project, signature)
);
"""

# Columns the ``projects`` table must carry. A fresh db gets them from _SCHEMA;
# an EXISTING db (created by an older schema) is migrated additively in init()
# via ``ALTER TABLE ... ADD COLUMN`` so no data is lost. Column names here are
# a fixed internal allow-list — never user input — so interpolating them into
# DDL/DML is safe.
_PROJECT_COLUMNS: dict[str, str] = {
    "autonomy": "TEXT",
    "voice": "TEXT",
    "verbose": "INTEGER",
    "effort": "TEXT",
    "cwd": "TEXT",
    "display_name": "TEXT",
    "created": "INTEGER NOT NULL DEFAULT 0",
}

# Whitelisted runtime-override fields (see :meth:`Store.set_override`). Each maps
# a per-project ProjectConfig attribute that a live /mode /voice /verbose /effort
# command may mutate and that must survive a restart. NOT model (kept yaml-only).
_OVERRIDE_FIELDS: frozenset[str] = frozenset(
    {"autonomy", "voice", "verbose", "effort"}
)


class Store:
    """Persistent routing/state store backed by SQLite via aiosqlite."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    async def init(self) -> None:
        """Create tables if they do not exist and migrate additively (idempotent).

        A fresh db gets the full schema. An EXISTING db created by an older
        schema is migrated in place: any missing ``projects`` column from
        :data:`_PROJECT_COLUMNS` is added via ``ALTER TABLE ... ADD COLUMN``,
        which preserves every existing row and its state (enabled toggles,
        session_ids). No seeding.
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
            await self._migrate_projects(db)
            await db.commit()

    async def _migrate_projects(self, db: aiosqlite.Connection) -> None:
        """Add any missing override/created columns to an existing projects table."""
        cur = await db.execute("PRAGMA table_info(projects)")
        existing = {row[1] for row in await cur.fetchall()}
        for column, decl in _PROJECT_COLUMNS.items():
            if column not in existing:
                await db.execute(
                    f"ALTER TABLE projects ADD COLUMN {column} {decl}"
                )

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
    # per-project runtime overrides (autonomy/voice/verbose/effort)
    # ------------------------------------------------------------------

    async def set_override(self, project: str, field: str, value) -> None:
        """Persist a per-project runtime override (upsert).

        ``field`` must be one of the whitelisted :data:`_OVERRIDE_FIELDS`
        (raises ``ValueError`` otherwise, since an unexpected field is a
        programming error). ``verbose`` is normalized to 0/1; ``value=None``
        clears the override (NULL = "use the yaml/config default"). Creates the
        project row lazily (enabled defaults to 1) if not yet seeded.
        """
        if field not in _OVERRIDE_FIELDS:
            raise ValueError(f"unknown override field: {field!r}")
        if field == "verbose" and value is not None:
            value = 1 if value else 0
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"INSERT INTO projects (name, {field}) VALUES (?, ?) "
                f"ON CONFLICT(name) DO UPDATE SET {field}=excluded.{field}",
                (project, value),
            )
            await db.commit()

    async def overrides(self) -> dict[str, dict]:
        """Return per-project overrides, each dict holding only non-null fields.

        Projects with no override at all are omitted entirely. ``verbose`` is
        returned as a bool. Precedence is decided by the caller (persisted
        override > yaml)."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT name, autonomy, voice, verbose, effort FROM projects"
            )
            rows = await cur.fetchall()
        out: dict[str, dict] = {}
        for name, autonomy, voice, verbose, effort in rows:
            fields: dict = {}
            if autonomy is not None:
                fields["autonomy"] = autonomy
            if voice is not None:
                fields["voice"] = voice
            if verbose is not None:
                fields["verbose"] = bool(verbose)
            if effort is not None:
                fields["effort"] = effort
            if fields:
                out[name] = fields
        return out

    # ------------------------------------------------------------------
    # dynamically-created projects (/newproject persistence)
    # ------------------------------------------------------------------

    async def add_created_project(
        self, name: str, cwd: str, display_name: str | None
    ) -> None:
        """Persist a runtime-created project so it is reloaded across restarts.

        Marks the row ``created=1`` and records its cwd/display_name. On a
        re-register of the same name the cwd/display_name are refreshed but the
        ``enabled`` toggle is preserved (a user who disabled it stays disabled).
        A brand-new row defaults to enabled=1 so it boots on the next restart.
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO projects (name, cwd, display_name, created, enabled) "
                "VALUES (?, ?, ?, 1, 1) "
                "ON CONFLICT(name) DO UPDATE SET "
                "cwd=excluded.cwd, display_name=excluded.display_name, created=1",
                (name, cwd, display_name),
            )
            await db.commit()

    async def created_projects(self) -> list[dict]:
        """Return every ``created=1`` project as ``{name, cwd, display_name}``."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT name, cwd, display_name FROM projects WHERE created = 1"
            )
            rows = await cur.fetchall()
        return [
            {"name": name, "cwd": cwd, "display_name": display_name}
            for name, cwd, display_name in rows
        ]

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


    # ------------------------------------------------------------------
    # always-allow approval policies (SECURITY): (project, signature) pairs
    # ------------------------------------------------------------------
    #
    # A policy short-circuits an approval prompt that WOULD otherwise be shown
    # for a matching risky/asked tool call in the same project. The
    # ``signature`` is a stable, action-specific descriptor derived by
    # :func:`voice_bridge.approvals.signature_for` (e.g. ``"git push"``,
    # ``"rm"``, ``"send_file"``) — never just the tool name — so a grant stays
    # scoped to what the user actually approved. The key space is deliberately
    # small (a project + a short signature string); no arbitrary user text is
    # interpolated into DDL. All methods are best-effort at the call site: a
    # write failure must never break the approval flow, and a failed
    # ``has_policy`` read must fail SAFE (caller falls through to prompting).

    async def add_policy(self, project: str, signature: str) -> None:
        """Record an always-allow policy for (project, signature) (idempotent)."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO approval_policy (project, signature) "
                "VALUES (?, ?)",
                (project, signature),
            )
            await db.commit()

    async def has_policy(self, project: str, signature: str) -> bool:
        """Return True if an always-allow policy exists for (project, signature)."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT 1 FROM approval_policy WHERE project = ? AND signature = ?",
                (project, signature),
            )
            row = await cur.fetchone()
        return row is not None

    async def list_policies(self) -> list[tuple[str, str]]:
        """Return every always-allow policy as ``(project, signature)``, sorted."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT project, signature FROM approval_policy "
                "ORDER BY project, signature"
            )
            rows = await cur.fetchall()
        return [(project, signature) for project, signature in rows]

    async def clear_policy(
        self, project: str | None = None, signature: str | None = None
    ) -> None:
        """Revoke always-allow policies.

        ``project=None`` clears ALL policies; ``project`` alone clears every
        policy for that project; ``project`` + ``signature`` clears one exact
        policy. This is the revocation half of the /policies command.
        """
        async with aiosqlite.connect(self.db_path) as db:
            if project is None:
                await db.execute("DELETE FROM approval_policy")
            elif signature is None:
                await db.execute(
                    "DELETE FROM approval_policy WHERE project = ?", (project,)
                )
            else:
                await db.execute(
                    "DELETE FROM approval_policy WHERE project = ? AND signature = ?",
                    (project, signature),
                )
            await db.commit()


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
