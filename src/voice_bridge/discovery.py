"""Best-effort local project discovery from editor and Claude history."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from .config import ProjectConfig


def discover_projects(
    limit: int,
    home: Path | None = None,
    explicit_cwds: set[str] | None = None,
) -> list[ProjectConfig]:
    """Return recently seen local projects, disabled by default.

    Discovery is intentionally conservative: only existing directories under
    ``~/Projects`` are returned, and already configured ``cwd`` values are
    skipped so ``projects.yaml`` remains authoritative.
    """
    if limit <= 0:
        return []

    home = Path.home() if home is None else home
    projects_root = (home / "Projects").resolve()
    explicit = explicit_cwds or set()
    seen_cwds = set(explicit)
    seen_names: set[str] = set()
    found: list[ProjectConfig] = []

    for path in _candidate_paths(home, projects_root):
        if len(found) >= limit:
            break
        try:
            resolved = path.resolve()
        except OSError:
            continue
        cwd = str(resolved)
        if cwd in seen_cwds:
            continue
        if not _is_local_project(resolved, projects_root):
            continue

        seen_cwds.add(cwd)
        name = _unique_name(_project_name(resolved), seen_names)
        found.append(ProjectConfig(name=name, cwd=cwd, enabled=False))

    return found


def merge_projects(
    explicit: list[ProjectConfig], discovered: list[ProjectConfig]
) -> list[ProjectConfig]:
    """Append discovered projects without duplicating configured names/cwds."""
    merged = list(explicit)
    seen_names = {project.name for project in explicit}
    seen_cwds = {_safe_resolve(project.cwd) for project in explicit}

    for project in discovered:
        cwd = _safe_resolve(project.cwd)
        if project.name in seen_names or cwd in seen_cwds:
            continue
        merged.append(project)
        seen_names.add(project.name)
        seen_cwds.add(cwd)

    return merged


def _candidate_paths(home: Path, projects_root: Path):
    yield from _vscode_paths(home)
    yield from _claude_history_paths(home, projects_root)


def _vscode_paths(home: Path):
    storage = home / ".config" / "Code" / "User" / "workspaceStorage"
    if not storage.exists():
        return

    workspace_files = sorted(
        storage.glob("*/workspace.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for workspace_file in workspace_files:
        try:
            data = json.loads(workspace_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        folder = data.get("folder")
        if not isinstance(folder, str) or not folder.startswith("file://"):
            continue

        parsed = urlparse(folder)
        if parsed.netloc not in {"", "localhost"}:
            continue
        yield Path(unquote(parsed.path))


def _claude_history_paths(home: Path, projects_root: Path):
    history = home / ".claude" / "projects"
    if not history.exists() or not projects_root.exists():
        return

    history_dirs = sorted(
        (path for path in history.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    encoded_history = {path.name for path in history_dirs}
    for project_dir in projects_root.iterdir():
        if not project_dir.is_dir():
            continue
        encoded = "-" + str(project_dir.resolve()).lstrip("/").replace("/", "-")
        if encoded in encoded_history:
            yield project_dir


def _is_local_project(path: Path, projects_root: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if path == projects_root:
        return False
    try:
        path.relative_to(projects_root)
    except ValueError:
        return False
    return True


def _project_name(path: Path) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", path.name.lower()).strip("-")
    return slug or "project"


def _unique_name(base: str, seen: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in seen:
        candidate = f"{base}-{suffix}"
        suffix += 1
    seen.add(candidate)
    return candidate


def _safe_resolve(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return path
