import json
import os

from voice_bridge.config import ProjectConfig
from voice_bridge.discovery import discover_projects, merge_projects


def _workspace(home, name, folder, mtime):
    path = home / ".config" / "Code" / "User" / "workspaceStorage" / name
    path.mkdir(parents=True)
    workspace = path / "workspace.json"
    workspace.write_text(json.dumps({"folder": folder}), encoding="utf-8")
    os.utime(workspace, (mtime, mtime))


def test_discover_projects_reads_recent_local_vscode_workspaces(tmp_path):
    home = tmp_path / "home"
    root = home / "Projects"
    aurora = root / "AuroraRecordsAI"
    anti = root / "Anti Imsi"
    aurora.mkdir(parents=True)
    anti.mkdir()

    _workspace(home, "old", aurora.as_uri(), 100)
    _workspace(home, "new", anti.as_uri(), 200)
    _workspace(home, "remote", "vscode-remote://ssh-remote+box/home/app", 300)

    projects = discover_projects(limit=10, home=home)

    assert [(p.name, p.cwd, p.enabled) for p in projects] == [
        ("anti-imsi", str(anti.resolve()), False),
        ("aurorarecordsai", str(aurora.resolve()), False),
    ]


def test_discover_projects_skips_explicit_cwds_and_honors_limit(tmp_path):
    home = tmp_path / "home"
    root = home / "Projects"
    qwing = root / "WhisperX"
    mach = root / "MachRadar"
    qwing.mkdir(parents=True)
    mach.mkdir()

    _workspace(home, "qwing", qwing.as_uri(), 200)
    _workspace(home, "mach", mach.as_uri(), 100)

    projects = discover_projects(
        limit=1,
        home=home,
        explicit_cwds={str(qwing.resolve())},
    )

    assert [(p.name, p.cwd) for p in projects] == [
        ("machradar", str(mach.resolve())),
    ]


def test_discover_projects_reads_claude_history_for_existing_project_dirs(tmp_path):
    home = tmp_path / "home"
    project = home / "Projects" / "DexscreenerUp"
    project.mkdir(parents=True)
    encoded = "-" + str(project.resolve()).lstrip("/").replace("/", "-")
    (home / ".claude" / "projects" / encoded).mkdir(parents=True)

    projects = discover_projects(limit=10, home=home)

    assert [(p.name, p.cwd, p.enabled) for p in projects] == [
        ("dexscreenerup", str(project.resolve()), False),
    ]


def test_merge_projects_keeps_explicit_config_authoritative(tmp_path):
    explicit = [
        ProjectConfig(name="qwing", cwd=str(tmp_path / "WhisperX"), enabled=True),
    ]
    discovered = [
        ProjectConfig(name="qwing", cwd=str(tmp_path / "Other"), enabled=False),
        ProjectConfig(name="bridge", cwd=str(tmp_path / "WhisperX"), enabled=False),
        ProjectConfig(name="machradar", cwd=str(tmp_path / "MachRadar"), enabled=False),
    ]

    merged = merge_projects(explicit, discovered)

    assert [(p.name, p.enabled) for p in merged] == [
        ("qwing", True),
        ("machradar", False),
    ]
