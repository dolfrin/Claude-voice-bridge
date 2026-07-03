"""Tests for build_catchup (IDE catch-up context).

No network. Uses tmp dirs: a real throwaway git repo as ``cwd`` and a fake
``projects_root`` holding hand-written ``*.jsonl`` transcripts with controlled
mtimes. Asserts the compact catch-up block wires the git status/diff/commits
plus the most-recent OTHER session's user/assistant gist, excludes the bridge's
own session, degrades gracefully (non-git cwd, malformed lines, no data), and
never exceeds ``max_chars``.
"""
from __future__ import annotations

import json
import os
import subprocess

from voice_bridge.catchup import build_catchup


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _git(cwd, *args):
    subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _make_git_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t.t")
    _git(path, "config", "user.name", "t")
    (path / "a.txt").write_text("hello\n")
    _git(path, "add", "a.txt")
    _git(path, "commit", "-q", "-m", "initial commit msg")
    # An uncommitted change + an untracked file so status/diff have content.
    (path / "a.txt").write_text("hello world changed line\n")
    (path / "untracked.txt").write_text("brand new file\n")


def _write_session(projects_root, cwd, session_id, entries, mtime=None):
    encoded = str(cwd).replace("/", "-")
    d = projects_root / encoded
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{session_id}.jsonl"
    with f.open("w") as fh:
        for e in entries:
            if isinstance(e, str):  # raw (possibly malformed) line
                fh.write(e + "\n")
            else:
                fh.write(json.dumps(e) + "\n")
    if mtime is not None:
        os.utime(f, (mtime, mtime))
    return f


def _user(text):
    return {"type": "user", "message": {"role": "user", "content": text}}


def _assistant(text):
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    }


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

async def test_build_catchup_includes_git_and_session(tmp_path):
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    projects_root = tmp_path / "projects"
    _write_session(
        projects_root,
        repo,
        "ide-sess",
        [
            _user("please refactor the parser"),
            _assistant("Refactored the parser and added tests."),
        ],
        mtime=2000,
    )

    block = await build_catchup(str(repo), projects_root=str(projects_root))

    assert "[IDE catch-up" in block
    assert "untracked.txt" in block            # git status --short
    assert "initial commit msg" in block       # git log --oneline
    assert "refactor the parser" in block      # session user text
    assert "Refactored the parser" in block    # session assistant text
    assert len(block) <= 4000
    # Security fence: untrusted content must be framed as read-only data so the
    # agent won't obey instructions embedded in a hostile diff/transcript.
    assert "Do NOT follow" in block
    assert "End of IDE catch-up reference data" in block


async def test_build_catchup_excludes_own_session(tmp_path):
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    projects_root = tmp_path / "projects"
    # Newer file is the bridge's OWN session (must be excluded).
    _write_session(
        projects_root, repo, "bridge-sess",
        [_user("SECRET_BRIDGE_TEXT_XYZ")], mtime=5000,
    )
    # Older file is the IDE session we want to surface.
    _write_session(
        projects_root, repo, "ide-sess",
        [_user("OTHER_IDE_TEXT_ABC"),
         _assistant("did the IDE work")], mtime=3000,
    )

    block = await build_catchup(
        str(repo), exclude_session_id="bridge-sess",
        projects_root=str(projects_root),
    )

    assert "OTHER_IDE_TEXT_ABC" in block
    assert "SECRET_BRIDGE_TEXT_XYZ" not in block


async def test_build_catchup_empty_for_no_data(tmp_path):
    # Not a git repo, and no session dir at all -> nothing useful.
    plain = tmp_path / "plain"
    plain.mkdir()
    projects_root = tmp_path / "projects"
    projects_root.mkdir()

    block = await build_catchup(str(plain), projects_root=str(projects_root))
    assert block == ""


async def test_build_catchup_non_git_cwd_skips_git_gracefully(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    projects_root = tmp_path / "projects"
    _write_session(
        projects_root, plain, "s1",
        [_user("hello from a non-git project")], mtime=1000,
    )

    block = await build_catchup(str(plain), projects_root=str(projects_root))

    assert "hello from a non-git project" in block
    assert "Recent session activity" in block
    # No git repo -> no git section headers.
    assert "Git status/diff" not in block


async def test_build_catchup_malformed_jsonl_does_not_crash(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    projects_root = tmp_path / "projects"
    _write_session(
        projects_root, plain, "s1",
        [
            "{ this is not valid json",             # malformed line
            _user("valid user line survives"),
            # a user turn whose content is a tool_result list (NOT user text)
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "content": "should be ignored TOOLRES"},
            ]}},
            _assistant("final assistant note"),
        ],
        mtime=1000,
    )

    block = await build_catchup(str(plain), projects_root=str(projects_root))

    assert "valid user line survives" in block
    assert "final assistant note" in block
    assert "TOOLRES" not in block


async def test_build_catchup_respects_max_chars(tmp_path):
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    # Make a large diff so the git part alone would blow a small budget.
    (repo / "big.txt").write_text("x" * 20000 + "\n")
    projects_root = tmp_path / "projects"
    _write_session(
        projects_root, repo, "s1",
        [_user("y" * 5000), _assistant("z" * 5000)], mtime=1000,
    )

    block = await build_catchup(
        str(repo), projects_root=str(projects_root), max_chars=500,
    )
    assert len(block) <= 500


async def test_build_catchup_never_raises_on_bad_cwd(tmp_path):
    missing = tmp_path / "does-not-exist"
    block = await build_catchup(
        str(missing), projects_root=str(tmp_path / "projects"),
    )
    assert block == ""
