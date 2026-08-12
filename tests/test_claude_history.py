"""Reading Claude Code's own session history off disk."""

import json

from voice_bridge import claude_history as ch


def write_session(directory, uuid, records):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def user(text, **extra):
    return {"type": "user", "message": {"content": text}, **extra}


def assistant(text):
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


# --------------------------------------------------------------------------
# identifying a session
# --------------------------------------------------------------------------
def test_summary_skips_claude_s_own_machinery(tmp_path):
    # Slash commands and hook output are stored as "user" entries; showing one
    # in the picker would identify nothing.
    path = write_session(tmp_path / "proj", "u1", [
        {"cwd": "/w"},
        user("<command-name>/model</command-name>"),
        user("<system-reminder>be nice</system-reminder>"),
        user("fix the topic routing bug"),
    ])

    assert ch.summary(path) == "fix the topic routing bug"


def test_summary_clips_long_first_messages(tmp_path):
    path = write_session(tmp_path / "proj", "u1", [user("x" * 100)])

    result = ch.summary(path)

    assert len(result) == ch.SUMMARY_LEN + 1
    assert result.endswith("…")


def test_title_prefers_the_users_rename_then_the_ai_title(tmp_path):
    path = write_session(tmp_path / "proj", "u1", [
        user("first message"),
        {"type": "ai-title", "aiTitle": "Auto name"},
        {"type": "custom-title", "customTitle": "My name"},
    ])

    assert ch.title(path) == "My name"


def test_the_last_rename_wins(tmp_path):
    # Renaming is repeatable; an earlier title must not stick.
    path = write_session(tmp_path / "proj", "u1", [
        {"type": "custom-title", "customTitle": "Old"},
        {"type": "custom-title", "customTitle": "New"},
    ])

    assert ch.title(path) == "New"


def test_title_falls_back_to_the_first_message(tmp_path):
    path = write_session(tmp_path / "proj", "u1", [user("only this")])

    assert ch.title(path) == "only this"


def test_last_prompt_is_where_we_left_off(tmp_path):
    path = write_session(tmp_path / "proj", "u1", [
        {"type": "last-prompt", "lastPrompt": "early"},
        {"type": "last-prompt", "lastPrompt": "add   a  test"},
    ])

    assert ch.last_prompt(path) == "add a test"


def test_a_broken_line_does_not_kill_the_read(tmp_path):
    directory = tmp_path / "proj"
    directory.mkdir()
    path = directory / "u1.jsonl"
    path.write_text('{"type": "user", "message": {"content": "good"}}\nnot json\n')

    assert ch.summary(path) == "good"


# --------------------------------------------------------------------------
# transcript tail
# --------------------------------------------------------------------------
def test_turns_returns_the_tail_oldest_first(tmp_path):
    path = write_session(tmp_path / "proj", "u1", [
        user("one"), assistant("a1"), user("two"), assistant("a2"),
    ])

    assert ch.turns(path, limit=3) == [
        ("assistant", "a1"), ("user", "two"), ("assistant", "a2"),
    ]


def test_turns_skips_meta_users(tmp_path):
    path = write_session(tmp_path / "proj", "u1", [
        user("<command-name>/clear</command-name>"),
        user("real one"),
    ])

    assert ch.turns(path) == [("user", "real one")]


# --------------------------------------------------------------------------
# listing
# --------------------------------------------------------------------------
def test_recent_is_newest_first_and_limited(tmp_path):
    directory = tmp_path / "proj"
    for i, uuid in enumerate(["a", "b", "c"]):
        path = write_session(directory, uuid, [{"cwd": "/w"}, user(f"m{i}")])
        import os
        os.utime(path, (1000 + i, 1000 + i))

    sessions = ch.recent(directory, limit=2)

    assert [s.uuid for s in sessions] == ["c", "b"]


def test_a_session_without_a_cwd_is_skipped(tmp_path):
    # Without the cwd it records, resume cannot be run from the right place.
    directory = tmp_path / "proj"
    write_session(directory, "a", [user("no cwd here")])

    assert ch.recent(directory) == []


def test_projects_lists_only_directories_holding_sessions(tmp_path):
    write_session(tmp_path / "one", "a", [{"cwd": "/w/one"}, user("hi")])
    (tmp_path / "empty").mkdir()

    found = ch.projects(tmp_path)

    assert [p.cwd for p in found] == ["/w/one"]
    assert found[0].label == "one"


def test_the_history_directory_is_found_by_cwd_not_by_name(tmp_path):
    # "-w-my-app" is not reversible into a path — a dash may be a separator or
    # part of a name — so the cwd inside the file is what we match on.
    write_session(tmp_path / "-w-my-app", "a", [{"cwd": "/w/my-app"}, user("hi")])

    found = ch.project_dir_for_cwd(tmp_path, "/w/my-app")

    assert found == tmp_path / "-w-my-app"
    assert ch.project_dir_for_cwd(tmp_path, "/w/other") is None


def test_session_file_locates_one_session(tmp_path):
    write_session(tmp_path / "d", "abc", [{"cwd": "/w"}, user("hi")])

    assert ch.session_file(tmp_path, "/w", "abc").name == "abc.jsonl"
    assert ch.session_file(tmp_path, "/w", "nope") is None


# --------------------------------------------------------------------------
# live sessions
# --------------------------------------------------------------------------
def test_live_sessions_ignores_dead_processes(tmp_path):
    # ~/.claude/sessions is full of leftovers from processes long gone; a stale
    # entry would make us fork a session nothing is holding.
    (tmp_path / "1.json").write_text(json.dumps(
        {"sessionId": "alive", "pid": 11, "entrypoint": "cli"}))
    (tmp_path / "2.json").write_text(json.dumps(
        {"sessionId": "dead", "pid": 22, "entrypoint": "cli"}))
    (tmp_path / "3.json").write_text(json.dumps(
        {"sessionId": "vsc", "pid": 33, "entrypoint": "vscode-extension"}))

    live = ch.live_sessions(tmp_path, is_alive=lambda pid: pid in {11, 33})

    assert live == {"alive": "CLI", "vsc": "VSCode"}


def test_live_sessions_survives_a_corrupt_file(tmp_path):
    (tmp_path / "bad.json").write_text("{{{")
    (tmp_path / "ok.json").write_text(json.dumps({"sessionId": "s", "pid": 1}))

    assert ch.live_sessions(tmp_path, is_alive=lambda pid: True) == {"s": "CLI"}


def test_no_sessions_directory_is_not_an_error(tmp_path):
    assert ch.live_sessions(tmp_path / "nope") == {}
