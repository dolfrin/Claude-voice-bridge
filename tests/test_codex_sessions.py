from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from voice_bridge.codex_app_server import CodexRequestError
from voice_bridge.codex_sessions import CodexSessionManager
from voice_bridge.config import Config, ProjectConfig


def make_config(tmp_path: Path, *, autonomy: str = "safe") -> Config:
    return Config(
        telegram_bot_token="test",
        telegram_allowed_user_id=1,
        anthropic_api_key="",
        openai_api_key="",
        together_api_key="",
        together_tts_model="model",
        together_tts_language="",
        tts_backend="piper",
        tts_voice="voice",
        piper_voice_path="",
        whisper_model="tiny",
        autonomy_mode=autonomy,
        approval_timeout=1,
        db_path=str(tmp_path / "bridge.db"),
    )


class FakeStore:
    def __init__(self, enabled=None, sessions=None, policies=None):
        self.enabled = dict(enabled or {})
        self.sessions = dict(sessions or {})
        self.policies = set(policies or set())
        self.saved = []

    async def is_enabled(self, project):
        return self.enabled.get(project, True)

    async def set_enabled(self, project, enabled):
        self.enabled[project] = enabled

    async def get_agent_session_id(self, project, backend):
        return self.sessions.get((project, backend))

    async def set_agent_session_id(self, project, backend, session_id):
        self.sessions[(project, backend)] = session_id
        self.saved.append((project, backend, session_id))

    async def has_policy(self, project, signature):
        return (project, signature) in self.policies


class FakeApprovals:
    def __init__(self, result=True):
        self.result = result
        self.calls = []

    async def request(self, project, tool, tool_input, policy_signature=None):
        self.calls.append((project, tool, tool_input, policy_signature))
        return self.result


class FakeClient:
    def __init__(self, *, auto_complete=True, start_error=None):
        self.running = False
        self.start_calls = 0
        self.auto_complete = auto_complete
        self.start_error = start_error
        self.notification_handlers = []
        self.request_handlers = {}
        self.requests = []
        self.closed = False
        self.thread_counter = 0
        self.turn_counter = 0
        self.invalid_resume = set()
        self.active_writer = set()
        self.turn_payloads = []
        self.complete_before_return = False

    def add_notification_handler(self, handler):
        self.notification_handlers.append(handler)

    def add_request_handler(self, method, handler):
        self.request_handlers[method] = handler

    async def start(self):
        if self.start_error:
            raise self.start_error
        self.start_calls += 1
        self.running = True
        return {"userAgent": "fake"}

    async def close(self):
        self.running = False
        self.closed = True

    async def request(self, method, params=None, **kwargs):
        params = dict(params or {})
        self.requests.append((method, params))
        if method == "thread/start":
            self.thread_counter += 1
            thread_id = f"thread-{self.thread_counter}"
            return {"thread": {"id": thread_id}, "model": "gpt-test"}
        if method == "thread/resume":
            thread_id = params["threadId"]
            if thread_id in self.active_writer:
                raise CodexRequestError(-32600, f"thread {thread_id} already has an active writer")
            if thread_id in self.invalid_resume:
                raise CodexRequestError(-1, "missing")
            return {"thread": {"id": thread_id}, "model": "gpt-resumed"}
        if method == "turn/start":
            self.turn_counter += 1
            turn_id = f"turn-{self.turn_counter}"
            payload = (
                self.turn_payloads.pop(0)
                if self.turn_payloads
                else {
                    "id": turn_id,
                    "status": "completed",
                    "items": [
                        {
                            "type": "agentMessage",
                            "id": f"message-{turn_id}",
                            "text": f"answer-{turn_id}",
                            "phase": "final_answer",
                        }
                    ],
                }
            )
            payload["id"] = turn_id
            if self.auto_complete:
                if self.complete_before_return:
                    await self.emit(
                        "turn/completed", {"threadId": params["threadId"], "turn": payload}
                    )
                else:
                    asyncio.get_running_loop().call_soon(
                        lambda: asyncio.create_task(
                            self.emit(
                                "turn/completed",
                                {"threadId": params["threadId"], "turn": payload},
                            )
                        )
                    )
            return {"turn": {"id": turn_id, "status": "inProgress", "items": []}}
        if method == "turn/interrupt":
            return {}
        raise AssertionError(f"unexpected method: {method}")

    async def emit(self, method, params):
        for handler in list(self.notification_handlers):
            await handler(method, params)


async def wait_for(predicate, timeout=1):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), timeout)


def project(tmp_path: Path, name="alpha", **overrides):
    root = tmp_path / name
    root.mkdir()
    return ProjectConfig(name=name, cwd=str(root), **overrides)


def manager(
    tmp_path,
    projects,
    *,
    store=None,
    approvals=None,
    client=None,
    autonomy="safe",
    ask_user=None,
):
    outbound = []

    async def send(item):
        outbound.append(item)

    instance = CodexSessionManager(
        projects,
        make_config(tmp_path, autonomy=autonomy),
        store or FakeStore(),
        send,
        approvals or FakeApprovals(),
        ask_user,
        client=client or FakeClient(),
    )
    return instance, outbound


@pytest.mark.asyncio
async def test_start_creates_and_persists_thread(tmp_path):
    store = FakeStore()
    client = FakeClient()
    item = project(tmp_path)
    sessions, _ = manager(tmp_path, [item], store=store, client=client)
    await sessions.start_all()
    assert sessions.is_running("alpha")
    assert store.saved == [("alpha", "codex", "thread-1")]
    method, params = client.requests[0]
    assert method == "thread/start"
    assert params["cwd"] == item.cwd
    assert params["approvalPolicy"] == "on-request"
    assert params["sandbox"] == "workspace-write"
    assert sessions.last_model("alpha") == "gpt-test"
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_start_resumes_stored_thread(tmp_path):
    store = FakeStore(sessions={("alpha", "codex"): "thread-old"})
    client = FakeClient()
    sessions, _ = manager(tmp_path, [project(tmp_path)], store=store, client=client)
    await sessions.start_all()
    assert client.requests[0][0] == "thread/resume"
    assert client.requests[0][1]["threadId"] == "thread-old"
    assert sessions.last_model("alpha") == "gpt-resumed"
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_invalid_stored_thread_falls_back_to_new_thread(tmp_path):
    store = FakeStore(sessions={("alpha", "codex"): "gone"})
    client = FakeClient()
    client.invalid_resume.add("gone")
    sessions, _ = manager(tmp_path, [project(tmp_path)], store=store, client=client)
    await sessions.start_all()
    assert [method for method, _ in client.requests[:2]] == ["thread/resume", "thread/start"]
    assert store.sessions[("alpha", "codex")] == "thread-1"
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_active_writer_never_replaces_shared_thread(tmp_path):
    store = FakeStore(sessions={("alpha", "codex"): "thread-in-ide"})
    client = FakeClient()
    client.active_writer.add("thread-in-ide")
    sessions, outbound = manager(
        tmp_path, [project(tmp_path)], store=store, client=client
    )

    await sessions.start_all()

    assert [method for method, _ in client.requests] == ["thread/resume"]
    assert store.saved == []
    assert not sessions.is_running("alpha")
    assert "reload the VS Code window" in outbound[-1].text
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_delivery_emits_one_final_answer_and_reuses_thread(tmp_path):
    client = FakeClient()
    sessions, outbound = manager(tmp_path, [project(tmp_path)], client=client)
    await sessions.start_all()
    await sessions.deliver("alpha", "first")
    await sessions.deliver("alpha", "second")
    await wait_for(lambda: len([x for x in outbound if x.text.startswith("answer-")]) == 2)
    answers = [x.text for x in outbound if x.text.startswith("answer-")]
    assert answers == ["answer-turn-1", "answer-turn-2"]
    starts = [params for method, params in client.requests if method == "turn/start"]
    assert [row["threadId"] for row in starts] == ["thread-1", "thread-1"]
    assert starts[0]["input"][0]["text"] == "first"
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_turn_rebuilds_a_transport_that_died(tmp_path):
    client = FakeClient()
    sessions, outbound = manager(tmp_path, [project(tmp_path)], client=client)
    await sessions.start_all()
    assert client.start_calls == 1
    await sessions.deliver("alpha", "first")
    await wait_for(lambda: any(x.text == "answer-turn-1" for x in outbound))

    client.running = False  # the app-server connection died mid-service

    await sessions.deliver("alpha", "second")
    await wait_for(lambda: any(x.text == "answer-turn-2" for x in outbound))
    assert client.start_calls == 2
    assert not [x for x in outbound if x.alert]
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_completion_before_turn_start_response_is_not_lost(tmp_path):
    client = FakeClient()
    client.complete_before_return = True
    sessions, outbound = manager(tmp_path, [project(tmp_path)], client=client)
    await sessions.start_all()
    await sessions.deliver("alpha", "race")
    await wait_for(lambda: any(x.text == "answer-turn-1" for x in outbound))
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_final_answer_phase_wins_over_commentary(tmp_path):
    client = FakeClient()
    client.turn_payloads.append(
        {
            "status": "completed",
            "items": [
                {"type": "agentMessage", "text": "working", "phase": "commentary"},
                {"type": "agentMessage", "text": "done", "phase": "final_answer"},
            ],
        }
    )
    sessions, outbound = manager(tmp_path, [project(tmp_path)], client=client)
    await sessions.start_all()
    await sessions.deliver("alpha", "go")
    await wait_for(lambda: any(x.text == "done" for x in outbound))
    assert not any(x.text == "working" for x in outbound)
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_two_projects_keep_threads_and_outputs_separate(tmp_path):
    client = FakeClient()
    sessions, outbound = manager(
        tmp_path, [project(tmp_path, "alpha"), project(tmp_path, "beta")], client=client
    )
    await sessions.start_all()
    await sessions.deliver("beta", "for beta")
    await wait_for(lambda: any(x.text.startswith("answer-") for x in outbound))
    answer = next(x for x in outbound if x.text.startswith("answer-"))
    assert answer.project == "beta"
    turn = next(params for method, params in client.requests if method == "turn/start")
    assert turn["threadId"] == "thread-2"
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_interrupt_targets_only_active_turn(tmp_path):
    client = FakeClient(auto_complete=False)
    sessions, _ = manager(tmp_path, [project(tmp_path)], client=client)
    await sessions.start_all()
    await sessions.deliver("alpha", "long")
    await wait_for(lambda: sessions._sessions["alpha"].active_turn_id is not None)
    assert await sessions.interrupt("alpha") is True
    interrupt = [params for method, params in client.requests if method == "turn/interrupt"]
    assert interrupt[-1] == {"threadId": "thread-1", "turnId": "turn-1"}
    await client.emit(
        "turn/completed",
        {"threadId": "thread-1", "turn": {"id": "turn-1", "status": "interrupted", "items": []}},
    )
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_safe_approval_auto_accepts_safe_and_prompts_risky(tmp_path):
    approvals = FakeApprovals(result=False)
    client = FakeClient()
    sessions, _ = manager(
        tmp_path, [project(tmp_path)], approvals=approvals, client=client
    )
    await sessions.start_all()
    handler = client.request_handlers["item/commandExecution/requestApproval"]
    safe = await handler({"threadId": "thread-1", "command": "pwd"})
    risky = await handler({"threadId": "thread-1", "command": "rm file.txt"})
    assert safe == {"decision": "accept"}
    assert risky == {"decision": "decline"}
    assert len(approvals.calls) == 1
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_ask_prompts_safe_action_and_full_prompts_nothing(tmp_path):
    ask_approvals = FakeApprovals(result=True)
    ask_client = FakeClient()
    ask_sessions, _ = manager(
        tmp_path,
        [project(tmp_path, autonomy="ask")],
        approvals=ask_approvals,
        client=ask_client,
    )
    await ask_sessions.start_all()
    result = await ask_client.request_handlers["item/commandExecution/requestApproval"](
        {"threadId": "thread-1", "command": "pwd"}
    )
    assert result == {"decision": "accept"}
    assert len(ask_approvals.calls) == 1
    await ask_sessions.stop_all()

    full_approvals = FakeApprovals(result=False)
    full_client = FakeClient()
    full_sessions, _ = manager(
        tmp_path,
        [ProjectConfig(name="full", cwd=str(tmp_path / "alpha"), autonomy="full")],
        approvals=full_approvals,
        client=full_client,
    )
    await full_sessions.start_all()
    result = await full_client.request_handlers["item/commandExecution/requestApproval"](
        {"threadId": "thread-1", "command": "rm file.txt"}
    )
    assert result == {"decision": "accept"}
    assert full_approvals.calls == []
    await full_sessions.stop_all()


@pytest.mark.asyncio
async def test_unknown_thread_approval_fails_closed(tmp_path):
    client = FakeClient()
    sessions, _ = manager(tmp_path, [project(tmp_path)], client=client)
    await sessions.start_all()
    result = await client.request_handlers["item/fileChange/requestApproval"](
        {"threadId": "unknown", "grantRoot": str(tmp_path)}
    )
    assert result == {"decision": "decline"}
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_permission_profile_always_prompts_in_safe_mode(tmp_path):
    approvals = FakeApprovals(result=True)
    client = FakeClient()
    sessions, _ = manager(
        tmp_path, [project(tmp_path)], approvals=approvals, client=client
    )
    await sessions.start_all()
    result = await client.request_handlers["item/permissions/requestApproval"](
        {
            "threadId": "thread-1",
            "cwd": str(tmp_path / "alpha"),
            "reason": "need network",
            "permissions": {"network": {"enabled": True}, "fileSystem": None},
        }
    )
    assert result == {
        "permissions": {"network": {"enabled": True}},
        "scope": "turn",
    }
    assert approvals.calls[0][1] == "Permissions"
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_denied_or_malformed_permissions_grant_nothing(tmp_path):
    approvals = FakeApprovals(result=False)
    client = FakeClient()
    sessions, _ = manager(
        tmp_path, [project(tmp_path)], approvals=approvals, client=client
    )
    await sessions.start_all()
    handler = client.request_handlers["item/permissions/requestApproval"]
    assert await handler(
        {"threadId": "thread-1", "permissions": {"network": {"enabled": True}}}
    ) == {"permissions": {}, "scope": "turn"}
    assert await handler({"threadId": "unknown", "permissions": {}}) == {
        "permissions": {},
        "scope": "turn",
    }
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_request_user_input_uses_existing_telegram_question_path(tmp_path):
    asked = []

    async def ask(project_name, question, choices):
        asked.append((project_name, question, choices))
        return "Option A"

    client = FakeClient()
    sessions, _ = manager(
        tmp_path, [project(tmp_path)], client=client, ask_user=ask
    )
    await sessions.start_all()
    result = await client.request_handlers["item/tool/requestUserInput"](
        {
            "threadId": "thread-1",
            "questions": [
                {
                    "id": "choice",
                    "question": "Which one?",
                    "isSecret": False,
                    "options": [
                        {"label": "Option A", "description": "first"},
                        {"label": "Option B", "description": "second"},
                    ],
                },
                {
                    "id": "password",
                    "question": "Secret?",
                    "isSecret": True,
                    "options": None,
                },
            ],
        }
    )
    assert asked == [("alpha", "Which one?", ["Option A", "Option B"])]
    assert result == {"answers": {"choice": {"answers": ["Option A"]}}}
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_verbose_item_activity_is_text_only(tmp_path):
    client = FakeClient()
    sessions, outbound = manager(
        tmp_path, [project(tmp_path, verbose=True)], client=client
    )
    await sessions.start_all()
    await client.emit(
        "item/completed",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "item": {"type": "commandExecution", "command": "pytest -q"},
        },
    )
    assert outbound[-1].text == "🔧 Bash: pytest -q"
    assert outbound[-1].spoken == " "
    assert outbound[-1].transient is True
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_failed_turn_surfaces_error(tmp_path):
    client = FakeClient()
    client.turn_payloads.append(
        {"status": "failed", "items": [], "error": {"message": "model unavailable"}}
    )
    sessions, outbound = manager(tmp_path, [project(tmp_path)], client=client)
    await sessions.start_all()
    await sessions.deliver("alpha", "go")
    await wait_for(lambda: any(x.text == "model unavailable" for x in outbound))
    error = next(x for x in outbound if x.text == "model unavailable")
    assert error.alert is True
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_mode_and_effort_apply_to_next_turn(tmp_path):
    client = FakeClient()
    sessions, _ = manager(tmp_path, [project(tmp_path)], client=client)
    await sessions.start_all()
    await sessions.set_mode("alpha", "full")
    await sessions.set_effort("alpha", "high")
    await sessions.deliver("alpha", "go")
    await wait_for(lambda: any(method == "turn/start" for method, _ in client.requests))
    params = next(params for method, params in client.requests if method == "turn/start")
    assert params["approvalPolicy"] == "never"
    assert params["effort"] == "high"
    assert params["sandboxPolicy"] == {"type": "dangerFullAccess"}
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_disabled_project_does_not_start_client(tmp_path):
    store = FakeStore(enabled={"alpha": False})
    client = FakeClient()
    sessions, _ = manager(
        tmp_path, [project(tmp_path)], store=store, client=client
    )
    await sessions.start_all()
    assert not client.running
    assert client.requests == []
    await sessions.stop_all()


@pytest.mark.asyncio
async def test_app_server_start_failure_keeps_manager_alive(tmp_path):
    client = FakeClient(start_error=RuntimeError("not logged in"))
    sessions, outbound = manager(tmp_path, [project(tmp_path)], client=client)
    await sessions.start_all()
    assert not sessions.is_running("alpha")
    assert len(outbound) == 1
    assert "not logged in" in outbound[0].text
    await sessions.stop_all()
