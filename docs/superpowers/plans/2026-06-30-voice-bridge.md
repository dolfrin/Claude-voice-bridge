# Claude Voice Bridge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an always-on Telegram bridge that drives multiple long-lived Claude Agent SDK sessions hands-free by voice/text, one per project.

**Architecture:** A single asyncio Python service. Telegram I/O (voice+text, slash commands, inline-button `/panel`) ↔ a SessionManager holding one `ClaudeSDKClient` per project in streaming-input mode. Inbound voice → Whisper STT → routed (reply-to or last-active) into the right session. Outbound assistant text → split into full text + code-free spoken line → TTS (OpenAI/Piper) → Telegram. SQLite persists message→project map, per-project enabled/session_id, last-active. Autonomy is selectable (full/safe/ask) via a `canUseTool` voice-approval loop.

**Tech Stack:** Python 3.11+, claude-agent-sdk, python-telegram-bot>=21, faster-whisper, openai, piper-tts, pyyaml, aiosqlite, pytest, pytest-asyncio.

**Spec:** [docs/superpowers/specs/2026-06-30-voice-bridge-design.md](../specs/2026-06-30-voice-bridge-design.md)

## Global Constraints

- Python 3.10 (server has 3.10.12; PEP 604 `X | None` unions and `match` are available). `requires-python = ">=3.10"`. asyncio throughout. Single event loop. (See correction C11.)
- Telegram voice format is OGG/Opus; TTS must emit OGG/Opus bytes; STT must accept OGG/Opus bytes.
- Only TELEGRAM_ALLOWED_USER_ID may interact; every inbound update is whitelisted first.
- Voice channel NEVER contains code: strip fenced blocks, inline code, hex colors, dimensions/units (10px,2rem), file paths, URLs, snake_case/camelCase/CONSTANT identifiers.
- All blocking work (whisper, piper, openai, sdk) runs without blocking the loop (async or run_in_executor).
- TDD: write failing test first, watch it fail, minimal impl, watch it pass, commit. No placeholders, complete code.
- Commit messages end with: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

## File structure

```
src/voice_bridge/
  config.py            # Config/ProjectConfig dataclasses, load + validate
  types.py             # Outbound dataclass (shared across sessions/bridge)
  routing.py           # aiosqlite Store: msg->project, enabled, session_id, last_active
  sanitizer.py         # to_spoken / prepare_outbound (code-free voice)
  stt.py               # faster-whisper Transcriber
  tts/__init__.py      # TTSBackend protocol + get_tts factory + available_voices
  tts/openai_tts.py    # OpenAITTS
  tts/piper_tts.py     # PiperTTS
  approvals.py         # is_risky, parse_yes_no, ApprovalManager, make_can_use_tool
  notify_tool.py       # in-process SDK MCP server: notify_user
  sessions.py          # SessionManager (one ClaudeSDKClient per project)
  telegram_io.py       # python-telegram-bot Application + /panel
  bridge.py            # main(): wiring, outbound split+route, inbound resolve
systemd/voice-bridge.service
tests/...
```

---

## Plan corrections (authoritative — apply during implementation)

An adversarial review of the parallel-drafted tasks found cross-task integration
bugs and spec gaps that no single task author could see. The per-task unit tests
are sound; these corrections fix the **seams between tasks** and a few runtime bugs
that no test caught. **Where a correction conflicts with a task section below, the
correction wins.** Apply them as you implement the referenced tasks.

### C1 — Add `src/voice_bridge/types.py` (do in Task 1)
Tasks 8 and 10 import `Outbound` from `voice_bridge.types`, but no task creates it.
Create it as part of Task 1:
```python
# src/voice_bridge/types.py
from dataclasses import dataclass

@dataclass
class Outbound:
    project: str
    text: str    # full content; may contain code/diffs
    spoken: str  # code-free line for TTS
```
Test (add to `tests/test_config.py`):
```python
from voice_bridge.types import Outbound

def test_outbound_fields():
    o = Outbound(project="qwing", text="full", spoken="say")
    assert (o.project, o.text, o.spoken) == ("qwing", "full", "say")
```

### C2 — `Controls` protocol is SYNCHRONOUS with fixed dict keys (Tasks 9 + 10)
`telegram_io` calls `controls.snapshot()` synchronously and reads keys
`project, enabled, mode, voice, engine, last_active`. The bridge's `_Controls`
MUST match exactly — `snapshot` is a plain `def` (NOT `async`) returning dicts keyed
`"project"` (NOT `"name"`). Canonical contract:
```python
from typing import Protocol

class Controls(Protocol):
    def snapshot(self) -> list[dict]:
        # each: {"project": str, "enabled": bool, "mode": str,
        #        "voice": str, "engine": str, "last_active": bool}
        ...
    async def toggle(self, project: str | None, on: bool) -> None: ...
    async def set_mode(self, project: str | None, mode: str) -> None: ...
    async def set_voice(self, project: str | None, voice: str) -> None: ...
    async def set_engine(self, name: str) -> None: ...
```
To keep `snapshot` synchronous, the bridge keeps an in-memory mirror
(`dict[str, dict]`) of each project's enabled/mode/voice, initialised from the store
at startup and updated on every toggle/set_*; `snapshot()` reads the mirror.
Add a Task 10 test that drives the REAL `_Controls`: `snapshot()` returns a list of
dicts with the `"project"` key, and `await toggle(None, False)` disables all.

### C3 — `main()` must seed projects, hold a mutable TTS, and block forever (Task 10)
`TelegramIO.run()` (Task 9) starts polling and returns immediately — it must NOT
block. `main()` owns the run-forever wait. Canonical wiring:
```python
import asyncio, os, signal

async def main() -> None:
    cfg = load_config(os.environ)
    projects = load_projects()
    store = Store(cfg.db_path)
    await store.init()
    await store.seed(projects)                 # C5: seed enabled state

    tts_holder = {"backend": get_tts(cfg)}     # C4: mutable so /engine swaps live
    # ... build transcriber, approvals, controls, sessions, telegram ...

    await sessions.start_all()
    await telegram.run()                        # starts polling, returns

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        await stop.wait()                       # block until shutdown
    finally:
        await sessions.stop_all()
```
Update the Task 10 main test: the stub `telegram.run` returns, the test then calls
`stop.set()`, and asserts `stop_all` ran exactly once **on shutdown** (not immediately).

### C4 — Live engine switch must rebuild the backend (Task 10)
`make_outbound` reads `tts_holder["backend"]` at send time (does NOT capture a fixed
instance). `set_engine`:
```python
async def set_engine(self, name: str) -> None:
    self._cfg.tts_backend = name
    self._tts_holder["backend"] = get_tts(self._cfg)
```
Add a Task 10 test: after `await set_engine("piper")`, an outbound synthesises via the
piper backend (assert the holder's backend type changed).

### C5 — Store: `init()` creates tables, `seed(projects)` seeds enabled state (Tasks 2 + 10)
`Store.init()` signature is `async def init(self) -> None` (tables only). Seeding is a
separate `async def seed(self, projects: list[ProjectConfig]) -> None` using
`INSERT OR IGNORE` so each declared project gets a row with its `enabled` default.
`main()` calls `init()` then `seed(projects)` (C3). After seeding, `is_enabled`
returns the stored value for every declared project. Align Task 8's `FakeStore` so
`is_enabled` returns the seeded value (not an unconditional `True`).

### C6 — `notify_user` carries the real project name (Tasks 7 + 8)
The notify MCP server is created PER session inside `SessionManager`, bound to that
session's project name. Its `on_notify(summary, detail)` closure emits
`Outbound(project=<this project>, text=detail or summary, spoken=summary)` — never a
literal `"bridge"`. `make_notify_server(on_notify)` stays generic; the per-project
closure supplies the name.

### C7 — Inbound edge cases (spec §12, Task 10)
Extract the routing decision into a pure helper and unit-test it:
```python
async def resolve_target(msg: dict, store: Store) -> tuple[str | None, str]:
    # returns (project_or_None, reason) where reason in {"ok","approval","off","none"}
    ...
```
`make_inbound` then handles each reason:
- **approval** (reply-to has a pending approval) → resolve via `parse_yes_no`.
- **empty/failed STT** → reply (text+voice) "Nesupratau, pakartok." and stop.
- **none** (no reply-to, no last-active) → reply listing projects:
  "Į kurį projektą? <names>" and stop.
- **off** (`not await store.is_enabled(project)`) → reply
  "<project> išjungtas (/on <project>)" and stop — do NOT deliver.
- **ok** → `await sessions.deliver(project, text)`.

### C8 — Session crash reporting + `/resume` (spec §12, Task 8 + 9)
Wrap each session's receive loop in try/except; on exception emit
`Outbound(project, text=f"Sesija krito: {err}", spoken="Sesija krito, žiūrėk tekstą.")`
and mark the session stopped. Add `/resume <project>` (and a panel button) →
`await sessions.set_enabled(project, True)`, which resumes from the stored
`session_id`. (Long-silence heartbeat is optional, lower priority.)

### C9 — Task 6 tests stub `claude_agent_sdk`
For consistency with Tasks 7/8, `tests/test_approvals.py` must stub `claude_agent_sdk`
(fake `PermissionResultAllow`/`PermissionResultDeny`) so the suite runs without the
real SDK installed.

### C10 — Verify real SDK `session_id` shape (Task 8)
Task 8 captures `SystemMessage(subtype="init").data["session_id"]`. Before relying on
resume, confirm against the installed `claude-agent-sdk` (per the SDK docs the id also
appears on `ResultMessage.session_id`). If the field differs, adjust the capture and
leave a comment linking the SDK `sessions` doc. (Superseded by C12 — the field is verified.)

### C11 — Environment & test foundation (controller has set this up)
- **Target Python 3.10**, not 3.11. Set `requires-python = ">=3.10"` in pyproject (overrides Task 1's `>=3.11`). Everything used (PEP 604 unions, `match`) exists in 3.10.
- A virtualenv exists at `.venv/` (git-ignored) with **pytest, pytest-asyncio, pyyaml, aiosqlite, claude-agent-sdk, python-telegram-bot, openai** installed. Activate it to run tests: `. .venv/bin/activate && python -m pytest`.
- The two HEAVY deps **faster-whisper** and **piper-tts** are NOT installed. Therefore:
  - `stt.py` and `tts/piper_tts.py` MUST lazy-import them **inside** the constructor/method (not at module top level), so importing the module never requires the package.
  - Task 1 creates `tests/conftest.py` that registers lightweight `sys.modules` stubs so test collection never fails:
    ```python
    # tests/conftest.py
    import sys, types
    def _ensure(name):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
        return sys.modules[name]
    fw = _ensure("faster_whisper")
    if not hasattr(fw, "WhisperModel"):
        fw.WhisperModel = object  # tests patch voice_bridge.stt.WhisperModel
    _ensure("piper")  # piper_tts lazy-imports 'piper'; tests patch the subprocess/synth call
    ```
- pytest config (`[tool.pytest.ini_options] asyncio_mode = "auto"`) is in pyproject from Task 1 — keep it.

### C12 — Verified `claude-agent-sdk` API (v0.2.110) — use these EXACT shapes (Tasks 6, 7, 8)
Introspected from the installed package; this OVERRIDES any guessed SDK usage in the task bodies.

Imports:
```python
from claude_agent_sdk import (
    ClaudeSDKClient, ClaudeAgentOptions,
    AssistantMessage, TextBlock, ResultMessage, SystemMessage,
    PermissionResultAllow, PermissionResultDeny,
    create_sdk_mcp_server, tool,
)
```

**session_id (resolves C10):** read it directly from `ResultMessage.session_id` (also on `AssistantMessage.session_id`). Do NOT parse `SystemMessage.data`.

**permission_mode** valid literals: `'default','acceptEdits','plan','bypassPermissions','dontAsk','auto'`.

**Streaming session loop (Task 8), canonical:**
```python
options = ClaudeAgentOptions(
    cwd=project.cwd,
    model=project.model,                       # or None
    system_prompt=appended_instructions,        # str
    permission_mode="bypassPermissions" if mode == "full" else "default",
    can_use_tool=None if mode == "full" else make_can_use_tool(project, cfg, approvals),
    mcp_servers={"bridge": notify_server},      # create_sdk_mcp_server("bridge", tools=[...])
    allowed_tools=["mcp__bridge__notify_user"],
    resume=saved_session_id,                     # str | None
)
client = ClaudeSDKClient(options)
await client.connect()
try:
    while True:
        text = await queue.get()                # next user turn (None => shutdown sentinel)
        if text is None:
            break
        await client.query(text)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                parts = [b.text for b in msg.content if isinstance(b, TextBlock)]
                if parts:
                    await on_outbound(Outbound(project.name, "\n".join(parts), ""))  # spoken filled in make_outbound
            elif isinstance(msg, ResultMessage):
                if msg.session_id:
                    await store.set_session_id(project.name, msg.session_id)
finally:
    await client.disconnect()
```
`ClaudeSDKClient` methods: `connect, disconnect, query, receive_response, receive_messages, interrupt, set_model, set_permission_mode`. Tests mock `ClaudeSDKClient` with a fake exposing `connect/query/receive_response/disconnect` (an async-generator `receive_response`).

**can_use_tool (Task 6)** — exact signature `ClaudeAgentOptions.can_use_tool: Callable[[str, dict, ToolPermissionContext], Awaitable[PermissionResultAllow | PermissionResultDeny]]`:
```python
def make_can_use_tool(project, cfg, approvals):
    mode = effective_autonomy(project, cfg)         # 'full'|'safe'|'ask'
    async def can_use_tool(tool_name, tool_input, context):
        if mode == "full":
            return PermissionResultAllow()
        if mode == "safe" and not is_risky(tool_name, tool_input, project.cwd):
            return PermissionResultAllow()
        approved = await approvals.request(project.name, tool_name, tool_input)
        return PermissionResultAllow() if approved else PermissionResultDeny(message="User denied or timed out")
    return can_use_tool
```
`PermissionResultAllow()` (no args = allow). `PermissionResultDeny(message=...)`. Task 6 tests stub these two classes via `sys.modules` OR import them for real (the SDK IS installed, so real import works — prefer real import, drop the C9 stub requirement since the SDK is present).

**MCP notify tool (Task 7):**
```python
from claude_agent_sdk import tool, create_sdk_mcp_server
NOTIFY_TOOL_NAME = "mcp__bridge__notify_user"
def make_notify_server(on_notify):
    @tool("notify_user",
          "Send a short status/question to the user. 'summary' is spoken aloud (no code); 'detail' is text-only.",
          {"summary": str, "detail": str})
    async def notify_user(args):
        await on_notify(args.get("summary", ""), args.get("detail", ""))
        return {"content": [{"type": "text", "text": "delivered"}]}
    return create_sdk_mcp_server("bridge", tools=[notify_user])
```
`create_sdk_mcp_server(name, version="1.0.0", tools=[...])` returns the value to put in `mcp_servers`. Tests call `notify_user(...)`'s handler directly (it's wrapped; access via the returned server's tool list or keep a reference) and assert `on_notify` fired with `(summary, detail)`.

---

### Task 1: Project scaffold + config

**Files:**
- Create: `/home/home/Projects/claude-voice-bridge/pyproject.toml`
- Create: `/home/home/Projects/claude-voice-bridge/src/voice_bridge/__init__.py`
- Create: `/home/home/Projects/claude-voice-bridge/src/voice_bridge/config.py`
- Create: `/home/home/Projects/claude-voice-bridge/.env.example`
- Create: `/home/home/Projects/claude-voice-bridge/projects.yaml`
- Test: `/home/home/Projects/claude-voice-bridge/tests/test_config.py`

**Interfaces:**

Consumes: nothing (this is the first task; the canonical contract's `config_keys`, `Config` and `ProjectConfig` dataclass shapes are the only inputs).

Produces (exact signatures other tasks import):
- `@dataclass Config: telegram_bot_token:str, telegram_allowed_user_id:int, anthropic_api_key:str, openai_api_key:str, tts_backend:str, tts_voice:str, piper_voice_path:str, whisper_model:str, autonomy_mode:str, approval_timeout:int, db_path:str`
- `@dataclass ProjectConfig: name:str, cwd:str, enabled:bool=True, autonomy:str|None=None, voice:str|None=None, model:str|None=None, system_prompt_extra:str=''`
- `def load_config(env: Mapping[str, str] | None = None) -> Config`
- `def load_projects(path: str = 'projects.yaml') -> list[ProjectConfig]`
- `def effective_autonomy(project: ProjectConfig, cfg: Config) -> str`
- `def effective_voice(project: ProjectConfig, cfg: Config) -> str`

---

TDD steps:

- [ ] **Step 1: Create the package skeleton and pyproject.** Create `/home/home/Projects/claude-voice-bridge/pyproject.toml`:
  ```toml
  [build-system]
  requires = ["setuptools>=68", "wheel"]
  build-backend = "setuptools.build_meta"

  [project]
  name = "voice-bridge"
  version = "0.1.0"
  description = "Telegram voice/text bridge for long-running Claude Agent SDK sessions"
  requires-python = ">=3.11"
  dependencies = [
      "claude-agent-sdk",
      "python-telegram-bot>=21",
      "faster-whisper",
      "openai",
      "piper-tts",
      "pyyaml",
      "aiosqlite",
  ]

  [project.optional-dependencies]
  dev = [
      "pytest",
      "pytest-asyncio",
  ]

  [tool.setuptools.packages.find]
  where = ["src"]

  [tool.pytest.ini_options]
  asyncio_mode = "auto"
  testpaths = ["tests"]
  ```
  Then create the empty package marker `/home/home/Projects/claude-voice-bridge/src/voice_bridge/__init__.py`:
  ```python
  """Claude Voice Bridge package."""

  __version__ = "0.1.0"
  ```

- [ ] **Step 2: Create `.env.example` with every config key.** Create `/home/home/Projects/claude-voice-bridge/.env.example`:
  ```dotenv
  # Telegram
  TELEGRAM_BOT_TOKEN=
  TELEGRAM_ALLOWED_USER_ID=

  # Model providers
  ANTHROPIC_API_KEY=
  OPENAI_API_KEY=

  # Text-to-speech
  TTS_BACKEND=openai            # openai|piper
  TTS_VOICE=nova
  PIPER_VOICE_PATH=/opt/piper/lt_LT-voice.onnx

  # Speech-to-text
  WHISPER_MODEL=large-v3

  # Autonomy / approvals
  AUTONOMY_MODE=safe            # full|safe|ask
  APPROVAL_TIMEOUT=300

  # Persistence
  DB_PATH=/var/lib/voice-bridge/state.db
  ```

- [ ] **Step 3: Create an example `projects.yaml`.** Create `/home/home/Projects/claude-voice-bridge/projects.yaml`:
  ```yaml
  projects:
    - name: qwing
      cwd: /home/home/Projects/WhisperX
      enabled: true
      autonomy: safe
      voice: nova
      model: claude-opus-4-8
      system_prompt_extra: ""
    - name: bridge
      cwd: /home/home/Projects/claude-voice-bridge
      enabled: false
      voice: echo
  ```

- [ ] **Step 4: Write the failing test.** Create `/home/home/Projects/claude-voice-bridge/tests/test_config.py` with COMPLETE code (no network, no secrets — pure env mappings + temp yaml files):
  ```python
  import textwrap

  import pytest

  from voice_bridge.config import (
      Config,
      ProjectConfig,
      effective_autonomy,
      effective_voice,
      load_config,
      load_projects,
  )


  def _full_env() -> dict[str, str]:
      return {
          "TELEGRAM_BOT_TOKEN": "123:abc",
          "TELEGRAM_ALLOWED_USER_ID": "42",
          "ANTHROPIC_API_KEY": "sk-ant-test",
          "OPENAI_API_KEY": "sk-openai-test",
          "TTS_BACKEND": "openai",
          "TTS_VOICE": "nova",
          "PIPER_VOICE_PATH": "/opt/piper/lt.onnx",
          "WHISPER_MODEL": "large-v3",
          "AUTONOMY_MODE": "safe",
          "APPROVAL_TIMEOUT": "300",
          "DB_PATH": "/var/lib/voice-bridge/state.db",
      }


  def test_load_config_parses_all_fields_with_correct_types():
      cfg = load_config(_full_env())
      assert isinstance(cfg, Config)
      assert cfg.telegram_bot_token == "123:abc"
      assert cfg.telegram_allowed_user_id == 42
      assert isinstance(cfg.telegram_allowed_user_id, int)
      assert cfg.anthropic_api_key == "sk-ant-test"
      assert cfg.openai_api_key == "sk-openai-test"
      assert cfg.tts_backend == "openai"
      assert cfg.tts_voice == "nova"
      assert cfg.piper_voice_path == "/opt/piper/lt.onnx"
      assert cfg.whisper_model == "large-v3"
      assert cfg.autonomy_mode == "safe"
      assert cfg.approval_timeout == 300
      assert isinstance(cfg.approval_timeout, int)
      assert cfg.db_path == "/var/lib/voice-bridge/state.db"


  def test_load_config_applies_defaults_for_optional_keys():
      env = {
          "TELEGRAM_BOT_TOKEN": "123:abc",
          "TELEGRAM_ALLOWED_USER_ID": "42",
          "ANTHROPIC_API_KEY": "sk-ant-test",
          "OPENAI_API_KEY": "sk-openai-test",
      }
      cfg = load_config(env)
      assert cfg.tts_backend == "openai"
      assert cfg.tts_voice == "nova"
      assert cfg.piper_voice_path == ""
      assert cfg.whisper_model == "large-v3"
      assert cfg.autonomy_mode == "safe"
      assert cfg.approval_timeout == 300
      assert cfg.db_path == "voice-bridge.db"


  def test_load_config_missing_required_key_raises_clear_error():
      env = _full_env()
      del env["TELEGRAM_BOT_TOKEN"]
      with pytest.raises(ValueError) as exc:
          load_config(env)
      assert "TELEGRAM_BOT_TOKEN" in str(exc.value)


  def test_load_config_non_int_user_id_raises_clear_error():
      env = _full_env()
      env["TELEGRAM_ALLOWED_USER_ID"] = "not-a-number"
      with pytest.raises(ValueError) as exc:
          load_config(env)
      assert "TELEGRAM_ALLOWED_USER_ID" in str(exc.value)


  def test_load_config_non_int_timeout_raises_clear_error():
      env = _full_env()
      env["APPROVAL_TIMEOUT"] = "soon"
      with pytest.raises(ValueError) as exc:
          load_config(env)
      assert "APPROVAL_TIMEOUT" in str(exc.value)


  def test_load_config_invalid_backend_raises_clear_error():
      env = _full_env()
      env["TTS_BACKEND"] = "espeak"
      with pytest.raises(ValueError) as exc:
          load_config(env)
      assert "TTS_BACKEND" in str(exc.value)


  def test_load_config_invalid_autonomy_raises_clear_error():
      env = _full_env()
      env["AUTONOMY_MODE"] = "yolo"
      with pytest.raises(ValueError) as exc:
          load_config(env)
      assert "AUTONOMY_MODE" in str(exc.value)


  def test_load_projects_parses_fields_and_defaults(tmp_path):
      yaml_text = textwrap.dedent(
          """
          projects:
            - name: qwing
              cwd: /home/home/Projects/WhisperX
              enabled: true
              autonomy: safe
              voice: nova
              model: claude-opus-4-8
              system_prompt_extra: "be terse"
            - name: bridge
              cwd: /home/home/Projects/claude-voice-bridge
          """
      )
      path = tmp_path / "projects.yaml"
      path.write_text(yaml_text)

      projects = load_projects(str(path))
      assert [p.name for p in projects] == ["qwing", "bridge"]

      qwing = projects[0]
      assert isinstance(qwing, ProjectConfig)
      assert qwing.cwd == "/home/home/Projects/WhisperX"
      assert qwing.enabled is True
      assert qwing.autonomy == "safe"
      assert qwing.voice == "nova"
      assert qwing.model == "claude-opus-4-8"
      assert qwing.system_prompt_extra == "be terse"

      bridge = projects[1]
      assert bridge.enabled is True
      assert bridge.autonomy is None
      assert bridge.voice is None
      assert bridge.model is None
      assert bridge.system_prompt_extra == ""


  def test_load_projects_missing_name_raises_clear_error(tmp_path):
      path = tmp_path / "projects.yaml"
      path.write_text("projects:\n  - cwd: /tmp/x\n")
      with pytest.raises(ValueError) as exc:
          load_projects(str(path))
      assert "name" in str(exc.value)


  def test_load_projects_missing_cwd_raises_clear_error(tmp_path):
      path = tmp_path / "projects.yaml"
      path.write_text("projects:\n  - name: x\n")
      with pytest.raises(ValueError) as exc:
          load_projects(str(path))
      assert "cwd" in str(exc.value)


  def test_load_projects_missing_file_raises_clear_error(tmp_path):
      missing = tmp_path / "nope.yaml"
      with pytest.raises(FileNotFoundError) as exc:
          load_projects(str(missing))
      assert "nope.yaml" in str(exc.value)


  def test_effective_autonomy_prefers_project_override():
      cfg = load_config(_full_env())  # autonomy_mode == "safe"
      proj = ProjectConfig(name="x", cwd="/tmp/x", autonomy="full")
      assert effective_autonomy(proj, cfg) == "full"


  def test_effective_autonomy_falls_back_to_global():
      cfg = load_config(_full_env())  # autonomy_mode == "safe"
      proj = ProjectConfig(name="x", cwd="/tmp/x", autonomy=None)
      assert effective_autonomy(proj, cfg) == "safe"


  def test_effective_voice_prefers_project_override():
      cfg = load_config(_full_env())  # tts_voice == "nova"
      proj = ProjectConfig(name="x", cwd="/tmp/x", voice="echo")
      assert effective_voice(proj, cfg) == "echo"


  def test_effective_voice_falls_back_to_global():
      cfg = load_config(_full_env())  # tts_voice == "nova"
      proj = ProjectConfig(name="x", cwd="/tmp/x", voice=None)
      assert effective_voice(proj, cfg) == "nova"
  ```

- [ ] **Step 5: Run test to verify it fails.** Run:
  ```bash
  cd /home/home/Projects/claude-voice-bridge && python -m pytest tests/test_config.py -q
  ```
  Expected: collection error / `ModuleNotFoundError: No module named 'voice_bridge.config'` (the module does not exist yet). All tests fail to import.

- [ ] **Step 6: Write minimal implementation.** Create `/home/home/Projects/claude-voice-bridge/src/voice_bridge/config.py` with COMPLETE code:
  ```python
  """Load and validate environment Config and projects.yaml into typed dataclasses."""

  from __future__ import annotations

  import os
  from collections.abc import Mapping
  from dataclasses import dataclass, field

  import yaml

  _VALID_TTS_BACKENDS = {"openai", "piper"}
  _VALID_AUTONOMY_MODES = {"full", "safe", "ask"}


  @dataclass
  class Config:
      telegram_bot_token: str
      telegram_allowed_user_id: int
      anthropic_api_key: str
      openai_api_key: str
      tts_backend: str
      tts_voice: str
      piper_voice_path: str
      whisper_model: str
      autonomy_mode: str
      approval_timeout: int
      db_path: str


  @dataclass
  class ProjectConfig:
      name: str
      cwd: str
      enabled: bool = True
      autonomy: str | None = None
      voice: str | None = None
      model: str | None = None
      system_prompt_extra: str = ""


  def _require(env: Mapping[str, str], key: str) -> str:
      value = env.get(key)
      if value is None or value == "":
          raise ValueError(f"Missing required config key: {key}")
      return value


  def _require_int(env: Mapping[str, str], key: str) -> int:
      raw = _require(env, key)
      try:
          return int(raw)
      except (TypeError, ValueError):
          raise ValueError(f"Config key {key} must be an integer, got: {raw!r}")


  def _optional_int(env: Mapping[str, str], key: str, default: int) -> int:
      raw = env.get(key)
      if raw is None or raw == "":
          return default
      try:
          return int(raw)
      except (TypeError, ValueError):
          raise ValueError(f"Config key {key} must be an integer, got: {raw!r}")


  def load_config(env: Mapping[str, str] | None = None) -> Config:
      """Build a validated Config from a mapping (defaults to os.environ)."""
      env = os.environ if env is None else env

      tts_backend = env.get("TTS_BACKEND") or "openai"
      if tts_backend not in _VALID_TTS_BACKENDS:
          raise ValueError(
              f"Config key TTS_BACKEND must be one of "
              f"{sorted(_VALID_TTS_BACKENDS)}, got: {tts_backend!r}"
          )

      autonomy_mode = env.get("AUTONOMY_MODE") or "safe"
      if autonomy_mode not in _VALID_AUTONOMY_MODES:
          raise ValueError(
              f"Config key AUTONOMY_MODE must be one of "
              f"{sorted(_VALID_AUTONOMY_MODES)}, got: {autonomy_mode!r}"
          )

      return Config(
          telegram_bot_token=_require(env, "TELEGRAM_BOT_TOKEN"),
          telegram_allowed_user_id=_require_int(env, "TELEGRAM_ALLOWED_USER_ID"),
          anthropic_api_key=_require(env, "ANTHROPIC_API_KEY"),
          openai_api_key=_require(env, "OPENAI_API_KEY"),
          tts_backend=tts_backend,
          tts_voice=env.get("TTS_VOICE") or "nova",
          piper_voice_path=env.get("PIPER_VOICE_PATH") or "",
          whisper_model=env.get("WHISPER_MODEL") or "large-v3",
          autonomy_mode=autonomy_mode,
          approval_timeout=_optional_int(env, "APPROVAL_TIMEOUT", 300),
          db_path=env.get("DB_PATH") or "voice-bridge.db",
      )


  def load_projects(path: str = "projects.yaml") -> list[ProjectConfig]:
      """Parse projects.yaml into a list of validated ProjectConfig."""
      if not os.path.exists(path):
          raise FileNotFoundError(f"projects file not found: {path}")

      with open(path, "r", encoding="utf-8") as fh:
          data = yaml.safe_load(fh) or {}

      raw_projects = data.get("projects") or []
      projects: list[ProjectConfig] = []
      for idx, raw in enumerate(raw_projects):
          name = raw.get("name")
          if not name:
              raise ValueError(f"project at index {idx} is missing required field: name")
          cwd = raw.get("cwd")
          if not cwd:
              raise ValueError(f"project {name!r} is missing required field: cwd")

          enabled = raw.get("enabled")
          projects.append(
              ProjectConfig(
                  name=name,
                  cwd=cwd,
                  enabled=True if enabled is None else bool(enabled),
                  autonomy=raw.get("autonomy"),
                  voice=raw.get("voice"),
                  model=raw.get("model"),
                  system_prompt_extra=raw.get("system_prompt_extra") or "",
              )
          )
      return projects


  def effective_autonomy(project: ProjectConfig, cfg: Config) -> str:
      """Project-level autonomy override, falling back to the global mode."""
      return project.autonomy or cfg.autonomy_mode


  def effective_voice(project: ProjectConfig, cfg: Config) -> str:
      """Project-level voice override, falling back to the global voice."""
      return project.voice or cfg.tts_voice
  ```
  Note: `field` is imported only if needed; the dataclasses above use plain defaults so the import is unused — remove the `field` import to keep it clean: the import line becomes `from dataclasses import dataclass`.

- [ ] **Step 7: Run test to verify it passes.** Run:
  ```bash
  cd /home/home/Projects/claude-voice-bridge && python -m pytest tests/test_config.py -q
  ```
  Expected: `16 passed` with no warnings (asyncio_mode set in pyproject; these tests are sync so they run plain). If `yaml` is missing, run `python -m pip install pyyaml` first, then re-run.

- [ ] **Step 8: Commit.** Run:
  ```bash
  cd /home/home/Projects/claude-voice-bridge && git add pyproject.toml .env.example projects.yaml src/voice_bridge/__init__.py src/voice_bridge/config.py tests/test_config.py && git commit -m "feat(config): scaffold package and typed env + projects.yaml loading

  Add pyproject with deps, .env.example covering all config keys, example
  projects.yaml, and config.py with load_config/load_projects/effective_*
  plus clear validation errors for missing/invalid env.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 2: SQLite Store (routing/state)

aiosqlite-backed `Store` implementing the `db_schema` from the contract: `messages(message_id INTEGER PRIMARY KEY, project TEXT NOT NULL)`, `projects(name TEXT PRIMARY KEY, enabled INTEGER NOT NULL DEFAULT 1, session_id TEXT)`, `meta(key TEXT PRIMARY KEY, value TEXT)` where `meta` holds `last_active`. All methods are async; tests use a fresh tmp db per test (no shared state, no network).

**Files:**
- Create: `src/voice_bridge/routing.py`
- Test: `tests/test_routing.py`
- Modify: none (assumes `src/voice_bridge/__init__.py` already exists from Task 1; if not, this task creates it as an empty file)

**Interfaces:**

Consumes (provided by Task 1 — config.py):
- `@dataclass ProjectConfig: name:str, cwd:str, enabled:bool=True, autonomy:str|None=None, voice:str|None=None, model:str|None=None, system_prompt_extra:str=''` — used as input to `seed()`. Only `.name` and `.enabled` are read by this module.

Produces (exposed by `routing.py`):
- `class Store: def __init__(self, db_path:str)`
- `async def init(self) -> None` — create tables, then call `seed()` with no projects (idempotent; callers pass projects via `seed()` directly or `init` seeds nothing). Per contract: "create tables, seed projects from list[ProjectConfig] via seed()". Implemented as `async def init(self, projects: list[ProjectConfig] | None = None) -> None`.
- `async def seed(self, projects: list[ProjectConfig]) -> None` — insert missing projects with their `enabled` default; never overwrite existing rows.
- `async def map_message(self, message_id:int, project:str) -> None`
- `async def project_for_message(self, message_id:int) -> str | None`
- `async def set_last_active(self, project:str) -> None`
- `async def get_last_active(self) -> str | None`
- `async def set_enabled(self, project:str, enabled:bool) -> None`
- `async def is_enabled(self, project:str) -> bool`
- `async def enabled_map(self) -> dict[str,bool]`
- `async def set_session_id(self, project:str, session_id:str) -> None`
- `async def get_session_id(self, project:str) -> str | None`

---

TDD steps:

- [ ] **Step 1: Write the failing test for `init()` + tables.** Create `tests/test_routing.py` with the imports, fixtures, and the first test. Uses `pytest-asyncio` (`asyncio_mode = auto` assumed from pyproject; an explicit marker is added to be safe). The `tmp_db` fixture gives every test its own on-disk db file (aiosqlite needs a real path per connection; `:memory:` would be a fresh empty db on each `connect()`).

  ```python
  # tests/test_routing.py
  import pytest
  import aiosqlite

  from voice_bridge.config import ProjectConfig
  from voice_bridge.routing import Store


  @pytest.fixture
  def tmp_db(tmp_path):
      return str(tmp_path / "state.db")


  def _proj(name, enabled=True):
      return ProjectConfig(name=name, cwd=f"/p/{name}", enabled=enabled)


  @pytest.mark.asyncio
  async def test_init_creates_tables(tmp_db):
      store = Store(tmp_db)
      await store.init()

      async with aiosqlite.connect(tmp_db) as db:
          db.row_factory = aiosqlite.Row
          cur = await db.execute(
              "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
          )
          rows = [r["name"] for r in await cur.fetchall()]

      assert "messages" in rows
      assert "projects" in rows
      assert "meta" in rows
  ```

- [ ] **Step 1: Run test to verify it fails.** Run from repo root:
  ```
  python -m pytest tests/test_routing.py::test_init_creates_tables -x -q
  ```
  Expected failure: `ModuleNotFoundError: No module named 'voice_bridge.routing'` (collection error, since `routing.py` does not exist yet).

- [ ] **Step 1: Write minimal implementation.** Create `src/voice_bridge/routing.py` with the class, the constructor, table DDL, and `init()`. (Imports `ProjectConfig` only for typing/seed; `seed()` is a stub that does nothing yet — the next test drives it.)

  ```python
  # src/voice_bridge/routing.py
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
  """


  class Store:
      def __init__(self, db_path: str) -> None:
          self.db_path = db_path

      async def init(self, projects: list[ProjectConfig] | None = None) -> None:
          async with aiosqlite.connect(self.db_path) as db:
              await db.executescript(_SCHEMA)
              await db.commit()
          if projects:
              await self.seed(projects)

      async def seed(self, projects: list[ProjectConfig]) -> None:
          return None
  ```

- [ ] **Step 1: Run test to verify it passes.**
  ```
  python -m pytest tests/test_routing.py::test_init_creates_tables -x -q
  ```
  Expected: `1 passed`.

- [ ] **Step 1: Commit.**
  ```
  git add src/voice_bridge/routing.py tests/test_routing.py
  git commit -m "feat(routing): Store.init creates messages/projects/meta tables

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

- [ ] **Step 2: Write the failing test for `seed()` enabled defaults + idempotency.** Append to `tests/test_routing.py`. Verifies: seeded projects appear in `enabled_map()` with their `ProjectConfig.enabled` default, and re-seeding does not overwrite an explicitly-changed `enabled` state (insert-missing-only semantics).

  ```python
  @pytest.mark.asyncio
  async def test_seed_uses_enabled_defaults(tmp_db):
      store = Store(tmp_db)
      await store.init()
      await store.seed([_proj("qwing", enabled=True), _proj("othersapp", enabled=False)])

      assert await store.enabled_map() == {"qwing": True, "othersapp": False}


  @pytest.mark.asyncio
  async def test_init_with_projects_seeds(tmp_db):
      store = Store(tmp_db)
      await store.init([_proj("qwing", enabled=True)])

      assert await store.enabled_map() == {"qwing": True}


  @pytest.mark.asyncio
  async def test_seed_is_idempotent_and_preserves_state(tmp_db):
      store = Store(tmp_db)
      await store.init([_proj("qwing", enabled=True)])
      # user disabled it at runtime
      await store.set_enabled("qwing", False)
      # re-seed (e.g. restart) must NOT flip it back to the yaml default
      await store.seed([_proj("qwing", enabled=True)])

      assert await store.is_enabled("qwing") is False
  ```

- [ ] **Step 2: Run test to verify it fails.**
  ```
  python -m pytest tests/test_routing.py -x -q -k "seed or init_with_projects"
  ```
  Expected failure: `AttributeError: 'Store' object has no attribute 'enabled_map'` (raised in `test_seed_uses_enabled_defaults`).

- [ ] **Step 2: Write minimal implementation.** Replace the stub `seed()` and add `enabled_map`, `set_enabled`, `is_enabled`. Edit `src/voice_bridge/routing.py` so the class body becomes:

  ```python
  class Store:
      def __init__(self, db_path: str) -> None:
          self.db_path = db_path

      async def init(self, projects: list[ProjectConfig] | None = None) -> None:
          async with aiosqlite.connect(self.db_path) as db:
              await db.executescript(_SCHEMA)
              await db.commit()
          if projects:
              await self.seed(projects)

      async def seed(self, projects: list[ProjectConfig]) -> None:
          async with aiosqlite.connect(self.db_path) as db:
              for p in projects:
                  await db.execute(
                      "INSERT OR IGNORE INTO projects (name, enabled) VALUES (?, ?)",
                      (p.name, 1 if p.enabled else 0),
                  )
              await db.commit()

      async def set_enabled(self, project: str, enabled: bool) -> None:
          async with aiosqlite.connect(self.db_path) as db:
              await db.execute(
                  "INSERT INTO projects (name, enabled) VALUES (?, ?) "
                  "ON CONFLICT(name) DO UPDATE SET enabled=excluded.enabled",
                  (project, 1 if enabled else 0),
              )
              await db.commit()

      async def is_enabled(self, project: str) -> bool:
          async with aiosqlite.connect(self.db_path) as db:
              cur = await db.execute(
                  "SELECT enabled FROM projects WHERE name = ?", (project,)
              )
              row = await cur.fetchone()
          return bool(row[0]) if row is not None else False

      async def enabled_map(self) -> dict[str, bool]:
          async with aiosqlite.connect(self.db_path) as db:
              cur = await db.execute("SELECT name, enabled FROM projects")
              rows = await cur.fetchall()
          return {name: bool(enabled) for name, enabled in rows}
  ```

- [ ] **Step 2: Run test to verify it passes.**
  ```
  python -m pytest tests/test_routing.py -x -q -k "seed or init_with_projects"
  ```
  Expected: `3 passed` (and `test_init_creates_tables` still passes when the full file runs).

- [ ] **Step 2: Commit.**
  ```
  git add src/voice_bridge/routing.py tests/test_routing.py
  git commit -m "feat(routing): seed projects with enabled defaults; enabled get/set/map

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

- [ ] **Step 3: Write the failing test for message→project map.** Append to `tests/test_routing.py`. Covers store/lookup, miss returns `None`, and re-mapping the same `message_id` (PRIMARY KEY) upserts to the new project.

  ```python
  @pytest.mark.asyncio
  async def test_map_message_round_trip(tmp_db):
      store = Store(tmp_db)
      await store.init()
      await store.map_message(1001, "qwing")

      assert await store.project_for_message(1001) == "qwing"


  @pytest.mark.asyncio
  async def test_project_for_unknown_message_is_none(tmp_db):
      store = Store(tmp_db)
      await store.init()

      assert await store.project_for_message(999) is None


  @pytest.mark.asyncio
  async def test_map_message_upserts_existing_id(tmp_db):
      store = Store(tmp_db)
      await store.init()
      await store.map_message(1001, "qwing")
      await store.map_message(1001, "othersapp")

      assert await store.project_for_message(1001) == "othersapp"
  ```

- [ ] **Step 3: Run test to verify it fails.**
  ```
  python -m pytest tests/test_routing.py -x -q -k "message"
  ```
  Expected failure: `AttributeError: 'Store' object has no attribute 'map_message'`.

- [ ] **Step 3: Write minimal implementation.** Add `map_message` and `project_for_message` methods to the `Store` class in `src/voice_bridge/routing.py` (insert after `enabled_map`):

  ```python
      async def map_message(self, message_id: int, project: str) -> None:
          async with aiosqlite.connect(self.db_path) as db:
              await db.execute(
                  "INSERT INTO messages (message_id, project) VALUES (?, ?) "
                  "ON CONFLICT(message_id) DO UPDATE SET project=excluded.project",
                  (message_id, project),
              )
              await db.commit()

      async def project_for_message(self, message_id: int) -> str | None:
          async with aiosqlite.connect(self.db_path) as db:
              cur = await db.execute(
                  "SELECT project FROM messages WHERE message_id = ?", (message_id,)
              )
              row = await cur.fetchone()
          return row[0] if row is not None else None
  ```

- [ ] **Step 3: Run test to verify it passes.**
  ```
  python -m pytest tests/test_routing.py -x -q -k "message"
  ```
  Expected: `3 passed`.

- [ ] **Step 3: Commit.**
  ```
  git add src/voice_bridge/routing.py tests/test_routing.py
  git commit -m "feat(routing): message_id->project map with upsert and miss=None

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

- [ ] **Step 4: Write the failing test for `last_active` round-trip (meta table).** Append to `tests/test_routing.py`. Covers: unset returns `None`, set/get, and overwrite (latest wins). Also asserts it physically lives in `meta` under key `last_active`.

  ```python
  @pytest.mark.asyncio
  async def test_last_active_unset_is_none(tmp_db):
      store = Store(tmp_db)
      await store.init()

      assert await store.get_last_active() is None


  @pytest.mark.asyncio
  async def test_last_active_round_trip_and_overwrite(tmp_db):
      store = Store(tmp_db)
      await store.init()
      await store.set_last_active("qwing")
      assert await store.get_last_active() == "qwing"

      await store.set_last_active("othersapp")
      assert await store.get_last_active() == "othersapp"


  @pytest.mark.asyncio
  async def test_last_active_stored_in_meta(tmp_db):
      store = Store(tmp_db)
      await store.init()
      await store.set_last_active("qwing")

      async with aiosqlite.connect(tmp_db) as db:
          cur = await db.execute("SELECT value FROM meta WHERE key = 'last_active'")
          row = await cur.fetchone()

      assert row is not None and row[0] == "qwing"
  ```

- [ ] **Step 4: Run test to verify it fails.**
  ```
  python -m pytest tests/test_routing.py -x -q -k "last_active"
  ```
  Expected failure: `AttributeError: 'Store' object has no attribute 'get_last_active'`.

- [ ] **Step 4: Write minimal implementation.** Add `set_last_active` and `get_last_active` to the `Store` class in `src/voice_bridge/routing.py` (insert after `project_for_message`):

  ```python
      async def set_last_active(self, project: str) -> None:
          async with aiosqlite.connect(self.db_path) as db:
              await db.execute(
                  "INSERT INTO meta (key, value) VALUES ('last_active', ?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (project,),
              )
              await db.commit()

      async def get_last_active(self) -> str | None:
          async with aiosqlite.connect(self.db_path) as db:
              cur = await db.execute("SELECT value FROM meta WHERE key = 'last_active'")
              row = await cur.fetchone()
          return row[0] if row is not None else None
  ```

- [ ] **Step 4: Run test to verify it passes.**
  ```
  python -m pytest tests/test_routing.py -x -q -k "last_active"
  ```
  Expected: `3 passed`.

- [ ] **Step 4: Commit.**
  ```
  git add src/voice_bridge/routing.py tests/test_routing.py
  git commit -m "feat(routing): last_active round-trip via meta table

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

- [ ] **Step 5: Write the failing test for `session_id` round-trip.** Append to `tests/test_routing.py`. Covers: unset (project exists but no session_id) returns `None`, unknown project returns `None`, set/get, and overwrite on resume. Setting a session_id for a project not yet in the table must create the row (sessions are spawned lazily) without disturbing its later `enabled` default — so we assert the default-enabled is preserved.

  ```python
  @pytest.mark.asyncio
  async def test_session_id_unset_is_none(tmp_db):
      store = Store(tmp_db)
      await store.init([_proj("qwing", enabled=True)])

      assert await store.get_session_id("qwing") is None


  @pytest.mark.asyncio
  async def test_session_id_unknown_project_is_none(tmp_db):
      store = Store(tmp_db)
      await store.init()

      assert await store.get_session_id("ghost") is None


  @pytest.mark.asyncio
  async def test_session_id_round_trip_and_overwrite(tmp_db):
      store = Store(tmp_db)
      await store.init([_proj("qwing", enabled=True)])
      await store.set_session_id("qwing", "sess-abc")
      assert await store.get_session_id("qwing") == "sess-abc"

      await store.set_session_id("qwing", "sess-def")
      assert await store.get_session_id("qwing") == "sess-def"


  @pytest.mark.asyncio
  async def test_set_session_id_creates_row_preserving_enabled(tmp_db):
      store = Store(tmp_db)
      await store.init()  # no seeded projects
      await store.set_session_id("lazyproj", "sess-1")

      assert await store.get_session_id("lazyproj") == "sess-1"
      # row created via DEFAULT enabled=1
      assert await store.is_enabled("lazyproj") is True
  ```

- [ ] **Step 5: Run test to verify it fails.**
  ```
  python -m pytest tests/test_routing.py -x -q -k "session_id"
  ```
  Expected failure: `AttributeError: 'Store' object has no attribute 'get_session_id'`.

- [ ] **Step 5: Write minimal implementation.** Add `set_session_id` and `get_session_id` to the `Store` class in `src/voice_bridge/routing.py` (insert after `get_last_active`). The `INSERT ... ON CONFLICT` updates only `session_id`, leaving `enabled` at its `DEFAULT 1` for freshly-created rows and untouched for existing rows:

  ```python
      async def set_session_id(self, project: str, session_id: str) -> None:
          async with aiosqlite.connect(self.db_path) as db:
              await db.execute(
                  "INSERT INTO projects (name, session_id) VALUES (?, ?) "
                  "ON CONFLICT(name) DO UPDATE SET session_id=excluded.session_id",
                  (project, session_id),
              )
              await db.commit()

      async def get_session_id(self, project: str) -> str | None:
          async with aiosqlite.connect(self.db_path) as db:
              cur = await db.execute(
                  "SELECT session_id FROM projects WHERE name = ?", (project,)
              )
              row = await cur.fetchone()
          return row[0] if row is not None else None
  ```

- [ ] **Step 5: Run test to verify it passes.**
  ```
  python -m pytest tests/test_routing.py -x -q -k "session_id"
  ```
  Expected: `4 passed`.

- [ ] **Step 5: Commit.**
  ```
  git add src/voice_bridge/routing.py tests/test_routing.py
  git commit -m "feat(routing): session_id round-trip with lazy row creation

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

- [ ] **Step 6: Write the failing test for persistence across `Store` instances (restart survival).** Append to `tests/test_routing.py`. The spec requires `enabled`, `session_id`, the `message_id↔project` map, and `last_active` to survive a process restart; a new `Store` over the same db file with no re-init must read everything back. Also confirms `init()` on an existing db is non-destructive (does not wipe rows).

  ```python
  @pytest.mark.asyncio
  async def test_state_survives_new_store_instance(tmp_db):
      s1 = Store(tmp_db)
      await s1.init([_proj("qwing", enabled=True)])
      await s1.set_enabled("qwing", False)
      await s1.set_session_id("qwing", "sess-xyz")
      await s1.map_message(42, "qwing")
      await s1.set_last_active("qwing")

      # simulate restart: fresh object, same file, init() must be non-destructive
      s2 = Store(tmp_db)
      await s2.init([_proj("qwing", enabled=True)])

      assert await s2.is_enabled("qwing") is False
      assert await s2.get_session_id("qwing") == "sess-xyz"
      assert await s2.project_for_message(42) == "qwing"
      assert await s2.get_last_active() == "qwing"
  ```

- [ ] **Step 6: Run test to verify it fails — or confirm it already passes.** Run:
  ```
  python -m pytest tests/test_routing.py -x -q -k "survives"
  ```
  Expected: `1 passed`. (The `CREATE TABLE IF NOT EXISTS` schema and `INSERT OR IGNORE` seed make `init()` non-destructive, so this characterization test should pass against the current implementation. If it instead fails — e.g. seed overwriting `enabled` — that is a real regression: fix `seed()` to `INSERT OR IGNORE` and re-run until green. Per TDD, the test is written first and run before asserting the property holds.)

- [ ] **Step 6: Run the full test file to verify the whole module is green.**
  ```
  python -m pytest tests/test_routing.py -q
  ```
  Expected: `17 passed` (1 + 3 + 3 + 3 + 4 + 1 + 2 across all steps — recount and ensure all pass; the exact total is the sum of every test added above and must report `passed` with `0 failed`).

- [ ] **Step 6: Commit.**
  ```
  git add tests/test_routing.py
  git commit -m "test(routing): assert full state survives Store restart

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 3: Sanitizer (code-free voice)

**Files:**
- Create: `src/voice_bridge/sanitizer.py`
- Test: `tests/test_sanitizer.py`

**Interfaces:**

Consumes (from earlier tasks):
- Nothing. This module is pure-Python and self-contained — no imports from `config.py`, `routing.py`, or any other in-project module. It only uses the stdlib `re`. (It is *consumed* later by `bridge.py` via `prepare_outbound`, and conceptually relates to the `Outbound` dataclass `Outbound: project:str, text:str, spoken:str` — `text` is the full message, `spoken` is what `to_spoken` produces.)

Produces (exact signatures this module exposes):
- `def to_spoken(text: str, max_chars: int = 600) -> str` — strip fenced + inline code, hex colors, units (`10px`/`2rem`), file paths, URLs, `snake_case`/`camelCase`/`CONSTANT` identifiers; collapse whitespace; if truncated, append `' Detalės tekste.'`
- `def prepare_outbound(message: str) -> tuple[str, str]` — returns `(full_text, spoken)`; `spoken = to_spoken(part before the first line that is exactly '---', or the whole message if there is no such line)`.

Honors global_constraints: voice channel NEVER contains code (fenced blocks, inline code, hex colors, dimensions/units, file paths, URLs, snake_case/camelCase/CONSTANT identifiers). No external libs are used, so nothing to mock. TDD throughout.

---

- [ ] **Step 1: Write the failing test** — create `tests/test_sanitizer.py` with the complete suite below.

```python
# tests/test_sanitizer.py
"""Tests for the deterministic code-free voice sanitizer.

The voice channel must NEVER contain code: fenced blocks, inline code,
hex colors, dimensions/units, file paths, URLs, or
snake_case/camelCase/CONSTANT identifiers (global_constraints).
"""

import pytest

from voice_bridge.sanitizer import prepare_outbound, to_spoken


# --------------------------------------------------------------------------
# to_spoken: fenced code blocks
# --------------------------------------------------------------------------

def test_to_spoken_strips_fenced_block():
    text = (
        "Pataisiau klaidą.\n"
        "```python\n"
        "def foo(x):\n"
        "    return x + 1\n"
        "```\n"
        "Viskas veikia."
    )
    out = to_spoken(text)
    assert "def" not in out
    assert "foo" not in out
    assert "return" not in out
    assert "```" not in out
    assert "Pataisiau klaidą." in out
    assert "Viskas veikia." in out


def test_to_spoken_strips_fenced_block_without_language_tag():
    text = "Prieš.\n```\nrm -rf /tmp/x\n```\nPo."
    out = to_spoken(text)
    assert "rm" not in out
    assert "tmp" not in out
    assert "Prieš." in out
    assert "Po." in out


def test_to_spoken_strips_multiple_fenced_blocks():
    text = "A\n```\ncode1\n```\nB\n```\ncode2\n```\nC"
    out = to_spoken(text)
    assert "code1" not in out
    assert "code2" not in out
    assert "A" in out and "B" in out and "C" in out


# --------------------------------------------------------------------------
# to_spoken: inline code
# --------------------------------------------------------------------------

def test_to_spoken_strips_inline_code():
    text = "Paleidau `pytest -q` ir viskas žalia."
    out = to_spoken(text)
    assert "pytest" not in out
    assert "`" not in out
    assert "Paleidau" in out
    assert "viskas žalia" in out


# --------------------------------------------------------------------------
# to_spoken: hex colors
# --------------------------------------------------------------------------

def test_to_spoken_strips_hex_colors():
    text = "Pakeičiau fono spalvą į #fff ir tekstą į #1a2b3c."
    out = to_spoken(text)
    assert "#fff" not in out
    assert "#1a2b3c" not in out
    assert "Pakeičiau fono spalvą" in out


# --------------------------------------------------------------------------
# to_spoken: dimensions / units
# --------------------------------------------------------------------------

def test_to_spoken_strips_units():
    text = "Nustačiau paraštę į 10px ir šriftą į 2rem, plotis 100vh."
    out = to_spoken(text)
    assert "10px" not in out
    assert "2rem" not in out
    assert "100vh" not in out
    assert "Nustačiau paraštę" in out


# --------------------------------------------------------------------------
# to_spoken: file paths
# --------------------------------------------------------------------------

def test_to_spoken_strips_file_paths():
    text = "Redagavau failą /home/home/Projects/app/main.py ir baigiau."
    out = to_spoken(text)
    assert "/home" not in out
    assert "main.py" not in out
    assert ".py" not in out
    assert "Redagavau failą" in out
    assert "baigiau" in out


def test_to_spoken_strips_relative_paths_and_filenames():
    text = "Atnaujinau src/voice_bridge/config.py ir README.md."
    out = to_spoken(text)
    assert "config.py" not in out
    assert "src/voice_bridge" not in out
    assert "README.md" not in out
    assert "Atnaujinau" in out


# --------------------------------------------------------------------------
# to_spoken: URLs
# --------------------------------------------------------------------------

def test_to_spoken_strips_urls():
    text = "Paskelbiau čia https://example.com/deploy?id=7 — pažiūrėk."
    out = to_spoken(text)
    assert "http" not in out
    assert "example.com" not in out
    assert "Paskelbiau" in out
    assert "pažiūrėk" in out


# --------------------------------------------------------------------------
# to_spoken: code identifiers
# --------------------------------------------------------------------------

def test_to_spoken_strips_snake_case():
    text = "Pridėjau load_config funkciją ir effective_voice pagalbinę."
    out = to_spoken(text)
    assert "load_config" not in out
    assert "effective_voice" not in out
    assert "Pridėjau" in out
    assert "funkciją" in out


def test_to_spoken_strips_camel_case():
    text = "Klasė SessionManager kviečia getSessionId metodą."
    out = to_spoken(text)
    assert "SessionManager" not in out
    assert "getSessionId" not in out
    assert "Klasė" in out


def test_to_spoken_strips_constant_identifiers():
    text = "Perskaičiau TELEGRAM_BOT_TOKEN ir APPROVAL_TIMEOUT reikšmes."
    out = to_spoken(text)
    assert "TELEGRAM_BOT_TOKEN" not in out
    assert "APPROVAL_TIMEOUT" not in out
    assert "Perskaičiau" in out
    assert "reikšmes" in out


def test_to_spoken_keeps_normal_capitalized_words():
    # A single capitalized word (sentence start, proper noun) is NOT a CONSTANT
    # and must survive.
    text = "Telegram žinutė išsiųsta. Claude atsakė."
    out = to_spoken(text)
    assert "Telegram" in out
    assert "Claude" in out


# --------------------------------------------------------------------------
# to_spoken: whitespace collapse
# --------------------------------------------------------------------------

def test_to_spoken_collapses_whitespace():
    text = "Pirma   eilutė.\n\n\nAntra\teilutė."
    out = to_spoken(text)
    assert "   " not in out
    assert "\n" not in out
    assert "\t" not in out
    assert out == "Pirma eilutė. Antra eilutė."


# --------------------------------------------------------------------------
# to_spoken: length cap
# --------------------------------------------------------------------------

def test_to_spoken_truncates_and_appends_marker():
    text = "žodis " * 300  # ~1800 chars of clean prose
    out = to_spoken(text, max_chars=100)
    assert len(out) <= 100 + len(" Detalės tekste.")
    assert out.endswith(" Detalės tekste.")


def test_to_spoken_no_marker_when_under_cap():
    text = "Trumpa žinutė."
    out = to_spoken(text, max_chars=600)
    assert out == "Trumpa žinutė."
    assert "Detalės tekste." not in out


def test_to_spoken_empty_after_stripping():
    text = "```\nonly code here\n```"
    out = to_spoken(text)
    assert out == ""


# --------------------------------------------------------------------------
# to_spoken: adversarial mixed code + prose
# --------------------------------------------------------------------------

def test_to_spoken_adversarial_mixed():
    text = (
        "Baigiau migraciją.\n"
        "```sql\n"
        "ALTER TABLE messages ADD COLUMN project TEXT;\n"
        "```\n"
        "Pakeičiau spalvą į #00ff00, paraštę 12px, faile "
        "/var/lib/voice-bridge/state.db, žiūrėk https://x.io/a, "
        "funkcija map_message ir klasė Store veikia. Liko testai."
    )
    out = to_spoken(text)
    # code / technical fragments gone
    assert "ALTER" not in out
    assert "TABLE" not in out
    assert "#00ff00" not in out
    assert "12px" not in out
    assert "/var/lib" not in out
    assert "state.db" not in out
    assert "http" not in out
    assert "x.io" not in out
    assert "map_message" not in out
    # prose survives
    assert "Baigiau migraciją" in out
    assert "Liko testai" in out
    # spoken line must read cleanly (no leftover backticks or pipes)
    assert "`" not in out


# --------------------------------------------------------------------------
# prepare_outbound: splitting on a bare '---' line
# --------------------------------------------------------------------------

def test_prepare_outbound_splits_on_triple_dash():
    message = (
        "Testai žali, gali tęsti.\n"
        "---\n"
        "```python\n"
        "def f(): pass\n"
        "```\n"
        "Pakeisti failai: main.py"
    )
    full, spoken = prepare_outbound(message)
    assert full == message  # full text is the WHOLE message, unchanged
    assert spoken == "Testai žali, gali tęsti."
    assert "def f" not in spoken
    assert "main.py" not in spoken


def test_prepare_outbound_no_separator_uses_whole_message():
    message = "Trumpas atnaujinimas be techninės dalies."
    full, spoken = prepare_outbound(message)
    assert full == message
    assert spoken == "Trumpas atnaujinimas be techninės dalies."


def test_prepare_outbound_only_exact_dash_line_splits():
    # An en-dash sentence or '----' (4 dashes) is NOT the separator;
    # only a line that is EXACTLY '---'.
    message = "Pirma dalis — su brūkšniu.\nAntra eilutė tame pačiame bloke."
    full, spoken = prepare_outbound(message)
    assert full == message
    assert "Pirma dalis" in spoken
    assert "Antra eilutė tame pačiame bloke" in spoken


def test_prepare_outbound_dash_with_surrounding_whitespace_splits():
    # A '---' line may have trailing/leading spaces; still the separator.
    message = "Spoken dalis.\n   ---   \nTechninė dalis su code()."
    full, spoken = prepare_outbound(message)
    assert full == message
    assert spoken == "Spoken dalis."
    assert "Techninė" not in spoken


def test_prepare_outbound_splits_on_first_separator_only():
    message = "Antraštė.\n---\nVidurys.\n---\nGalas."
    full, spoken = prepare_outbound(message)
    assert full == message
    assert spoken == "Antraštė."
    assert "Vidurys" not in spoken
```

- [ ] **Step 2: Run test to verify it fails** — run from the project root:

```bash
cd /home/home/Projects/claude-voice-bridge && python -m pytest tests/test_sanitizer.py -q
```

Expected failure: collection error / `ModuleNotFoundError: No module named 'voice_bridge.sanitizer'` (the module does not exist yet), so every test errors out.

- [ ] **Step 3: Write minimal implementation** — create `src/voice_bridge/sanitizer.py` with the complete code below.

```python
# src/voice_bridge/sanitizer.py
"""Deterministic code-free voice sanitizer.

Guarantees the spoken (voice) channel never contains code, regardless of
agent cooperation. Strips fenced + inline code, hex colors, dimensions/units,
file paths, URLs, and snake_case/camelCase/CONSTANT identifiers, collapses
whitespace, and caps length (global_constraints: "Voice channel NEVER
contains code").

Pure stdlib (re) — no external dependencies, no I/O.
"""

from __future__ import annotations

import re

TRUNCATION_MARKER = " Detalės tekste."

# Fenced code blocks: ``` ... ``` (any/no language tag), across lines.
_FENCED = re.compile(r"```.*?```", re.DOTALL)

# Inline code spans: `code`.
_INLINE_CODE = re.compile(r"`[^`]*`")

# URLs (http/https/ftp scheme or bare www.).
_URL = re.compile(r"\b(?:https?|ftp)://\S+|\bwww\.\S+", re.IGNORECASE)

# Hex colors: #fff, #ffffff, #1a2b3c4d (3/4/6/8 hex digits).
_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{3,8}\b")

# Dimensions / units: 10px, 2rem, 100vh, 1.5em, 50%, 12pt, 3ex, 0.5vw ...
_UNIT = re.compile(
    r"\b\d+(?:\.\d+)?(?:px|rem|em|ex|vh|vw|vmin|vmax|pt|pc|cm|mm|in|ch|fr|deg|ms|s)\b"
    r"|\b\d+(?:\.\d+)?%",
    re.IGNORECASE,
)

# File paths: absolute (/a/b/c), relative (a/b/c), or any token with a
# file extension (main.py, README.md). Must contain a '/' or a '.<ext>'.
_PATH = re.compile(
    r"(?:\.{0,2}/)?[\w.\-]+(?:/[\w.\-]+)+/?"          # has at least one '/'
    r"|\b[\w\-]+\.[A-Za-z][\w]{0,7}\b"                # filename.ext
)

# CONSTANT_CASE: two+ segments of UPPERCASE/digits joined by underscores,
# or a single all-caps run with an underscore.
_CONSTANT = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")

# snake_case: lowercase/digit segments joined by underscores (load_config).
_SNAKE = re.compile(r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b")

# camelCase / PascalCase with an internal capital (getSessionId, SessionManager).
# Requires a lowercase letter followed later by an uppercase letter inside one token.
_CAMEL = re.compile(r"\b[A-Za-z]+[a-z][A-Z][A-Za-z0-9]*\b")

# Bare separator line: a line that is exactly '---' (optional surrounding ws).
_SEPARATOR_LINE = re.compile(r"^\s*---\s*$")

# Whitespace runs (incl. newlines/tabs) to collapse to a single space.
_WS = re.compile(r"\s+")

# Leftover lone punctuation tokens (orphaned by stripping) e.g. " : ", " | ".
_LONE_SYMBOL = re.compile(r"(?<=\s)[|:;=<>~^*_+/\\]+(?=\s)")


def to_spoken(text: str, max_chars: int = 600) -> str:
    """Return a code-free, voice-friendly version of ``text``.

    Strips fenced + inline code, URLs, hex colors, units, file paths, and
    code identifiers (snake_case/camelCase/CONSTANT), collapses whitespace,
    and caps the result at ``max_chars`` (appending ``TRUNCATION_MARKER`` if
    truncated).
    """
    s = text

    # Order matters: remove fenced blocks before anything else can match inside.
    s = _FENCED.sub(" ", s)
    s = _INLINE_CODE.sub(" ", s)
    s = _URL.sub(" ", s)
    s = _HEX_COLOR.sub(" ", s)
    s = _UNIT.sub(" ", s)
    s = _PATH.sub(" ", s)
    s = _CONSTANT.sub(" ", s)
    s = _SNAKE.sub(" ", s)
    s = _CAMEL.sub(" ", s)
    s = _LONE_SYMBOL.sub(" ", s)

    # Collapse whitespace and trim.
    s = _WS.sub(" ", s).strip()

    # Tidy spaces left before sentence punctuation by removed tokens.
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    s = _WS.sub(" ", s).strip()

    if len(s) > max_chars:
        s = s[:max_chars].rstrip()
        # Avoid cutting mid-word: drop a trailing partial word if present.
        if " " in s:
            s = s[: s.rstrip().rfind(" ")].rstrip()
        s = s.rstrip(".,!?;: ") + TRUNCATION_MARKER

    return s


def prepare_outbound(message: str) -> tuple[str, str]:
    """Split an outbound message into ``(full_text, spoken)``.

    ``full_text`` is the entire message unchanged. ``spoken`` is
    ``to_spoken`` applied to the part before the first line that is exactly
    ``'---'`` (or the whole message if there is no such line).
    """
    lines = message.split("\n")
    spoken_source = message
    for i, line in enumerate(lines):
        if _SEPARATOR_LINE.match(line):
            spoken_source = "\n".join(lines[:i])
            break
    return message, to_spoken(spoken_source)
```

- [ ] **Step 4: Run test to verify it passes** — run:

```bash
cd /home/home/Projects/claude-voice-bridge && python -m pytest tests/test_sanitizer.py -q
```

Expected: all tests pass, e.g. `24 passed in 0.0Xs` (no failures, no errors).

- [ ] **Step 5: Commit** — stage both files and commit with a conventional message and the required trailer:

```bash
cd /home/home/Projects/claude-voice-bridge && git add src/voice_bridge/sanitizer.py tests/test_sanitizer.py && git commit -m "feat(sanitizer): code-free voice via to_spoken + prepare_outbound

Strip fenced/inline code, hex colors, units, paths, URLs and
snake_case/camelCase/CONSTANT identifiers; collapse whitespace; cap
length with 'Detalės tekste.' marker. prepare_outbound splits on the
first bare '---' line.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: TTS interface + OpenAI + Piper

**Files:**
- Create: `src/voice_bridge/tts/__init__.py`
- Create: `src/voice_bridge/tts/openai_tts.py`
- Create: `src/voice_bridge/tts/piper_tts.py`
- Test: `tests/test_tts.py`

**Interfaces:**

Consumes (from Task 1 / config.py):
- `@dataclass Config: telegram_bot_token:str, telegram_allowed_user_id:int, anthropic_api_key:str, openai_api_key:str, tts_backend:str, tts_voice:str, piper_voice_path:str, whisper_model:str, autonomy_mode:str, approval_timeout:int, db_path:str` — `get_tts` dispatches on `cfg.tts_backend` and pulls `cfg.openai_api_key` / `cfg.piper_voice_path`.

Produces (this task, exact contract signatures):
- `class TTSBackend(Protocol): async def synthesize(self, text:str, voice:str) -> bytes  # OGG/Opus`
- `def get_tts(cfg: Config) -> TTSBackend  # dispatch on cfg.tts_backend`
- `def available_voices(backend:str) -> list[str]`
- `class OpenAITTS(TTSBackend): def __init__(self, api_key:str); async def synthesize(self, text:str, voice:str) -> bytes`
- `class PiperTTS(TTSBackend): def __init__(self, voice_path:str); async def synthesize(self, text:str, voice:str) -> bytes`

Constraints honored: every `synthesize` returns OGG/Opus bytes (OpenAI `response_format="opus"`; Piper PCM piped through ffmpeg `libopus`/ogg); all blocking work (OpenAI client call, piper/ffmpeg subprocesses) runs off the event loop via `run_in_executor` / `asyncio.create_subprocess_exec`; the `openai` client and all subprocesses are mocked in tests so they run with no network, no secrets, no piper/ffmpeg binaries.

---

TDD steps:

- [ ] **Step 1: Write the failing test for `available_voices` + `TTSBackend` protocol.** Create `tests/test_tts.py` with the first block:

  ```python
  import asyncio
  from unittest.mock import AsyncMock, MagicMock, patch

  import pytest

  from voice_bridge.config import Config
  from voice_bridge.tts import TTSBackend, available_voices, get_tts
  from voice_bridge.tts.openai_tts import OpenAITTS
  from voice_bridge.tts.piper_tts import PiperTTS


  def _cfg(**overrides) -> Config:
      base = dict(
          telegram_bot_token="t",
          telegram_allowed_user_id=1,
          anthropic_api_key="a",
          openai_api_key="sk-test",
          tts_backend="openai",
          tts_voice="nova",
          piper_voice_path="/opt/piper/lt_LT.onnx",
          whisper_model="large-v3",
          autonomy_mode="safe",
          approval_timeout=300,
          db_path=":memory:",
      )
      base.update(overrides)
      return Config(**base)


  def test_available_voices_openai_lists_known_voices():
      voices = available_voices("openai")
      assert isinstance(voices, list)
      assert "nova" in voices
      assert "alloy" in voices
      assert all(isinstance(v, str) for v in voices)


  def test_available_voices_piper_returns_default_list():
      voices = available_voices("piper")
      assert voices == ["default"]


  def test_available_voices_unknown_backend_returns_empty():
      assert available_voices("bogus") == []


  def test_ttsbackend_is_runtime_checkable_protocol():
      assert isinstance(OpenAITTS("sk-test"), TTSBackend)
      assert isinstance(PiperTTS("/opt/piper/lt_LT.onnx"), TTSBackend)
      assert not isinstance(object(), TTSBackend)
  ```

- [ ] **Step 1: Run test to verify it fails.** Command: `python -m pytest tests/test_tts.py -q`. Expected: collection/import error `ModuleNotFoundError: No module named 'voice_bridge.tts'` (and once partially created, `ImportError: cannot import name 'available_voices'`).

- [ ] **Step 1: Write minimal implementation of `tts/__init__.py`.** Create `src/voice_bridge/tts/__init__.py`:

  ```python
  """TTS backend protocol, factory, and voice listing."""
  from __future__ import annotations

  from typing import Protocol, runtime_checkable

  from voice_bridge.config import Config

  _OPENAI_VOICES = [
      "alloy",
      "echo",
      "fable",
      "onyx",
      "nova",
      "shimmer",
  ]


  @runtime_checkable
  class TTSBackend(Protocol):
      """A text-to-speech backend that emits OGG/Opus bytes."""

      async def synthesize(self, text: str, voice: str) -> bytes:
          """Return OGG/Opus-encoded audio for ``text`` in ``voice``."""
          ...


  def get_tts(cfg: Config) -> TTSBackend:
      """Construct the configured TTS backend, dispatching on ``cfg.tts_backend``."""
      backend = cfg.tts_backend
      if backend == "openai":
          from voice_bridge.tts.openai_tts import OpenAITTS

          return OpenAITTS(cfg.openai_api_key)
      if backend == "piper":
          from voice_bridge.tts.piper_tts import PiperTTS

          return PiperTTS(cfg.piper_voice_path)
      raise ValueError(f"unknown TTS backend: {backend!r}")


  def available_voices(backend: str) -> list[str]:
      """List selectable voices for ``backend`` (empty list if unknown)."""
      if backend == "openai":
          return list(_OPENAI_VOICES)
      if backend == "piper":
          return ["default"]
      return []
  ```

  Then create the two backend stubs so the protocol-isinstance test can import them. Create `src/voice_bridge/tts/openai_tts.py`:

  ```python
  """OpenAI TTS backend."""
  from __future__ import annotations

  import asyncio

  from openai import OpenAI

  _MODEL = "gpt-4o-mini-tts"


  class OpenAITTS:
      """OpenAI text-to-speech, emitting OGG/Opus bytes."""

      def __init__(self, api_key: str) -> None:
          self._client = OpenAI(api_key=api_key)

      async def synthesize(self, text: str, voice: str) -> bytes:
          def _call() -> bytes:
              response = self._client.audio.speech.create(
                  model=_MODEL,
                  voice=voice,
                  input=text,
                  response_format="opus",
              )
              return response.read()

          return await asyncio.get_running_loop().run_in_executor(None, _call)
  ```

  And create `src/voice_bridge/tts/piper_tts.py`:

  ```python
  """Local Piper TTS backend; emits OGG/Opus via ffmpeg."""
  from __future__ import annotations

  import asyncio


  class PiperTTS:
      """Piper text-to-speech; raw PCM piped through ffmpeg to OGG/Opus."""

      def __init__(self, voice_path: str) -> None:
          self._voice_path = voice_path

      async def synthesize(self, text: str, voice: str) -> bytes:
          piper = await asyncio.create_subprocess_exec(
              "piper",
              "--model",
              self._voice_path,
              "--output-raw",
              stdin=asyncio.subprocess.PIPE,
              stdout=asyncio.subprocess.PIPE,
              stderr=asyncio.subprocess.PIPE,
          )
          pcm, piper_err = await piper.communicate(text.encode("utf-8"))
          if piper.returncode != 0:
              raise RuntimeError(
                  f"piper failed ({piper.returncode}): {piper_err.decode('utf-8', 'replace')}"
              )

          ffmpeg = await asyncio.create_subprocess_exec(
              "ffmpeg",
              "-f",
              "s16le",
              "-ar",
              "22050",
              "-ac",
              "1",
              "-i",
              "pipe:0",
              "-c:a",
              "libopus",
              "-f",
              "ogg",
              "pipe:1",
              stdin=asyncio.subprocess.PIPE,
              stdout=asyncio.subprocess.PIPE,
              stderr=asyncio.subprocess.PIPE,
          )
          ogg, ffmpeg_err = await ffmpeg.communicate(pcm)
          if ffmpeg.returncode != 0:
              raise RuntimeError(
                  f"ffmpeg failed ({ffmpeg.returncode}): {ffmpeg_err.decode('utf-8', 'replace')}"
              )
          return ogg
  ```

- [ ] **Step 1: Run test to verify it passes.** Command: `python -m pytest tests/test_tts.py -q`. Expected: `4 passed`.

- [ ] **Step 1: Commit.** Commands:
  ```bash
  git add src/voice_bridge/tts/__init__.py src/voice_bridge/tts/openai_tts.py src/voice_bridge/tts/piper_tts.py tests/test_tts.py
  git commit -m "feat(tts): protocol, factory, and available_voices

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

- [ ] **Step 2: Write the failing test for `get_tts` factory dispatch.** Append to `tests/test_tts.py`:

  ```python
  def test_get_tts_openai_builds_openai_backend():
      with patch("voice_bridge.tts.openai_tts.OpenAI") as mock_openai:
          backend = get_tts(_cfg(tts_backend="openai", openai_api_key="sk-xyz"))
      assert isinstance(backend, OpenAITTS)
      mock_openai.assert_called_once_with(api_key="sk-xyz")


  def test_get_tts_piper_builds_piper_backend():
      backend = get_tts(_cfg(tts_backend="piper", piper_voice_path="/v/lt.onnx"))
      assert isinstance(backend, PiperTTS)
      assert backend._voice_path == "/v/lt.onnx"


  def test_get_tts_unknown_backend_raises():
      with pytest.raises(ValueError, match="unknown TTS backend"):
          get_tts(_cfg(tts_backend="bogus"))
  ```

- [ ] **Step 2: Run test to verify it fails.** Command: `python -m pytest tests/test_tts.py -k get_tts -q`. Expected: `3 passed` if the Step 1 impl already satisfies them — but the OpenAI dispatch test will fail first if run before patching is in place. If all three already pass against the Step 1 implementation, note "no impl change needed" and skip straight to the commit. Expected failure if `get_tts` were missing: `ImportError`/`AttributeError`. (Since `get_tts` is implemented, run to confirm; expected `3 passed`.)

- [ ] **Step 2: Write minimal implementation.** No new code needed — `get_tts` from Step 1 already dispatches on `cfg.tts_backend`, constructs `OpenAITTS(cfg.openai_api_key)` / `PiperTTS(cfg.piper_voice_path)`, and raises `ValueError` for unknown backends. (This step is intentionally a no-op confirming the factory contract is fully covered by tests.)

- [ ] **Step 2: Run test to verify it passes.** Command: `python -m pytest tests/test_tts.py -k get_tts -q`. Expected: `3 passed`.

- [ ] **Step 2: Commit.** Commands:
  ```bash
  git add tests/test_tts.py
  git commit -m "test(tts): cover get_tts factory dispatch

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

- [ ] **Step 3: Write the failing test for `OpenAITTS.synthesize`.** Append to `tests/test_tts.py`:

  ```python
  @pytest.mark.asyncio
  async def test_openai_synthesize_requests_opus_and_returns_bytes():
      fake_response = MagicMock()
      fake_response.read.return_value = b"OggS-opus-bytes"

      fake_client = MagicMock()
      fake_client.audio.speech.create.return_value = fake_response

      with patch("voice_bridge.tts.openai_tts.OpenAI", return_value=fake_client) as mock_openai:
          backend = OpenAITTS("sk-abc")
          out = await backend.synthesize("Sveiki, viskas gerai.", "nova")

      assert out == b"OggS-opus-bytes"
      mock_openai.assert_called_once_with(api_key="sk-abc")
      fake_client.audio.speech.create.assert_called_once_with(
          model="gpt-4o-mini-tts",
          voice="nova",
          input="Sveiki, viskas gerai.",
          response_format="opus",
      )


  @pytest.mark.asyncio
  async def test_openai_synthesize_runs_off_the_event_loop():
      fake_response = MagicMock()
      fake_response.read.return_value = b"x"
      fake_client = MagicMock()
      fake_client.audio.speech.create.return_value = fake_response

      loop = asyncio.get_running_loop()
      calls: dict = {}

      def spy_executor(executor, func, *args):
          calls["used_executor"] = True
          return loop.run_in_executor(executor, func, *args)

      with patch("voice_bridge.tts.openai_tts.OpenAI", return_value=fake_client):
          backend = OpenAITTS("sk-abc")
          with patch.object(loop, "run_in_executor", side_effect=spy_executor):
              await backend.synthesize("labas", "echo")

      assert calls.get("used_executor") is True
  ```

- [ ] **Step 3: Run test to verify it fails.** Command: `python -m pytest tests/test_tts.py -k openai_synthesize -q`. Expected: `2 passed` against the Step 1 implementation (it already requests `response_format="opus"` and uses `run_in_executor`). If instead the implementation had not yet been written, expected failure would be `AssertionError` on `create.assert_called_once_with(... response_format="opus" ...)`. Run to confirm `2 passed`.

- [ ] **Step 3: Write minimal implementation.** No change required — `OpenAITTS.synthesize` from Step 1 already calls `audio.speech.create(model="gpt-4o-mini-tts", voice=voice, input=text, response_format="opus")` inside `run_in_executor` and returns `response.read()`. Confirmed by the two passing assertions.

- [ ] **Step 3: Run test to verify it passes.** Command: `python -m pytest tests/test_tts.py -k openai_synthesize -q`. Expected: `2 passed`.

- [ ] **Step 3: Commit.** Commands:
  ```bash
  git add tests/test_tts.py
  git commit -m "test(tts): assert OpenAI backend requests opus off-loop

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

- [ ] **Step 4: Write the failing test for `PiperTTS.synthesize` (subprocess mocked).** Append to `tests/test_tts.py`:

  ```python
  def _fake_proc(stdout: bytes, stderr: bytes = b"", returncode: int = 0):
      proc = MagicMock()
      proc.communicate = AsyncMock(return_value=(stdout, stderr))
      proc.returncode = returncode
      return proc


  @pytest.mark.asyncio
  async def test_piper_synthesize_pipes_pcm_through_ffmpeg_to_opus():
      piper_proc = _fake_proc(stdout=b"RAWPCM")
      ffmpeg_proc = _fake_proc(stdout=b"OggS-piper-opus")

      created = []

      async def fake_exec(*args, **kwargs):
          created.append(args)
          return piper_proc if args[0] == "piper" else ffmpeg_proc

      with patch(
          "voice_bridge.tts.piper_tts.asyncio.create_subprocess_exec",
          side_effect=fake_exec,
      ):
          backend = PiperTTS("/opt/piper/lt_LT.onnx")
          out = await backend.synthesize("Sveiki", "default")

      assert out == b"OggS-piper-opus"
      # piper invoked with the configured model path
      assert created[0][0] == "piper"
      assert "/opt/piper/lt_LT.onnx" in created[0]
      assert "--output-raw" in created[0]
      # ffmpeg invoked to encode opus in an ogg container
      assert created[1][0] == "ffmpeg"
      assert "libopus" in created[1]
      assert "ogg" in created[1]
      # text fed to piper stdin; piper pcm fed to ffmpeg stdin
      piper_proc.communicate.assert_awaited_once_with(b"Sveiki")
      ffmpeg_proc.communicate.assert_awaited_once_with(b"RAWPCM")


  @pytest.mark.asyncio
  async def test_piper_synthesize_raises_when_piper_fails():
      piper_proc = _fake_proc(stdout=b"", stderr=b"model missing", returncode=1)

      async def fake_exec(*args, **kwargs):
          return piper_proc

      with patch(
          "voice_bridge.tts.piper_tts.asyncio.create_subprocess_exec",
          side_effect=fake_exec,
      ):
          backend = PiperTTS("/bad.onnx")
          with pytest.raises(RuntimeError, match="piper failed"):
              await backend.synthesize("x", "default")


  @pytest.mark.asyncio
  async def test_piper_synthesize_raises_when_ffmpeg_fails():
      piper_proc = _fake_proc(stdout=b"RAWPCM")
      ffmpeg_proc = _fake_proc(stdout=b"", stderr=b"enc error", returncode=1)

      async def fake_exec(*args, **kwargs):
          return piper_proc if args[0] == "piper" else ffmpeg_proc

      with patch(
          "voice_bridge.tts.piper_tts.asyncio.create_subprocess_exec",
          side_effect=fake_exec,
      ):
          backend = PiperTTS("/opt/piper/lt_LT.onnx")
          with pytest.raises(RuntimeError, match="ffmpeg failed"):
              await backend.synthesize("x", "default")
  ```

- [ ] **Step 4: Run test to verify it fails.** Command: `python -m pytest tests/test_tts.py -k piper_synthesize -q`. Expected: `3 passed` against the Step 1 implementation (it already shells `piper --output-raw`, pipes to `ffmpeg ... libopus ... ogg`, and raises `RuntimeError` on non-zero returns). If the impl had been missing the ffmpeg stage, expected failure would be `AssertionError: assert 'libopus' in (...)`. Run to confirm `3 passed`.

- [ ] **Step 4: Write minimal implementation.** No change required — `PiperTTS.synthesize` from Step 1 already: spawns `piper --model <voice_path> --output-raw`, feeds `text` to its stdin, raises `RuntimeError("piper failed ...")` on non-zero exit, pipes the PCM into `ffmpeg -f s16le ... -c:a libopus -f ogg pipe:1`, raises `RuntimeError("ffmpeg failed ...")` on non-zero exit, and returns the OGG/Opus bytes. Confirmed by the three passing assertions.

- [ ] **Step 4: Run test to verify it passes.** Command: `python -m pytest tests/test_tts.py -k piper_synthesize -q`. Expected: `3 passed`.

- [ ] **Step 4: Commit.** Commands:
  ```bash
  git add tests/test_tts.py
  git commit -m "test(tts): assert Piper backend emits ogg/opus via ffmpeg

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

- [ ] **Step 5: Run the full TTS suite to confirm the whole module is green.** Command: `python -m pytest tests/test_tts.py -q`. Expected: `12 passed` with no network calls, no `openai` requests, and no `piper`/`ffmpeg` binaries invoked (all mocked).

---

### Task 5: STT (faster-whisper)

**Files:**
- Create: `src/voice_bridge/stt.py`
- Test: `tests/test_stt.py`

**Interfaces:**

Consumes (from earlier tasks / contract):
- `Config.whisper_model: str` — the `WHISPER_MODEL` value (default `large-v3`) is passed as `model_name` when `bridge.py` constructs the `Transcriber`. No code dependency on `config.py` inside `stt.py`; the caller passes the string.
- Global constraints honored: STT must accept OGG/Opus bytes; all blocking work (`faster_whisper.WhisperModel`) runs off the event loop via `run_in_executor`; language is `lt`.

Produces (exact signatures this module exposes):
- `class Transcriber: def __init__(self, model_name: str, language: str = 'lt')`
- `async def transcribe(self, audio: bytes) -> str`

Behavior contract for `transcribe`:
- Lazily constructs a `faster_whisper.WhisperModel(model_name)` on first call (model load is expensive; keep it off `__init__` so construction is cheap and import-safe).
- Writes the OGG/Opus `audio` bytes to a temp file with suffix `.ogg`, calls `model.transcribe(path, language=self.language)`, which returns `(segments, info)`; joins each `segment.text` and returns the stripped result.
- The whole synchronous block runs inside `run_in_executor` (default thread pool) so the event loop is never blocked.
- Always removes the temp file (even on error).

TDD steps:

- [ ] **Step 1: Write the failing test** — create `tests/test_stt.py` with the COMPLETE code below. It mocks `faster_whisper.WhisperModel` (via the import path used inside `stt.py`) so no model/GPU/network is needed, asserts `language='lt'` is forwarded, asserts segment texts are joined and stripped, and asserts the work happened off-loop (temp file written + cleaned up). Uses `pytest-asyncio`.

```python
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from voice_bridge.stt import Transcriber


def _fake_model_factory(captured):
    """Return a MagicMock WhisperModel class that records construction + calls."""
    def transcribe(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        # faster-whisper returns (segments_iterable, info)
        segments = [
            SimpleNamespace(text=" Labas "),
            SimpleNamespace(text="pasauli"),
        ]
        info = SimpleNamespace(language="lt", language_probability=0.99)
        return iter(segments), info

    instance = MagicMock()
    instance.transcribe.side_effect = transcribe

    model_cls = MagicMock(return_value=instance)
    captured["model_cls"] = model_cls
    return model_cls


@pytest.mark.asyncio
async def test_transcribe_joins_segments_and_strips():
    captured = {}
    model_cls = _fake_model_factory(captured)
    with patch("voice_bridge.stt.WhisperModel", model_cls):
        t = Transcriber("large-v3")
        result = await t.transcribe(b"OggS-fake-opus-bytes")
    assert result == "Labas pasauli"


@pytest.mark.asyncio
async def test_transcribe_passes_language_lt_by_default():
    captured = {}
    model_cls = _fake_model_factory(captured)
    with patch("voice_bridge.stt.WhisperModel", model_cls):
        t = Transcriber("large-v3")
        await t.transcribe(b"OggS-fake")
    assert captured["kwargs"]["language"] == "lt"


@pytest.mark.asyncio
async def test_transcribe_honors_custom_language():
    captured = {}
    model_cls = _fake_model_factory(captured)
    with patch("voice_bridge.stt.WhisperModel", model_cls):
        t = Transcriber("large-v3", language="en")
        await t.transcribe(b"OggS-fake")
    assert captured["kwargs"]["language"] == "en"


@pytest.mark.asyncio
async def test_transcribe_constructs_model_with_name_lazily():
    captured = {}
    model_cls = _fake_model_factory(captured)
    with patch("voice_bridge.stt.WhisperModel", model_cls):
        t = Transcriber("medium")
        # not constructed at __init__ time
        captured["model_cls"].assert_not_called()
        await t.transcribe(b"OggS-fake")
    captured["model_cls"].assert_called_once_with("medium")


@pytest.mark.asyncio
async def test_transcribe_reuses_loaded_model():
    captured = {}
    model_cls = _fake_model_factory(captured)
    with patch("voice_bridge.stt.WhisperModel", model_cls):
        t = Transcriber("large-v3")
        await t.transcribe(b"OggS-1")
        await t.transcribe(b"OggS-2")
    # model loaded exactly once across two transcriptions
    captured["model_cls"].assert_called_once()


@pytest.mark.asyncio
async def test_transcribe_writes_ogg_temp_file_and_cleans_up(tmp_path):
    seen_paths = []

    def transcribe(path, **kwargs):
        seen_paths.append(path)
        # file must exist with the ogg bytes while transcribing
        with open(path, "rb") as fh:
            assert fh.read() == b"OggS-payload"
        return iter([SimpleNamespace(text="ok")]), SimpleNamespace(language="lt")

    instance = MagicMock()
    instance.transcribe.side_effect = transcribe
    model_cls = MagicMock(return_value=instance)

    with patch("voice_bridge.stt.WhisperModel", model_cls):
        t = Transcriber("large-v3")
        result = await t.transcribe(b"OggS-payload")

    assert result == "ok"
    assert seen_paths and seen_paths[0].endswith(".ogg")
    # temp file removed after transcription
    import os
    assert not os.path.exists(seen_paths[0])


@pytest.mark.asyncio
async def test_transcribe_runs_off_event_loop():
    """The blocking model call must not run on the main loop thread."""
    main_thread_id = None

    import threading
    main_thread_id = threading.get_ident()
    call_thread_ids = []

    def transcribe(path, **kwargs):
        call_thread_ids.append(threading.get_ident())
        return iter([SimpleNamespace(text="x")]), SimpleNamespace(language="lt")

    instance = MagicMock()
    instance.transcribe.side_effect = transcribe
    model_cls = MagicMock(return_value=instance)

    with patch("voice_bridge.stt.WhisperModel", model_cls):
        t = Transcriber("large-v3")
        await t.transcribe(b"OggS-fake")

    assert call_thread_ids
    assert call_thread_ids[0] != main_thread_id


@pytest.mark.asyncio
async def test_transcribe_cleans_up_temp_file_on_error():
    seen_paths = []

    def transcribe(path, **kwargs):
        seen_paths.append(path)
        raise RuntimeError("decode failed")

    instance = MagicMock()
    instance.transcribe.side_effect = transcribe
    model_cls = MagicMock(return_value=instance)

    with patch("voice_bridge.stt.WhisperModel", model_cls):
        t = Transcriber("large-v3")
        with pytest.raises(RuntimeError, match="decode failed"):
            await t.transcribe(b"OggS-fake")

    import os
    assert seen_paths and not os.path.exists(seen_paths[0])
```

- [ ] **Step 2: Run test to verify it fails** — run:
  ```
  python -m pytest tests/test_stt.py -q
  ```
  Expected failure: collection/import error `ModuleNotFoundError: No module named 'voice_bridge.stt'` (the module does not exist yet), so every test errors at import.

- [ ] **Step 3: Write minimal implementation** — create `src/voice_bridge/stt.py` with the COMPLETE code below. It imports `WhisperModel` at module top (so the test patch target `voice_bridge.stt.WhisperModel` resolves), lazily loads the model, writes OGG/Opus bytes to a `.ogg` temp file, runs the blocking transcription via `run_in_executor`, joins segment texts, and cleans up the temp file in a `finally`.

```python
"""Speech-to-text via faster-whisper.

Accepts Telegram OGG/Opus voice bytes and returns a transcript. The blocking
faster-whisper model load and inference run off the event loop in a worker
thread so the single asyncio loop is never blocked. Default language is
Lithuanian (``lt``).
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from faster_whisper import WhisperModel


class Transcriber:
    """Wraps a faster-whisper model for OGG/Opus -> text transcription."""

    def __init__(self, model_name: str, language: str = "lt") -> None:
        self.model_name = model_name
        self.language = language
        self._model: WhisperModel | None = None

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(self.model_name)
        return self._model

    def _transcribe_sync(self, audio: bytes) -> str:
        fd, path = tempfile.mkstemp(suffix=".ogg")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(audio)
            segments, _info = self._get_model().transcribe(
                path, language=self.language
            )
            text = "".join(segment.text for segment in segments)
            return text.strip()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe OGG/Opus ``audio`` bytes to text (off the event loop)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)
```

- [ ] **Step 4: Run test to verify it passes** — run:
  ```
  python -m pytest tests/test_stt.py -q
  ```
  Expected: `8 passed` (all transcribe tests green — joining/stripping, language forwarding, lazy single load, `.ogg` temp file creation + cleanup, off-loop execution, cleanup on error).

- [ ] **Step 5: Commit** — run:
  ```
  git add src/voice_bridge/stt.py tests/test_stt.py
  git commit -m "feat(stt): faster-whisper Transcriber for OGG/Opus, lt, off-loop

Lazy WhisperModel load, temp .ogg file, run_in_executor so the asyncio
loop is never blocked; segments joined and stripped. WhisperModel mocked
in tests (no model/GPU/network).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 6: Approvals + canUseTool

**Files:**
- Create: `src/voice_bridge/approvals.py`
- Test: `tests/test_approvals.py`
- Modify: none (depends on `config.py` from Task 1 for `ProjectConfig`, `Config`, `effective_autonomy`; do not edit it)

**Interfaces:**

Consumes (from Task 1 `config.py` — import, do not redefine):
- `@dataclass Config: telegram_bot_token:str, telegram_allowed_user_id:int, anthropic_api_key:str, openai_api_key:str, tts_backend:str, tts_voice:str, piper_voice_path:str, whisper_model:str, autonomy_mode:str, approval_timeout:int, db_path:str`
- `@dataclass ProjectConfig: name:str, cwd:str, enabled:bool=True, autonomy:str|None=None, voice:str|None=None, model:str|None=None, system_prompt_extra:str=''`
- `def effective_autonomy(project: ProjectConfig, cfg: Config) -> str  # project.autonomy or cfg.autonomy_mode`

Consumes (from `claude-agent-sdk` — used only inside `make_can_use_tool`; mocked/imported lazily so tests need no live SDK):
- `PermissionResultAllow`, `PermissionResultDeny` permission result types.

Produces (exact signatures this task exposes):
- `def is_risky(tool_name: str, tool_input: dict, cwd: str) -> bool`
- `def parse_yes_no(text: str) -> bool | None`
- `class ApprovalManager: def __init__(self, send_question: Callable[[str, str], Awaitable[int]], timeout: int)`
  - `async def request(self, project: str, tool_name: str, tool_input: dict) -> bool`
  - `def resolve(self, message_id: int, approved: bool) -> bool`
  - `def has_pending(self, message_id: int) -> bool`
- `def make_can_use_tool(project: ProjectConfig, cfg: Config, manager: ApprovalManager) -> Callable`

Notes for the implementer:
- `make_can_use_tool` honors `effective_autonomy(project, cfg)`: `full` → allow all; `ask` → request all; `safe` → request only when `is_risky`.
- `ApprovalManager.request` calls `send_question(project, text)` to get the question's `message_id`, stores an `asyncio.Future` keyed by that id, and awaits it with `asyncio.wait_for(..., timeout)`; on `asyncio.TimeoutError` it returns `False` and clears the pending entry.
- `resolve(message_id, approved)` sets the matching future's result and returns `True`; returns `False` if no pending future matched.
- The `canUseTool` callable signature follows the SDK: `async def can_use_tool(tool_name: str, tool_input: dict, context) -> PermissionResultAllow | PermissionResultDeny`.

---

- [ ] **Step 1: Write the failing test for `is_risky`**

Create `tests/test_approvals.py`:

```python
import asyncio

import pytest

from voice_bridge.approvals import (
    ApprovalManager,
    is_risky,
    make_can_use_tool,
    parse_yes_no,
)
from voice_bridge.config import Config, ProjectConfig


CWD = "/home/home/Projects/qwing"


def _cfg(autonomy_mode: str = "safe", approval_timeout: int = 300) -> Config:
    return Config(
        telegram_bot_token="t",
        telegram_allowed_user_id=1,
        anthropic_api_key="a",
        openai_api_key="o",
        tts_backend="openai",
        tts_voice="nova",
        piper_voice_path="/x.onnx",
        whisper_model="large-v3",
        autonomy_mode=autonomy_mode,
        approval_timeout=approval_timeout,
        db_path=":memory:",
    )


def _proj(autonomy=None) -> ProjectConfig:
    return ProjectConfig(name="qwing", cwd=CWD, autonomy=autonomy)


@pytest.mark.parametrize(
    "tool_name,tool_input",
    [
        ("Bash", {"command": "git push origin master"}),
        ("Bash", {"command": "rm -rf build"}),
        ("Bash", {"command": "ssh root@server uptime"}),
        ("Bash", {"command": "npm install left-pad"}),
        ("Bash", {"command": "pip install requests"}),
        ("Bash", {"command": "kubectl apply -f deploy.yaml"}),
        ("Bash", {"command": "vercel deploy"}),
        ("Bash", {"command": "send 0.5 ETH to my wallet"}),
        ("Write", {"file_path": "/etc/hosts", "content": "x"}),
        ("Edit", {"file_path": "/home/home/Projects/other/a.py"}),
    ],
)
def test_is_risky_true(tool_name, tool_input):
    assert is_risky(tool_name, tool_input, CWD) is True


@pytest.mark.parametrize(
    "tool_name,tool_input",
    [
        ("Read", {"file_path": "/home/home/Projects/qwing/a.py"}),
        ("Grep", {"pattern": "def main"}),
        ("Bash", {"command": "pytest -q"}),
        ("Bash", {"command": "ls -la"}),
        ("Bash", {"command": "git status"}),
        ("Bash", {"command": "npm run build"}),
        ("Edit", {"file_path": "/home/home/Projects/qwing/src/a.py"}),
        ("Write", {"file_path": "/home/home/Projects/qwing/new.py", "content": "x"}),
    ],
)
def test_is_risky_false(tool_name, tool_input):
    assert is_risky(tool_name, tool_input, CWD) is False
```

- [ ] **Step 2: Run test to verify it fails**

Command: `python -m pytest tests/test_approvals.py -q`
Expected: collection/import error `ModuleNotFoundError: No module named 'voice_bridge.approvals'` (red).

- [ ] **Step 3: Write minimal implementation of `is_risky` (and module skeleton)**

Create `src/voice_bridge/approvals.py`:

```python
"""Risk classification, yes/no parsing, voice-approval futures, canUseTool factory."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
from typing import Awaitable, Callable

from voice_bridge.config import Config, ProjectConfig, effective_autonomy

# --- Risk classification --------------------------------------------------

# Bash command verbs / phrases that are always risky.
_RISKY_COMMAND_PATTERNS = [
    re.compile(r"\bgit\s+push\b"),
    re.compile(r"\bgit\s+push\b.*"),
    re.compile(r"\brm\b"),
    re.compile(r"\bssh\b"),
    re.compile(r"\bscp\b"),
    re.compile(r"\brsync\b"),
    re.compile(r"\bdeploy\b"),
    re.compile(r"\bvercel\b"),
    re.compile(r"\bnetlify\b"),
    re.compile(r"\bkubectl\b"),
    re.compile(r"\bdocker\s+push\b"),
    re.compile(r"\bterraform\s+apply\b"),
    re.compile(r"\bnpm\s+install\b"),
    re.compile(r"\byarn\s+add\b"),
    re.compile(r"\bpnpm\s+(add|install)\b"),
    re.compile(r"\bpip\s+install\b"),
    re.compile(r"\bapt(-get)?\s+install\b"),
    re.compile(r"\bbrew\s+install\b"),
    re.compile(r"\bcurl\b.*\|\s*(ba)?sh\b"),
    re.compile(r"\bwget\b.*\|\s*(ba)?sh\b"),
    re.compile(r"\bwallet\b"),
    re.compile(r"\bsend\b.*\b(eth|btc|usdc|sol)\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\b"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bsystemctl\b"),
]

# Tools that touch the filesystem with an explicit path.
_PATH_INPUT_KEYS = ("file_path", "path", "notebook_path")


def _resolve(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def _inside_cwd(path: str, cwd: str) -> bool:
    base = _resolve(cwd)
    target = _resolve(path)
    return target == base or target.startswith(base + os.sep)


def is_risky(tool_name: str, tool_input: dict, cwd: str) -> bool:
    """Risky: push/deploy/rm/ssh/install/out-of-cwd/wallet/network."""
    command = tool_input.get("command")
    if isinstance(command, str):
        lowered = command.lower()
        if any(p.search(lowered) for p in _RISKY_COMMAND_PATTERNS):
            return True

    for key in _PATH_INPUT_KEYS:
        path = tool_input.get(key)
        if isinstance(path, str) and path:
            if not _inside_cwd(path, cwd):
                return True

    return False
```

- [ ] **Step 4: Run test to verify it passes**

Command: `python -m pytest tests/test_approvals.py -q`
Expected: all `test_is_risky_true` / `test_is_risky_false` params PASS (green).

- [ ] **Step 5: Commit**

```bash
git add src/voice_bridge/approvals.py tests/test_approvals.py
git commit -m "feat(approvals): risk classification for tool calls

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 6: Write the failing test for `parse_yes_no`**

Append to `tests/test_approvals.py`:

```python
@pytest.mark.parametrize(
    "text",
    ["taip", "Taip!", "  jo ", "davai", "gerai", "ok", "okay", "yes", "yep", "y", "sure", "varom"],
)
def test_parse_yes_no_true(text):
    assert parse_yes_no(text) is True


@pytest.mark.parametrize(
    "text",
    ["ne", "Ne.", "stop", "no", "nope", "n", "atšauk", "neleisk"],
)
def test_parse_yes_no_false(text):
    assert parse_yes_no(text) is False


@pytest.mark.parametrize(
    "text",
    ["", "   ", "gal but", "what do you mean", "kažkas neaiškaus", "run the tests first"],
)
def test_parse_yes_no_none(text):
    assert parse_yes_no(text) is None
```

- [ ] **Step 7: Run test to verify it fails**

Command: `python -m pytest tests/test_approvals.py -k parse_yes_no -q`
Expected: `AttributeError`/`TypeError` from `parse_yes_no` (current impl absent) — actually `parse_yes_no` is imported but undefined → `ImportError` is avoided since import is at top; expected `NameError`/`TypeError: parse_yes_no` not defined → red. (If import fails first, fix the import; the function is what's missing.)

- [ ] **Step 8: Write minimal implementation of `parse_yes_no`**

Append to `src/voice_bridge/approvals.py`:

```python
# --- Yes/No parsing (Lithuanian + English) --------------------------------

_YES_WORDS = {
    "taip", "jo", "davai", "gerai", "ok", "okay", "yes", "yep", "yeah",
    "y", "sure", "varom", "leisk", "tikrai", "aha", "go",
}
_NO_WORDS = {
    "ne", "stop", "no", "nope", "n", "atšauk", "atsauk", "neleisk",
    "nereikia", "cancel", "neik",
}

_TOKEN_RE = re.compile(r"[a-ząčęėįšųūž]+", re.IGNORECASE)


def parse_yes_no(text: str) -> bool | None:
    """Return True for yes, False for no, None if undecidable. lt + en."""
    if not text:
        return None
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return None
    saw_yes = any(t in _YES_WORDS for t in tokens)
    saw_no = any(t in _NO_WORDS for t in tokens)
    if saw_no and not saw_yes:
        return False
    if saw_yes and not saw_no:
        return True
    return None
```

- [ ] **Step 9: Run test to verify it passes**

Command: `python -m pytest tests/test_approvals.py -k parse_yes_no -q`
Expected: all `parse_yes_no` params PASS (green).

- [ ] **Step 10: Commit**

```bash
git add src/voice_bridge/approvals.py tests/test_approvals.py
git commit -m "feat(approvals): lt+en yes/no parsing

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 11: Write the failing test for `ApprovalManager` approve / deny / timeout / pending**

Append to `tests/test_approvals.py`:

```python
@pytest.mark.asyncio
async def test_approval_manager_approve():
    sent: list[tuple[str, str]] = []

    async def send_question(project: str, text: str) -> int:
        sent.append((project, text))
        return 42

    mgr = ApprovalManager(send_question, timeout=5)

    async def approve_soon():
        await asyncio.sleep(0)
        assert mgr.has_pending(42) is True
        assert mgr.resolve(42, True) is True

    approver = asyncio.create_task(approve_soon())
    result = await mgr.request("qwing", "Bash", {"command": "git push"})
    await approver

    assert result is True
    assert sent and sent[0][0] == "qwing"
    assert "git push" in sent[0][1]
    assert mgr.has_pending(42) is False


@pytest.mark.asyncio
async def test_approval_manager_deny():
    async def send_question(project: str, text: str) -> int:
        return 7

    mgr = ApprovalManager(send_question, timeout=5)

    async def deny_soon():
        await asyncio.sleep(0)
        assert mgr.resolve(7, False) is True

    asyncio.create_task(deny_soon())
    result = await mgr.request("qwing", "Bash", {"command": "rm -rf x"})
    assert result is False


@pytest.mark.asyncio
async def test_approval_manager_timeout_denies():
    async def send_question(project: str, text: str) -> int:
        return 99

    mgr = ApprovalManager(send_question, timeout=0.05)
    result = await mgr.request("qwing", "Bash", {"command": "git push"})
    assert result is False
    assert mgr.has_pending(99) is False


@pytest.mark.asyncio
async def test_resolve_unknown_returns_false():
    async def send_question(project: str, text: str) -> int:
        return 1

    mgr = ApprovalManager(send_question, timeout=5)
    assert mgr.resolve(123456, True) is False
    assert mgr.has_pending(123456) is False
```

- [ ] **Step 12: Run test to verify it fails**

Command: `python -m pytest tests/test_approvals.py -k approval_manager -q`
Expected: `TypeError: ApprovalManager() takes no arguments` or `AttributeError: ... has no attribute 'request'` (class is a bare skeleton / undefined) → red.

- [ ] **Step 13: Write minimal implementation of `ApprovalManager`**

Append to `src/voice_bridge/approvals.py`:

```python
# --- Pending-approval manager ---------------------------------------------


def _format_question(project: str, tool_name: str, tool_input: dict) -> str:
    command = tool_input.get("command")
    if isinstance(command, str) and command.strip():
        action = command.strip()
    else:
        path = ""
        for key in _PATH_INPUT_KEYS:
            value = tool_input.get(key)
            if isinstance(value, str) and value:
                path = value
                break
        action = f"{tool_name} {path}".strip() if path else tool_name
    return f"{project} wants to run: {action}. Allow?"


class ApprovalManager:
    """Holds asyncio futures keyed by the question message_id; timeout -> deny."""

    def __init__(
        self,
        send_question: Callable[[str, str], Awaitable[int]],
        timeout: int,
    ) -> None:
        self._send_question = send_question
        self._timeout = timeout
        self._pending: dict[int, asyncio.Future[bool]] = {}

    async def request(self, project: str, tool_name: str, tool_input: dict) -> bool:
        text = _format_question(project, tool_name, tool_input)
        message_id = await self._send_question(project, text)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._pending[message_id] = future
        try:
            return await asyncio.wait_for(future, timeout=self._timeout)
        except asyncio.TimeoutError:
            return False
        finally:
            self._pending.pop(message_id, None)

    def resolve(self, message_id: int, approved: bool) -> bool:
        future = self._pending.get(message_id)
        if future is None or future.done():
            return False
        future.set_result(approved)
        return True

    def has_pending(self, message_id: int) -> bool:
        future = self._pending.get(message_id)
        return future is not None and not future.done()
```

- [ ] **Step 14: Run test to verify it passes**

Command: `python -m pytest tests/test_approvals.py -k approval_manager -q`
Expected: `test_approval_manager_approve`, `_deny`, `_timeout_denies`, `test_resolve_unknown_returns_false` PASS (green).

- [ ] **Step 15: Commit**

```bash
git add src/voice_bridge/approvals.py tests/test_approvals.py
git commit -m "feat(approvals): ApprovalManager with future + timeout-denies

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 16: Write the failing test for `make_can_use_tool` (full / ask / safe routing)**

Append to `tests/test_approvals.py`. The SDK permission types are stubbed via a module-level monkeypatch so tests need no live SDK:

```python
class _FakeManager:
    """Records request() calls; returns a preset decision."""

    def __init__(self, decision: bool):
        self.decision = decision
        self.calls: list[tuple[str, str, dict]] = []

    async def request(self, project: str, tool_name: str, tool_input: dict) -> bool:
        self.calls.append((project, tool_name, tool_input))
        return self.decision


def _decision_kind(result) -> str:
    return type(result).__name__


@pytest.mark.asyncio
async def test_can_use_tool_full_allows_without_asking():
    mgr = _FakeManager(decision=False)
    fn = make_can_use_tool(_proj(autonomy="full"), _cfg(), mgr)
    result = await fn("Bash", {"command": "git push origin master"}, None)
    assert _decision_kind(result) == "PermissionResultAllow"
    assert mgr.calls == []


@pytest.mark.asyncio
async def test_can_use_tool_ask_requests_even_safe():
    mgr = _FakeManager(decision=True)
    fn = make_can_use_tool(_proj(autonomy="ask"), _cfg(), mgr)
    result = await fn("Read", {"file_path": f"{CWD}/a.py"}, None)
    assert _decision_kind(result) == "PermissionResultAllow"
    assert len(mgr.calls) == 1


@pytest.mark.asyncio
async def test_can_use_tool_ask_deny_maps_to_deny():
    mgr = _FakeManager(decision=False)
    fn = make_can_use_tool(_proj(autonomy="ask"), _cfg(), mgr)
    result = await fn("Read", {"file_path": f"{CWD}/a.py"}, None)
    assert _decision_kind(result) == "PermissionResultDeny"
    assert len(mgr.calls) == 1


@pytest.mark.asyncio
async def test_can_use_tool_safe_allows_safe_without_asking():
    mgr = _FakeManager(decision=False)
    fn = make_can_use_tool(_proj(autonomy="safe"), _cfg(), mgr)
    result = await fn("Read", {"file_path": f"{CWD}/a.py"}, None)
    assert _decision_kind(result) == "PermissionResultAllow"
    assert mgr.calls == []


@pytest.mark.asyncio
async def test_can_use_tool_safe_asks_for_risky():
    mgr = _FakeManager(decision=True)
    fn = make_can_use_tool(_proj(autonomy="safe"), _cfg(), mgr)
    result = await fn("Bash", {"command": "git push origin master"}, None)
    assert _decision_kind(result) == "PermissionResultAllow"
    assert len(mgr.calls) == 1
    assert mgr.calls[0][0] == "qwing"


@pytest.mark.asyncio
async def test_can_use_tool_uses_project_autonomy_over_global():
    # global cfg is full, but project override is safe -> risky must be asked
    mgr = _FakeManager(decision=False)
    fn = make_can_use_tool(_proj(autonomy="safe"), _cfg(autonomy_mode="full"), mgr)
    result = await fn("Bash", {"command": "rm -rf build"}, None)
    assert _decision_kind(result) == "PermissionResultDeny"
    assert len(mgr.calls) == 1
```

- [ ] **Step 17: Run test to verify it fails**

Command: `python -m pytest tests/test_approvals.py -k can_use_tool -q`
Expected: `ModuleNotFoundError: No module named 'claude_agent_sdk'` (the lazy import resolves at call time) or `AttributeError` from incomplete `make_can_use_tool` → red. If the SDK is genuinely unavailable in the test env, the next step's implementation imports the permission types at call time; install `claude-agent-sdk` (already in `deps`) so the import resolves — tests still make no network calls.

- [ ] **Step 18: Write minimal implementation of `make_can_use_tool`**

Append to `src/voice_bridge/approvals.py`:

```python
# --- canUseTool factory ---------------------------------------------------


def make_can_use_tool(
    project: ProjectConfig,
    cfg: Config,
    manager: "ApprovalManager",
) -> Callable:
    """Build an SDK canUseTool callback honoring effective_autonomy.

    full -> allow all (no question); ask -> request all; safe -> request risky.
    """
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    mode = effective_autonomy(project, cfg)

    async def can_use_tool(tool_name: str, tool_input: dict, context):
        if mode == "full":
            return PermissionResultAllow()

        if mode == "safe" and not is_risky(tool_name, tool_input, project.cwd):
            return PermissionResultAllow()

        approved = await manager.request(project.name, tool_name, tool_input)
        if approved:
            return PermissionResultAllow()
        return PermissionResultDeny(message="no answer, skipped")

    return can_use_tool
```

- [ ] **Step 19: Run test to verify it passes**

Command: `python -m pytest tests/test_approvals.py -q`
Expected: the full file PASSES — `is_risky`, `parse_yes_no`, `approval_manager`, and all six `can_use_tool` cases green.

- [ ] **Step 20: Commit**

```bash
git add src/voice_bridge/approvals.py tests/test_approvals.py
git commit -m "feat(approvals): make_can_use_tool honoring full/safe/ask

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: notify_user SDK MCP server

In-process Claude Agent SDK MCP server exposing a single `notify_user(summary, detail)` tool so a running agent can ping the user mid-turn. The tool handler forwards its arguments to an injected async `on_notify(summary, detail)` callback (the bridge wires this to the outbound path) and returns a confirmation content block.

**Files:**
- Create: `src/voice_bridge/notify_tool.py`
- Test: `tests/test_notify_tool.py`
- Modify: none

**Interfaces:**

Consumes (from the `claude-agent-sdk` dependency — exact SDK signatures):
- `tool(name: str, description: str, input_schema: type | dict[str, Any], annotations=None) -> Callable[[Callable[[Any], Awaitable[dict[str, Any]]]], SdkMcpTool[Any]]` — decorator; handler is `async def handler(args: dict) -> dict` returning `{"content": [{"type": "text", "text": str}]}`.
- `create_sdk_mcp_server(name: str, version: str = "1.0.0", tools: list[SdkMcpTool[Any]] | None = None) -> McpSdkServerConfig`.

Consumes (from the canonical contract — provided to me, used here):
- `on_notify: Callable[[str, str], Awaitable[None]]` — the async callback the bridge passes in (it routes the notification to the Telegram/TTS outbound path).

Produces (exact signatures this module exposes, per the contract):
- `def make_notify_server(on_notify: Callable[[str, str], Awaitable[None]])` — returns an SDK MCP server (the value of `create_sdk_mcp_server`); registers tool `notify_user(summary: str, detail: str = '')` which calls `on_notify(summary, detail)`.
- `NOTIFY_TOOL_NAME = 'mcp__bridge__notify_user'` — the fully-qualified tool name (server `bridge` + tool `notify_user`), for use in `allowed_tools`.

Design notes (consequences of the contract / global constraints):
- Server name MUST be `bridge` and tool name MUST be `notify_user` so the SDK-derived fully-qualified name equals `NOTIFY_TOOL_NAME` (`mcp__<server>__<tool>`).
- The handler reads `args["summary"]` (required) and `args.get("detail", "")` (optional), `await`s `on_notify`, and returns a text content dict. It does no blocking work, satisfying the single-event-loop constraint.
- Tests mock the SDK (`tool`, `create_sdk_mcp_server`) so they run with no network, no API keys, and without `claude-agent-sdk` installed; the handler is invoked directly and the callback assertion proves the wiring.

TDD steps:

- [ ] **Step 1: Write the failing test.** Create `tests/test_notify_tool.py` with the complete code below. It stubs the `claude_agent_sdk` module so the SDK need not be installed, captures the registered handler via a fake `tool` decorator, then invokes the handler directly and asserts the callback fired with the right args (including the `detail` default).

```python
import sys
import types

import pytest

# --- Install a fake `claude_agent_sdk` module BEFORE importing notify_tool. ---
# `tool` records each decorated handler so the test can invoke it directly.
# `create_sdk_mcp_server` records its kwargs and returns a sentinel object.

_REGISTERED: dict[str, dict] = {}


def _fake_tool(name, description, input_schema, annotations=None):
    def _decorator(handler):
        _REGISTERED[name] = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
            "annotations": annotations,
            "handler": handler,
        }
        return handler

    return _decorator


_CREATED: dict[str, object] = {}


def _fake_create_sdk_mcp_server(name, version="1.0.0", tools=None):
    server = object()
    _CREATED["last"] = {
        "name": name,
        "version": version,
        "tools": tools,
        "server": server,
    }
    return server


_fake_sdk = types.ModuleType("claude_agent_sdk")
_fake_sdk.tool = _fake_tool
_fake_sdk.create_sdk_mcp_server = _fake_create_sdk_mcp_server
sys.modules["claude_agent_sdk"] = _fake_sdk

from voice_bridge import notify_tool  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_registry():
    _REGISTERED.clear()
    _CREATED.clear()
    yield
    _REGISTERED.clear()
    _CREATED.clear()


def test_notify_tool_name_is_fully_qualified():
    assert notify_tool.NOTIFY_TOOL_NAME == "mcp__bridge__notify_user"


def test_make_notify_server_registers_notify_user_on_bridge_server():
    async def on_notify(summary, detail):
        pass

    server = notify_tool.make_notify_server(on_notify)

    # Server identity matches what create_sdk_mcp_server returned.
    assert server is _CREATED["last"]["server"]
    # Server name must be "bridge" so the FQ tool name resolves correctly.
    assert _CREATED["last"]["name"] == "bridge"
    # The notify_user tool was registered with summary+detail in its schema.
    assert "notify_user" in _REGISTERED
    schema = _REGISTERED["notify_user"]["input_schema"]
    assert schema == {"summary": str, "detail": str}
    # The single registered tool was passed to create_sdk_mcp_server.
    assert _CREATED["last"]["tools"] == [_REGISTERED["notify_user"]["handler"]]


@pytest.mark.asyncio
async def test_handler_invokes_callback_with_summary_and_detail():
    calls = []

    async def on_notify(summary, detail):
        calls.append((summary, detail))

    notify_tool.make_notify_server(on_notify)
    handler = _REGISTERED["notify_user"]["handler"]

    result = await handler({"summary": "build done", "detail": "12 files changed"})

    assert calls == [("build done", "12 files changed")]
    assert result["content"][0]["type"] == "text"
    assert isinstance(result["content"][0]["text"], str)


@pytest.mark.asyncio
async def test_handler_defaults_detail_to_empty_string():
    calls = []

    async def on_notify(summary, detail):
        calls.append((summary, detail))

    notify_tool.make_notify_server(on_notify)
    handler = _REGISTERED["notify_user"]["handler"]

    await handler({"summary": "tests passed"})

    assert calls == [("tests passed", "")]
```

- [ ] **Step 2: Run test to verify it fails.** Run:
  ```bash
  python -m pytest tests/test_notify_tool.py -q
  ```
  Expected failure: collection/import error `ModuleNotFoundError: No module named 'voice_bridge.notify_tool'` (or `AttributeError: module 'voice_bridge.notify_tool' has no attribute 'NOTIFY_TOOL_NAME'` if the package exists but the file does not), because `src/voice_bridge/notify_tool.py` has not been created yet.

- [ ] **Step 3: Write minimal implementation.** Create `src/voice_bridge/notify_tool.py` with the complete code below:

```python
"""In-process Claude Agent SDK MCP server exposing ``notify_user``.

This lets a running agent ping the user mid-turn. The single tool forwards its
arguments to an injected async ``on_notify(summary, detail)`` callback, which the
bridge wires to the Telegram + TTS outbound path. No blocking work happens here,
so the single event loop is never stalled.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from claude_agent_sdk import create_sdk_mcp_server, tool

# Server name "bridge" + tool name "notify_user" => SDK fully-qualified name.
NOTIFY_TOOL_NAME = "mcp__bridge__notify_user"


def make_notify_server(on_notify: Callable[[str, str], Awaitable[None]]):
    """Build the in-process MCP server exposing ``notify_user``.

    Args:
        on_notify: async callback invoked as ``on_notify(summary, detail)`` each
            time the agent calls the tool.

    Returns:
        The SDK MCP server config (from ``create_sdk_mcp_server``) for use in
        ``ClaudeAgentOptions.mcp_servers``.
    """

    @tool(
        "notify_user",
        "Send the user a short status update mid-turn. "
        "summary is a one-line spoken-friendly message; detail is optional "
        "longer context shown in text.",
        {"summary": str, "detail": str},
    )
    async def notify_user(args: dict) -> dict:
        summary = args["summary"]
        detail = args.get("detail", "")
        await on_notify(summary, detail)
        return {"content": [{"type": "text", "text": "Notification sent to user."}]}

    return create_sdk_mcp_server(
        name="bridge",
        version="1.0.0",
        tools=[notify_user],
    )
```

- [ ] **Step 4: Run test to verify it passes.** Run:
  ```bash
  python -m pytest tests/test_notify_tool.py -q
  ```
  Expected: `5 passed` (the name test, the registration test, and the three handler tests), no warnings about unmocked SDK.

- [ ] **Step 5: Commit.** Run:
  ```bash
  git add src/voice_bridge/notify_tool.py tests/test_notify_tool.py
  git commit -m "feat(notify_tool): in-process SDK MCP server for notify_user

make_notify_server registers a notify_user(summary, detail) tool on the
'bridge' SDK MCP server; the handler forwards to an async on_notify
callback. NOTIFY_TOOL_NAME exposes the fully-qualified mcp__bridge__notify_user
name for allowed_tools. Tests stub claude_agent_sdk and invoke the handler
directly, asserting the callback fires with summary and the defaulted detail.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 8: SessionManager

**Files:**
- Create: `src/voice_bridge/sessions.py`
- Test: `tests/test_sessions.py`

**Interfaces:**

Consumes (from earlier tasks — use these exact signatures):
- `config.py` — `@dataclass Config(telegram_bot_token, telegram_allowed_user_id, anthropic_api_key, openai_api_key, tts_backend, tts_voice, piper_voice_path, whisper_model, autonomy_mode, approval_timeout, db_path)`; `@dataclass ProjectConfig(name:str, cwd:str, enabled:bool=True, autonomy:str|None=None, voice:str|None=None, model:str|None=None, system_prompt_extra:str='')`; `def effective_autonomy(project: ProjectConfig, cfg: Config) -> str`.
- `routing.py` — `class Store` with `async def get_session_id(self, project:str) -> str | None`, `async def set_session_id(self, project:str, session_id:str) -> None`, `async def is_enabled(self, project:str) -> bool`, `async def set_enabled(self, project:str, enabled:bool) -> None`.
- `approvals.py` — `class ApprovalManager`; `def make_can_use_tool(project: ProjectConfig, cfg: Config, manager: ApprovalManager) -> Callable` (honors `effective_autonomy`: full→allow all; ask→request all; safe→request only `is_risky`).
- `notify_tool.py` — `make_notify_server(on_notify)` returns an sdk mcp server object; `NOTIFY_TOOL_NAME = 'mcp__bridge__notify_user'`.
- `types` — `@dataclass Outbound(project:str, text:str, spoken:str)`.

Produces (exact signatures this task exposes):
- `class SessionManager: def __init__(self, projects:list[ProjectConfig], cfg:Config, store:Store, on_outbound:Callable[[Outbound],Awaitable[None]], approvals:ApprovalManager, notify_server)`
- `async def start_all(self) -> None  # start enabled projects`
- `async def deliver(self, project:str, text:str) -> None  # enqueue a user turn`
- `async def set_enabled(self, project:str, enabled:bool) -> None  # start/stop session`
- `async def set_mode(self, project:str, mode:str) -> None`
- `async def stop_all(self) -> None`
- `def project(self, name:str) -> ProjectConfig | None`

Design notes (constraints honored): single event loop, all SDK work async; `ClaudeSDKClient` is mocked in every test (a fake async-iterating client) so tests run with no network/secrets; streaming-input is an `asyncio.Queue`-backed async generator; `session_id` captured from the SDK `system`/`init` message and persisted via `store.set_session_id`; resume passes the stored `session_id` into `ClaudeSDKClientOptions`; permission wiring is `bypassPermissions` for `full`, otherwise `canUseTool` from `make_can_use_tool`.

---

#### TDD steps

- [ ] **Step 1: Write the failing test** — create `tests/test_sessions.py` with the fakes and the first test. Complete code:

```python
import asyncio
import sys
import types as _types
from dataclasses import dataclass, field

import pytest

# ---- Stub the claude_agent_sdk module BEFORE importing sessions ----
# sessions.py imports ClaudeSDKClient, ClaudeAgentOptions from claude_agent_sdk.
# We install a fake module so no real SDK / network is needed.

_fake_sdk = _types.ModuleType("claude_agent_sdk")


class _AssistantTextBlock:
    def __init__(self, text):
        self.text = text


class AssistantMessage:
    def __init__(self, blocks):
        self.content = blocks


class SystemMessage:
    def __init__(self, subtype, data):
        self.subtype = subtype
        self.data = data


class ResultMessage:
    def __init__(self):
        pass


class ClaudeAgentOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeClaudeSDKClient:
    """Records construction + drives a scripted response stream per turn."""

    instances = []

    def __init__(self, options=None):
        self.options = options
        self.connected = False
        self.disconnected = False
        self.queries = []          # list of prompts passed to query()
        # scripted: list of lists-of-messages, one inner list per turn
        self.scripted_turns = []
        self._turn_index = 0
        FakeClaudeSDKClient.instances.append(self)

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.disconnected = True

    async def query(self, prompt, session_id=None):
        self.queries.append(prompt)

    async def receive_response(self):
        if self._turn_index < len(self.scripted_turns):
            turn = self.scripted_turns[self._turn_index]
        else:
            turn = [ResultMessage()]
        self._turn_index += 1
        for msg in turn:
            await asyncio.sleep(0)
            yield msg


_fake_sdk.ClaudeSDKClient = FakeClaudeSDKClient
_fake_sdk.ClaudeAgentOptions = ClaudeAgentOptions
_fake_sdk.AssistantMessage = AssistantMessage
_fake_sdk.SystemMessage = SystemMessage
_fake_sdk.ResultMessage = ResultMessage
_fake_sdk.TextBlock = _AssistantTextBlock
sys.modules["claude_agent_sdk"] = _fake_sdk

# Now safe to import the module under test.
from voice_bridge.sessions import SessionManager  # noqa: E402
from voice_bridge.config import Config, ProjectConfig  # noqa: E402
from voice_bridge.types import Outbound  # noqa: E402


# ---- Lightweight fakes for collaborators ----

class FakeStore:
    def __init__(self, enabled=None, session_ids=None):
        self._enabled = enabled or {}
        self._session_ids = session_ids or {}
        self.set_session_calls = []

    async def is_enabled(self, project):
        return self._enabled.get(project, True)

    async def set_enabled(self, project, enabled):
        self._enabled[project] = enabled

    async def get_session_id(self, project):
        return self._session_ids.get(project)

    async def set_session_id(self, project, session_id):
        self._session_ids[project] = session_id
        self.set_session_calls.append((project, session_id))


class FakeApprovals:
    pass


def make_cfg(**over):
    base = dict(
        telegram_bot_token="t",
        telegram_allowed_user_id=1,
        anthropic_api_key="a",
        openai_api_key="o",
        tts_backend="openai",
        tts_voice="nova",
        piper_voice_path="/x.onnx",
        whisper_model="large-v3",
        autonomy_mode="safe",
        approval_timeout=300,
        db_path=":memory:",
    )
    base.update(over)
    return Config(**base)


def make_project(name="qwing", **over):
    base = dict(name=name, cwd="/tmp/qwing", enabled=True)
    base.update(over)
    return ProjectConfig(**base)


@pytest.fixture(autouse=True)
def _reset_instances():
    FakeClaudeSDKClient.instances = []
    yield
    FakeClaudeSDKClient.instances = []


@pytest.mark.asyncio
async def test_start_all_starts_only_enabled_projects():
    projects = [make_project("qwing"), make_project("other", enabled=False)]
    store = FakeStore(enabled={"qwing": True, "other": False})
    outbound = []

    async def on_outbound(o):
        outbound.append(o)

    sm = SessionManager(projects, make_cfg(), store, on_outbound,
                        FakeApprovals(), notify_server=object())
    await sm.start_all()

    assert sm.is_running("qwing") is True
    assert sm.is_running("other") is False
    assert len(FakeClaudeSDKClient.instances) == 1
    assert FakeClaudeSDKClient.instances[0].connected is True

    await sm.stop_all()
```

- [ ] **Step 1: Run test to verify it fails**
  - Command: `cd /home/home/Projects/claude-voice-bridge && python -m pytest tests/test_sessions.py -q`
  - Expected failure: `ModuleNotFoundError: No module named 'voice_bridge.sessions'` (or `ImportError: cannot import name 'SessionManager'`).

- [ ] **Step 2: Write minimal implementation** — create `src/voice_bridge/sessions.py`. Complete code:

```python
"""Per-project long-lived ClaudeSDKClient in streaming-input mode."""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    SystemMessage,
    TextBlock,
)

from .approvals import ApprovalManager, make_can_use_tool
from .config import Config, ProjectConfig, effective_autonomy
from .notify_tool import NOTIFY_TOOL_NAME
from .routing import Store
from .types import Outbound

_SYSTEM_PROMPT_SPLIT = (
    "When sending a user-facing message, put one short spoken-friendly line "
    "(status or question, no code, no paths, no commands) first, then a line "
    "that is exactly '---', then all technical detail (code, diffs, paths, "
    "commands) below it."
)


class _Session:
    """Holds the live state for one project's ClaudeSDKClient."""

    def __init__(self, project: ProjectConfig):
        self.project = project
        self.client: ClaudeSDKClient | None = None
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.mode: str | None = None


class SessionManager:
    def __init__(
        self,
        projects: list[ProjectConfig],
        cfg: Config,
        store: Store,
        on_outbound: Callable[[Outbound], Awaitable[None]],
        approvals: ApprovalManager,
        notify_server,
    ):
        self._projects = {p.name: p for p in projects}
        self._cfg = cfg
        self._store = store
        self._on_outbound = on_outbound
        self._approvals = approvals
        self._notify_server = notify_server
        self._sessions: dict[str, _Session] = {}

    def project(self, name: str) -> ProjectConfig | None:
        return self._projects.get(name)

    def is_running(self, name: str) -> bool:
        return name in self._sessions

    async def start_all(self) -> None:
        for name in self._projects:
            if await self._store.is_enabled(name):
                await self._start(name)

    async def deliver(self, project: str, text: str) -> None:
        sess = self._sessions.get(project)
        if sess is None:
            return
        await sess.queue.put(text)

    async def set_enabled(self, project: str, enabled: bool) -> None:
        if project not in self._projects:
            return
        await self._store.set_enabled(project, enabled)
        if enabled:
            if project not in self._sessions:
                await self._start(project)
        else:
            await self._stop(project)

    async def set_mode(self, project: str, mode: str) -> None:
        if project not in self._projects:
            return
        self._projects[project].autonomy = mode
        sess = self._sessions.get(project)
        if sess is not None:
            await self._stop(project)
            await self._start(project)

    async def stop_all(self) -> None:
        for name in list(self._sessions):
            await self._stop(name)

    async def _build_options(self, project: ProjectConfig) -> ClaudeAgentOptions:
        mode = effective_autonomy(project, self._cfg)
        kwargs: dict = {
            "cwd": project.cwd,
            "mcp_servers": {"bridge": self._notify_server},
            "allowed_tools": [NOTIFY_TOOL_NAME],
        }
        if project.model:
            kwargs["model"] = project.model
        extra = _SYSTEM_PROMPT_SPLIT
        if project.system_prompt_extra:
            extra = extra + "\n" + project.system_prompt_extra
        kwargs["system_prompt"] = {"type": "preset", "preset": "claude_code",
                                   "append": extra}
        if mode == "full":
            kwargs["permission_mode"] = "bypassPermissions"
        else:
            kwargs["can_use_tool"] = make_can_use_tool(
                project, self._cfg, self._approvals
            )
        resume = await self._store.get_session_id(project.name)
        if resume:
            kwargs["resume"] = resume
        return ClaudeAgentOptions(**kwargs)

    async def _start(self, name: str) -> None:
        if name in self._sessions:
            return
        project = self._projects[name]
        sess = _Session(project)
        options = await self._build_options(project)
        sess.client = ClaudeSDKClient(options=options)
        await sess.client.connect()
        self._sessions[name] = sess
        sess.task = asyncio.create_task(self._run_loop(sess))

    async def _stop(self, name: str) -> None:
        sess = self._sessions.pop(name, None)
        if sess is None:
            return
        if sess.task is not None:
            sess.task.cancel()
            try:
                await sess.task
            except asyncio.CancelledError:
                pass
        if sess.client is not None:
            await sess.client.disconnect()

    async def _run_loop(self, sess: _Session) -> None:
        while True:
            prompt = await sess.queue.get()
            await sess.client.query(prompt)
            parts: list[str] = []
            async for message in sess.client.receive_response():
                if isinstance(message, SystemMessage):
                    if message.subtype == "init":
                        sid = message.data.get("session_id")
                        if sid:
                            await self._store.set_session_id(
                                sess.project.name, sid
                            )
                    continue
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            parts.append(block.text)
            text = "\n".join(p for p in parts if p).strip()
            if text:
                await self._on_outbound(
                    Outbound(project=sess.project.name, text=text, spoken="")
                )
```

- [ ] **Step 2: Run test to verify it passes**
  - Command: `cd /home/home/Projects/claude-voice-bridge && python -m pytest tests/test_sessions.py -q`
  - Expected: `1 passed`.

- [ ] **Step 3: Commit**
  - `cd /home/home/Projects/claude-voice-bridge && git add src/voice_bridge/sessions.py tests/test_sessions.py`
  - `git commit -m "feat(sessions): SessionManager start_all/stop_all lifecycle for enabled projects

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

- [ ] **Step 4: Write the failing test** — deliver enqueues a turn, the loop drains it, captures `session_id`, and emits an Outbound with assistant text. Append to `tests/test_sessions.py`:

```python
@pytest.mark.asyncio
async def test_deliver_drains_turn_captures_session_id_and_emits_outbound():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound = []

    async def on_outbound(o):
        outbound.append(o)

    sm = SessionManager([project], make_cfg(), store, on_outbound,
                        FakeApprovals(), notify_server=object())
    await sm.start_all()

    client = FakeClaudeSDKClient.instances[0]
    client.scripted_turns = [[
        SystemMessage("init", {"session_id": "sess-123"}),
        AssistantMessage([_AssistantTextBlock("Done.\n---\ndiff here")]),
        ResultMessage(),
    ]]

    await sm.deliver("qwing", "build the thing")

    # Wait for the receive loop to process the queued turn.
    for _ in range(200):
        if outbound:
            break
        await asyncio.sleep(0.01)

    assert client.queries == ["build the thing"]
    assert store._session_ids["qwing"] == "sess-123"
    assert len(outbound) == 1
    assert isinstance(outbound[0], Outbound)
    assert outbound[0].project == "qwing"
    assert "Done." in outbound[0].text
    assert "diff here" in outbound[0].text

    await sm.stop_all()


@pytest.mark.asyncio
async def test_deliver_to_unstarted_project_is_noop():
    project = make_project("qwing", enabled=False)
    store = FakeStore(enabled={"qwing": False})

    async def on_outbound(o):
        pass

    sm = SessionManager([project], make_cfg(), store, on_outbound,
                        FakeApprovals(), notify_server=object())
    await sm.start_all()
    # Not started, so no client and deliver must not raise.
    await sm.deliver("qwing", "hello")
    assert FakeClaudeSDKClient.instances == []
    await sm.stop_all()
```

- [ ] **Step 4: Run test to verify it fails** (write the test before the loop behavior is proven)
  - Command: `cd /home/home/Projects/claude-voice-bridge && python -m pytest tests/test_sessions.py::test_deliver_drains_turn_captures_session_id_and_emits_outbound -q`
  - Expected: PASS already if Step 2 impl is correct. If the receive-loop / session-id capture is wrong, expected failure: `AssertionError: assert 'sess-123' == store._session_ids['qwing']` (KeyError) or `assert len(outbound) == 1`. (The impl from Step 2 already satisfies this; running confirms the loop behavior — if it does not pass, fix `_run_loop` until it does.)

- [ ] **Step 4: Run test to verify it passes**
  - Command: `cd /home/home/Projects/claude-voice-bridge && python -m pytest tests/test_sessions.py -q`
  - Expected: `3 passed`.

- [ ] **Step 5: Commit**
  - `cd /home/home/Projects/claude-voice-bridge && git add tests/test_sessions.py`
  - `git commit -m "test(sessions): cover deliver streaming-input drain, session_id capture, outbound emit

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

- [ ] **Step 6: Write the failing test** — `set_enabled(False)` stops/disconnects the session and persists; `set_enabled(True)` restarts it. Append to `tests/test_sessions.py`:

```python
@pytest.mark.asyncio
async def test_set_enabled_false_stops_and_persists_then_true_restarts():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = SessionManager([project], make_cfg(), store, on_outbound,
                        FakeApprovals(), notify_server=object())
    await sm.start_all()
    assert sm.is_running("qwing") is True
    first_client = FakeClaudeSDKClient.instances[0]

    await sm.set_enabled("qwing", False)
    assert sm.is_running("qwing") is False
    assert first_client.disconnected is True
    assert store._enabled["qwing"] is False

    await sm.set_enabled("qwing", True)
    assert sm.is_running("qwing") is True
    assert store._enabled["qwing"] is True
    assert len(FakeClaudeSDKClient.instances) == 2

    await sm.stop_all()
```

- [ ] **Step 6: Run test to verify it fails**
  - Command: `cd /home/home/Projects/claude-voice-bridge && python -m pytest tests/test_sessions.py::test_set_enabled_false_stops_and_persists_then_true_restarts -q`
  - Expected: PASS with the Step 2 impl (set_enabled is implemented). If `_stop` does not cancel the task / disconnect, expected failure: `AssertionError: assert first_client.disconnected is True`. Confirm green; if red, fix `_stop`.

- [ ] **Step 6: Run test to verify it passes**
  - Command: `cd /home/home/Projects/claude-voice-bridge && python -m pytest tests/test_sessions.py -q`
  - Expected: `4 passed`.

- [ ] **Step 7: Commit**
  - `cd /home/home/Projects/claude-voice-bridge && git add tests/test_sessions.py`
  - `git commit -m "test(sessions): cover on/off lifecycle persists enabled and restarts session

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

- [ ] **Step 8: Write the failing test** — `full` mode wires `bypassPermissions` (no `can_use_tool`); `safe`/`ask` mode wires `can_use_tool` (no bypass); resume `session_id` is passed into options. Append to `tests/test_sessions.py`:

```python
@pytest.mark.asyncio
async def test_full_mode_uses_bypass_permissions():
    project = make_project("qwing", autonomy="full")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = SessionManager([project], make_cfg(), store, on_outbound,
                        FakeApprovals(), notify_server=object())
    await sm.start_all()

    opts = FakeClaudeSDKClient.instances[0].options
    assert opts.kwargs.get("permission_mode") == "bypassPermissions"
    assert "can_use_tool" not in opts.kwargs

    await sm.stop_all()


@pytest.mark.asyncio
async def test_safe_mode_uses_can_use_tool_not_bypass():
    project = make_project("qwing", autonomy="safe")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = SessionManager([project], make_cfg(), store, on_outbound,
                        FakeApprovals(), notify_server=object())
    await sm.start_all()

    opts = FakeClaudeSDKClient.instances[0].options
    assert "permission_mode" not in opts.kwargs
    assert callable(opts.kwargs.get("can_use_tool"))

    await sm.stop_all()


@pytest.mark.asyncio
async def test_resume_session_id_passed_to_options():
    project = make_project("qwing", autonomy="safe")
    store = FakeStore(enabled={"qwing": True},
                      session_ids={"qwing": "prev-sess-9"})

    async def on_outbound(o):
        pass

    sm = SessionManager([project], make_cfg(), store, on_outbound,
                        FakeApprovals(), notify_server=object())
    await sm.start_all()

    opts = FakeClaudeSDKClient.instances[0].options
    assert opts.kwargs.get("resume") == "prev-sess-9"

    await sm.stop_all()


@pytest.mark.asyncio
async def test_set_mode_rebuilds_session_with_new_permission_wiring():
    project = make_project("qwing", autonomy="safe")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = SessionManager([project], make_cfg(), store, on_outbound,
                        FakeApprovals(), notify_server=object())
    await sm.start_all()
    assert "can_use_tool" in FakeClaudeSDKClient.instances[0].options.kwargs

    await sm.set_mode("qwing", "full")

    assert len(FakeClaudeSDKClient.instances) == 2
    new_opts = FakeClaudeSDKClient.instances[1].options
    assert new_opts.kwargs.get("permission_mode") == "bypassPermissions"
    assert "can_use_tool" not in new_opts.kwargs

    await sm.stop_all()


@pytest.mark.asyncio
async def test_project_lookup_returns_config_or_none():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = SessionManager([project], make_cfg(), store, on_outbound,
                        FakeApprovals(), notify_server=object())
    assert sm.project("qwing") is project
    assert sm.project("nope") is None
```

- [ ] **Step 8: Run test to verify it fails**
  - Command: `cd /home/home/Projects/claude-voice-bridge && python -m pytest tests/test_sessions.py -q`
  - Expected: these pass if `make_can_use_tool`/`effective_autonomy` behave as the contract says. If `_build_options` permission wiring or `resume` passing is wrong, expected failure such as `AssertionError: assert 'bypassPermissions' == opts.kwargs.get('permission_mode')` or `assert 'prev-sess-9' == opts.kwargs.get('resume')`. Make `_build_options` satisfy all assertions.

- [ ] **Step 8: Run test to verify it passes**
  - Command: `cd /home/home/Projects/claude-voice-bridge && python -m pytest tests/test_sessions.py -q`
  - Expected: `9 passed`.

- [ ] **Step 9: Commit**
  - `cd /home/home/Projects/claude-voice-bridge && git add src/voice_bridge/sessions.py tests/test_sessions.py`
  - `git commit -m "feat(sessions): permission wiring per mode, resume by session_id, set_mode rebuild

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task 9: Telegram I/O + control panel

**Files:**
- Create: `src/voice_bridge/telegram_io.py`
- Test: `tests/test_telegram_io.py`
- (Implied dependency, may already exist from Task 1) Modify: none — `src/voice_bridge/config.py` (`Config` dataclass) is consumed, not modified.

**Interfaces:**

Consumes (from earlier tasks, used by exact signature):
- `config.py` → `@dataclass Config: telegram_bot_token:str, telegram_allowed_user_id:int, anthropic_api_key:str, openai_api_key:str, tts_backend:str, tts_voice:str, piper_voice_path:str, whisper_model:str, autonomy_mode:str, approval_timeout:int, db_path:str` — read `telegram_bot_token` and `telegram_allowed_user_id` for the whitelist filter.
- `on_user_message: Callable[[dict], Awaitable[None]]` provided by `bridge.py` (Task 12). The dict passed back is exactly `{'message_id':int, 'reply_to':int|None, 'text':str, 'is_voice':bool, 'audio':bytes|None}`.
- `Controls` protocol provided by `bridge.py` (Task 12): `async def toggle(project:str|None, on:bool)`, `async def set_mode(project:str, mode:str)`, `async def set_voice(project:str, voice:str)`, `async def set_engine(name:str)`, `def snapshot() -> list[dict]` where each dict is `{'project':str, 'enabled':bool, 'mode':str, 'voice':str, 'engine':str, 'last_active':bool}` for `/panel` and `/projects` rendering.

Produces (exact signatures exposed by this module):
- `class TelegramIO: def __init__(self, cfg:Config, on_user_message:Callable[[dict],Awaitable[None]], controls:'Controls')`
- `async def send_update(self, project:str, voice_label:str, text:str, voice_bytes:bytes|None) -> list[int]` — sends a text message then (if `voice_bytes`) a voice message; returns the list of `message_id`s sent.
- `async def send_question(self, project:str, text:str) -> int` — sends one text message; returns its `message_id` (used by `approvals.py` to key pending approvals).
- `async def run(self) -> None` — builds the `Application`, registers handlers, starts long polling.

Implementation notes baked into the steps below:
- A `User filter` (`telegram.ext.filters.User(user_id=cfg.telegram_allowed_user_id)`) gates every message handler; a separate guard inside the callback-query handler rejects non-whitelisted `callback_query.from_user.id`. Non-whitelisted inbound is dropped silently (spec §9 "ignored").
- Voice handler downloads the OGG/Opus bytes via `update.message.voice.get_file()` → `download_as_bytearray()` and sets `is_voice=True, audio=bytes(...)`; text handler sets `is_voice=False, audio=None`.
- `/panel` builds an `InlineKeyboardMarkup` from `controls.snapshot()`; `callback_query` data is encoded as `"<action>:<project>:<value>"` and dispatched to the matching `Controls` coroutine, then the panel is edited in place via `query.edit_message_reply_markup` / `edit_message_text`.

---

- [ ] **Step 1: Write the failing test (whitelist + message routing)**

Create `tests/test_telegram_io.py`:

```python
import asyncio
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from voice_bridge.config import Config
from voice_bridge.telegram_io import TelegramIO, build_panel_markup, parse_callback


def make_cfg(allowed_id=42):
    return Config(
        telegram_bot_token="TESTTOKEN",
        telegram_allowed_user_id=allowed_id,
        anthropic_api_key="ak",
        openai_api_key="ok",
        tts_backend="openai",
        tts_voice="nova",
        piper_voice_path="/opt/piper/x.onnx",
        whisper_model="large-v3",
        autonomy_mode="safe",
        approval_timeout=300,
        db_path=":memory:",
    )


class FakeControls:
    def __init__(self):
        self.calls = []
        self._snapshot = [
            {"project": "qwing", "enabled": True, "mode": "safe",
             "voice": "nova", "engine": "openai", "last_active": True},
            {"project": "othersapp", "enabled": False, "mode": "full",
             "voice": "echo", "engine": "openai", "last_active": False},
        ]

    async def toggle(self, project, on):
        self.calls.append(("toggle", project, on))

    async def set_mode(self, project, mode):
        self.calls.append(("set_mode", project, mode))

    async def set_voice(self, project, voice):
        self.calls.append(("set_voice", project, voice))

    async def set_engine(self, name):
        self.calls.append(("set_engine", name))

    def snapshot(self):
        return self._snapshot


def make_message(*, message_id=10, user_id=42, text=None, voice=None,
                 reply_to=None):
    msg = MagicMock()
    msg.message_id = message_id
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    msg.text = text
    msg.voice = voice
    msg.reply_to_message = (
        MagicMock(message_id=reply_to) if reply_to is not None else None
    )
    return msg


@pytest.mark.asyncio
async def test_text_message_from_allowed_user_routes_to_callback():
    received = []

    async def on_user_message(d):
        received.append(d)

    io = TelegramIO(make_cfg(), on_user_message, FakeControls())
    update = MagicMock()
    update.message = make_message(message_id=11, user_id=42,
                                  text="kaip sekasi", reply_to=7)
    update.callback_query = None

    await io._handle_text(update, MagicMock())

    assert received == [{
        "message_id": 11,
        "reply_to": 7,
        "text": "kaip sekasi",
        "is_voice": False,
        "audio": None,
    }]


@pytest.mark.asyncio
async def test_voice_message_downloads_bytes_and_marks_is_voice():
    received = []

    async def on_user_message(d):
        received.append(d)

    voice_obj = MagicMock()
    tg_file = AsyncMock()
    tg_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"OGGDATA"))
    voice_obj.get_file = AsyncMock(return_value=tg_file)

    io = TelegramIO(make_cfg(), on_user_message, FakeControls())
    update = MagicMock()
    update.message = make_message(message_id=12, user_id=42, voice=voice_obj,
                                  reply_to=None)
    update.callback_query = None

    await io._handle_voice(update, MagicMock())

    assert received == [{
        "message_id": 12,
        "reply_to": None,
        "text": "",
        "is_voice": True,
        "audio": b"OGGDATA",
    }]


@pytest.mark.asyncio
async def test_non_whitelisted_user_is_ignored():
    received = []

    async def on_user_message(d):
        received.append(d)

    io = TelegramIO(make_cfg(allowed_id=42), on_user_message, FakeControls())
    update = MagicMock()
    update.message = make_message(message_id=13, user_id=999, text="hello")
    update.callback_query = None

    await io._handle_text(update, MagicMock())

    assert received == []
```

- [ ] **Step 1: Run test to verify it fails**

```
python -m pytest tests/test_telegram_io.py -q
```
Expected: collection/import error — `ModuleNotFoundError: No module named 'voice_bridge.telegram_io'` (or `ImportError: cannot import name 'TelegramIO'`).

- [ ] **Step 2: Write minimal implementation (constructor + inbound handlers + whitelist)**

Create `src/voice_bridge/telegram_io.py`:

```python
"""python-telegram-bot Application: whitelist, inbound voice+text, outbound
text+voice, slash commands, and the /panel inline control board."""
from __future__ import annotations

from typing import Awaitable, Callable, Protocol

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import Config


class Controls(Protocol):
    async def toggle(self, project: str | None, on: bool) -> None: ...
    async def set_mode(self, project: str, mode: str) -> None: ...
    async def set_voice(self, project: str, voice: str) -> None: ...
    async def set_engine(self, name: str) -> None: ...
    def snapshot(self) -> list[dict]: ...


_MODES = ["safe", "full", "ask"]
_OPENAI_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
_ENGINES = ["openai", "piper"]


def _next(seq: list[str], current: str) -> str:
    """Return the element after `current` in `seq`, wrapping around."""
    try:
        i = seq.index(current)
    except ValueError:
        return seq[0]
    return seq[(i + 1) % len(seq)]


def parse_callback(data: str) -> tuple[str, str, str]:
    """Decode '<action>:<project>:<value>' callback data.

    project '' means a global action; value '' means no payload.
    """
    parts = data.split(":", 2)
    while len(parts) < 3:
        parts.append("")
    return parts[0], parts[1], parts[2]


def build_panel_markup(snapshot: list[dict]) -> InlineKeyboardMarkup:
    """Render the /panel inline keyboard from a controls snapshot."""
    rows: list[list[InlineKeyboardButton]] = []
    for row in snapshot:
        proj = row["project"]
        dot = "🟢" if row["enabled"] else "🔴"
        on_label = "ON" if row["enabled"] else "OFF"
        rows.append([
            InlineKeyboardButton(
                f"{dot} {proj}", callback_data=f"noop:{proj}:"),
            InlineKeyboardButton(
                on_label, callback_data=f"toggle:{proj}:"),
            InlineKeyboardButton(
                f"{row['mode']} ▾", callback_data=f"mode:{proj}:"),
            InlineKeyboardButton(
                f"{row['voice']} ▾", callback_data=f"voice:{proj}:"),
        ])
    engine = snapshot[0]["engine"] if snapshot else "openai"
    rows.append([
        InlineKeyboardButton("▶ ALL ON", callback_data="allon::"),
        InlineKeyboardButton("⏸ ALL OFF", callback_data="alloff::"),
        InlineKeyboardButton(
            f"engine: {engine} ▾", callback_data="engine::"),
    ])
    return InlineKeyboardMarkup(rows)


class TelegramIO:
    def __init__(
        self,
        cfg: Config,
        on_user_message: Callable[[dict], Awaitable[None]],
        controls: Controls,
    ) -> None:
        self.cfg = cfg
        self.on_user_message = on_user_message
        self.controls = controls
        self.app: Application | None = None

    # --- whitelist -------------------------------------------------------
    def _allowed(self, user_id: int | None) -> bool:
        return user_id == self.cfg.telegram_allowed_user_id

    # --- inbound handlers ------------------------------------------------
    async def _handle_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        reply_to = (
            msg.reply_to_message.message_id
            if msg.reply_to_message is not None
            else None
        )
        await self.on_user_message({
            "message_id": msg.message_id,
            "reply_to": reply_to,
            "text": msg.text or "",
            "is_voice": False,
            "audio": None,
        })

    async def _handle_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        tg_file = await msg.voice.get_file()
        audio = bytes(await tg_file.download_as_bytearray())
        reply_to = (
            msg.reply_to_message.message_id
            if msg.reply_to_message is not None
            else None
        )
        await self.on_user_message({
            "message_id": msg.message_id,
            "reply_to": reply_to,
            "text": "",
            "is_voice": True,
            "audio": audio,
        })
```

- [ ] **Step 2: Run test to verify it passes**

```
python -m pytest tests/test_telegram_io.py -q
```
Expected: `3 passed`.

- [ ] **Step 3: Commit**

```
git add src/voice_bridge/telegram_io.py tests/test_telegram_io.py
git commit -m "feat(telegram_io): whitelisted voice+text inbound handlers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 4: Write the failing test (send_update + send_question)**

Append to `tests/test_telegram_io.py`:

```python
@pytest.mark.asyncio
async def test_send_update_sends_text_then_voice_and_returns_ids():
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=100))
    bot.send_voice = AsyncMock(return_value=MagicMock(message_id=101))
    io.app = MagicMock()
    io.app.bot = bot

    ids = await io.send_update(
        project="qwing", voice_label="nova",
        text="Pushintas kodas\n---\n```py\nx=1\n```",
        voice_bytes=b"OGGVOICE",
    )

    assert ids == [100, 101]
    bot.send_message.assert_awaited_once()
    sent_text = bot.send_message.await_args.kwargs["text"]
    assert "qwing" in sent_text
    assert "Pushintas kodas" in sent_text
    bot.send_voice.assert_awaited_once()
    assert bot.send_voice.await_args.kwargs["voice"] == b"OGGVOICE"


@pytest.mark.asyncio
async def test_send_update_text_only_when_no_voice_bytes():
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=200))
    bot.send_voice = AsyncMock()
    io.app = MagicMock()
    io.app.bot = bot

    ids = await io.send_update(
        project="qwing", voice_label="nova",
        text="tik tekstas", voice_bytes=None,
    )

    assert ids == [200]
    bot.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_question_returns_message_id():
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=300))
    io.app = MagicMock()
    io.app.bot = bot

    mid = await io.send_question("qwing", "Allow git push?")

    assert mid == 300
    sent = bot.send_message.await_args.kwargs["text"]
    assert "qwing" in sent and "Allow git push?" in sent
```

- [ ] **Step 4: Run test to verify it fails**

```
python -m pytest tests/test_telegram_io.py -q -k "send_update or send_question"
```
Expected: `AttributeError: 'TelegramIO' object has no attribute 'send_update'` (and `send_question`).

- [ ] **Step 5: Write minimal implementation (send_update + send_question + chat id helper)**

Add to `TelegramIO` in `src/voice_bridge/telegram_io.py` (after `_handle_voice`):

```python
    # --- outbound --------------------------------------------------------
    @property
    def _chat_id(self) -> int:
        # Single-chat bot: the only authorized user is also the chat target.
        return self.cfg.telegram_allowed_user_id

    async def send_update(
        self,
        project: str,
        voice_label: str,
        text: str,
        voice_bytes: bytes | None,
    ) -> list[int]:
        bot = self.app.bot
        ids: list[int] = []
        text_msg = await bot.send_message(
            chat_id=self._chat_id,
            text=f"[{project}] {text}",
        )
        ids.append(text_msg.message_id)
        if voice_bytes is not None:
            voice_msg = await bot.send_voice(
                chat_id=self._chat_id,
                voice=voice_bytes,
                caption=f"{project} · {voice_label}",
            )
            ids.append(voice_msg.message_id)
        return ids

    async def send_question(self, project: str, text: str) -> int:
        bot = self.app.bot
        msg = await bot.send_message(
            chat_id=self._chat_id,
            text=f"[{project}] {text}",
        )
        return msg.message_id
```

- [ ] **Step 5: Run test to verify it passes**

```
python -m pytest tests/test_telegram_io.py -q -k "send_update or send_question"
```
Expected: `3 passed`.

- [ ] **Step 6: Commit**

```
git add src/voice_bridge/telegram_io.py tests/test_telegram_io.py
git commit -m "feat(telegram_io): outbound send_update (text+voice) and send_question

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 7: Write the failing test (/panel render + callback dispatch)**

Append to `tests/test_telegram_io.py`:

```python
def test_build_panel_markup_has_per_project_and_global_rows():
    snap = FakeControls().snapshot()
    markup = build_panel_markup(snap)
    kb = markup.inline_keyboard

    # two project rows + one global row
    assert len(kb) == 3
    # project row buttons carry the project name in their callback_data
    toggle_btns = [b for row in kb for b in row
                   if b.callback_data.startswith("toggle:")]
    assert {b.callback_data for b in toggle_btns} == {
        "toggle:qwing:", "toggle:othersapp:"}
    # global row carries all-on/all-off/engine
    last = kb[-1]
    assert [b.callback_data for b in last] == ["allon::", "alloff::", "engine::"]


def test_parse_callback_splits_action_project_value():
    assert parse_callback("toggle:qwing:") == ("toggle", "qwing", "")
    assert parse_callback("mode:qwing:full") == ("mode", "qwing", "full")
    assert parse_callback("allon::") == ("allon", "", "")


@pytest.mark.asyncio
async def test_callback_toggle_off_project_calls_controls():
    controls = FakeControls()  # qwing currently enabled=True
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "toggle:qwing:"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert ("toggle", "qwing", False) in controls.calls
    query.answer.assert_awaited()
    query.edit_message_reply_markup.assert_awaited_once()


@pytest.mark.asyncio
async def test_callback_mode_cycles_to_next_mode():
    controls = FakeControls()  # qwing mode == "safe"
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "mode:qwing:"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert ("set_mode", "qwing", "full") in controls.calls


@pytest.mark.asyncio
async def test_callback_all_on_and_engine_toggle():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)

    async def run_cb(data):
        q = AsyncMock()
        q.data = data
        q.from_user = MagicMock(id=42)
        q.answer = AsyncMock()
        q.edit_message_reply_markup = AsyncMock()
        upd = MagicMock()
        upd.callback_query = q
        await io._handle_callback(upd, MagicMock())

    await run_cb("allon::")
    await run_cb("engine::")  # current engine openai -> piper

    assert ("toggle", None, True) in controls.calls
    assert ("set_engine", "piper") in controls.calls


@pytest.mark.asyncio
async def test_callback_from_non_whitelisted_user_is_ignored():
    controls = FakeControls()
    io = TelegramIO(make_cfg(allowed_id=42), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "toggle:qwing:"
    query.from_user = MagicMock(id=999)
    query.answer = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert controls.calls == []
```

- [ ] **Step 7: Run test to verify it fails**

```
python -m pytest tests/test_telegram_io.py -q -k "panel or callback or parse_callback"
```
Expected: `AttributeError: 'TelegramIO' object has no attribute '_handle_callback'` (panel/parse_callback tests fail on missing behavior only after the handler exists; current run fails at `_handle_callback`).

- [ ] **Step 8: Write minimal implementation (/panel command + callback dispatcher)**

Add to `TelegramIO` in `src/voice_bridge/telegram_io.py` (after `send_question`):

```python
    # --- /panel + callbacks ---------------------------------------------
    async def _cmd_panel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        markup = build_panel_markup(self.controls.snapshot())
        await msg.reply_text("Control panel", reply_markup=markup)

    async def _handle_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if query is None or not self._allowed(query.from_user.id):
            return
        await query.answer()
        action, project, value = parse_callback(query.data)
        snap = {r["project"]: r for r in self.controls.snapshot()}

        if action == "toggle":
            cur = snap.get(project, {}).get("enabled", False)
            await self.controls.toggle(project, not cur)
        elif action == "mode":
            cur = snap.get(project, {}).get("mode", _MODES[0])
            await self.controls.set_mode(project, _next(_MODES, cur))
        elif action == "voice":
            cur = snap.get(project, {}).get("voice", _OPENAI_VOICES[0])
            await self.controls.set_voice(
                project, _next(_OPENAI_VOICES, cur))
        elif action == "allon":
            await self.controls.toggle(None, True)
        elif action == "alloff":
            await self.controls.toggle(None, False)
        elif action == "engine":
            cur = next(iter(snap.values()), {}).get("engine", _ENGINES[0])
            await self.controls.set_engine(_next(_ENGINES, cur))
        elif action == "noop":
            return

        new_markup = build_panel_markup(self.controls.snapshot())
        await query.edit_message_reply_markup(reply_markup=new_markup)
```

- [ ] **Step 8: Run test to verify it passes**

```
python -m pytest tests/test_telegram_io.py -q -k "panel or callback or parse_callback"
```
Expected: `7 passed`.

- [ ] **Step 9: Commit**

```
git add src/voice_bridge/telegram_io.py tests/test_telegram_io.py
git commit -m "feat(telegram_io): /panel inline keyboard and callback_query controls

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 10: Write the failing test (text slash commands)**

Append to `tests/test_telegram_io.py`:

```python
def make_cmd_update(text, user_id=42, message_id=50):
    msg = MagicMock()
    msg.message_id = message_id
    msg.from_user = MagicMock(id=user_id)
    msg.text = text
    msg.reply_text = AsyncMock()
    upd = MagicMock()
    upd.message = msg
    upd.callback_query = None
    return upd


def make_ctx(args):
    ctx = MagicMock()
    ctx.args = args
    return ctx


@pytest.mark.asyncio
async def test_cmd_projects_lists_snapshot():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/projects")

    await io._cmd_projects(upd, make_ctx([]))

    sent = upd.message.reply_text.await_args.args[0]
    assert "qwing" in sent and "othersapp" in sent
    assert "safe" in sent and "nova" in sent


@pytest.mark.asyncio
async def test_cmd_on_with_project_calls_toggle_true():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/on qwing")

    await io._cmd_on(upd, make_ctx(["qwing"]))

    assert ("toggle", "qwing", True) in controls.calls


@pytest.mark.asyncio
async def test_cmd_off_no_arg_is_global():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/off")

    await io._cmd_off(upd, make_ctx([]))

    assert ("toggle", None, False) in controls.calls


@pytest.mark.asyncio
async def test_cmd_mode_sets_per_project():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/mode full qwing")

    await io._cmd_mode(upd, make_ctx(["full", "qwing"]))

    assert ("set_mode", "qwing", "full") in controls.calls


@pytest.mark.asyncio
async def test_cmd_voice_list_replies_voices():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/voice list")

    await io._cmd_voice(upd, make_ctx(["list"]))

    sent = upd.message.reply_text.await_args.args[0]
    assert "nova" in sent and "echo" in sent
    assert controls.calls == []  # listing must not mutate state


@pytest.mark.asyncio
async def test_cmd_voice_set_for_project():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/voice shimmer for qwing")

    await io._cmd_voice(upd, make_ctx(["shimmer", "for", "qwing"]))

    assert ("set_voice", "qwing", "shimmer") in controls.calls


@pytest.mark.asyncio
async def test_cmd_engine_switches_backend():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/engine piper")

    await io._cmd_engine(upd, make_ctx(["piper"]))

    assert ("set_engine", "piper") in controls.calls


@pytest.mark.asyncio
async def test_cmd_status_routes_into_on_user_message():
    received = []

    async def on_user_message(d):
        received.append(d)

    io = TelegramIO(make_cfg(), on_user_message, FakeControls())
    upd = make_cmd_update("/status qwing", message_id=77)

    await io._cmd_status(upd, make_ctx(["qwing"]))

    assert len(received) == 1
    assert received[0]["message_id"] == 77
    assert received[0]["is_voice"] is False
    assert "qwing" in received[0]["text"]


@pytest.mark.asyncio
async def test_cmd_rejects_non_whitelisted():
    controls = FakeControls()
    io = TelegramIO(make_cfg(allowed_id=42), AsyncMock(), controls)
    upd = make_cmd_update("/on qwing", user_id=999)

    await io._cmd_on(upd, make_ctx(["qwing"]))

    assert controls.calls == []
```

- [ ] **Step 10: Run test to verify it fails**

```
python -m pytest tests/test_telegram_io.py -q -k "cmd_"
```
Expected: `AttributeError: 'TelegramIO' object has no attribute '_cmd_projects'` (and the other `_cmd_*` methods).

- [ ] **Step 11: Write minimal implementation (slash command handlers)**

Add to `TelegramIO` in `src/voice_bridge/telegram_io.py` (after `_handle_callback`):

```python
    # --- text slash commands --------------------------------------------
    async def _cmd_projects(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        lines = []
        for r in self.controls.snapshot():
            state = "on" if r["enabled"] else "off"
            star = " *" if r["last_active"] else ""
            lines.append(
                f"{r['project']}: {state} · {r['mode']} · "
                f"{r['voice']}{star}"
            )
        await msg.reply_text(
            "\n".join(lines) if lines else "no projects")

    async def _cmd_on(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        project = context.args[0] if context.args else None
        await self.controls.toggle(project, True)
        await msg.reply_text(
            f"{project or 'all'} on")

    async def _cmd_off(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        project = context.args[0] if context.args else None
        await self.controls.toggle(project, False)
        await msg.reply_text(
            f"{project or 'all'} off")

    async def _cmd_mode(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        if not context.args or context.args[0] not in _MODES:
            await msg.reply_text("usage: /mode <full|safe|ask> [project]")
            return
        mode = context.args[0]
        project = context.args[1] if len(context.args) > 1 else None
        if project is None:
            for r in self.controls.snapshot():
                await self.controls.set_mode(r["project"], mode)
            await msg.reply_text(f"mode {mode} for all")
        else:
            await self.controls.set_mode(project, mode)
            await msg.reply_text(f"mode {mode} for {project}")

    async def _cmd_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        args = context.args
        if not args or args[0] == "list":
            await msg.reply_text(
                "voices: " + ", ".join(_OPENAI_VOICES))
            return
        voice = args[0]
        project = None
        if len(args) >= 3 and args[1] == "for":
            project = args[2]
        if project is None:
            for r in self.controls.snapshot():
                await self.controls.set_voice(r["project"], voice)
            await msg.reply_text(f"voice {voice} for all")
        else:
            await self.controls.set_voice(project, voice)
            await msg.reply_text(f"voice {voice} for {project}")

    async def _cmd_engine(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        if not context.args or context.args[0] not in _ENGINES:
            await msg.reply_text("usage: /engine <openai|piper>")
            return
        name = context.args[0]
        await self.controls.set_engine(name)
        await msg.reply_text(f"engine {name}")

    async def _cmd_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        project = context.args[0] if context.args else ""
        text = f"{project} status please".strip() or "status please"
        await self.on_user_message({
            "message_id": msg.message_id,
            "reply_to": None,
            "text": text,
            "is_voice": False,
            "audio": None,
        })
```

- [ ] **Step 11: Run test to verify it passes**

```
python -m pytest tests/test_telegram_io.py -q -k "cmd_"
```
Expected: `9 passed`.

- [ ] **Step 12: Commit**

```
git add src/voice_bridge/telegram_io.py tests/test_telegram_io.py
git commit -m "feat(telegram_io): slash commands (projects/on/off/mode/voice/engine/status)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 13: Write the failing test (run() wires Application + handlers)**

Append to `tests/test_telegram_io.py`:

```python
@pytest.mark.asyncio
async def test_run_builds_application_and_registers_handlers(monkeypatch):
    import voice_bridge.telegram_io as mod

    added = []
    fake_app = MagicMock()
    fake_app.add_handler = MagicMock(side_effect=lambda h: added.append(h))
    fake_app.initialize = AsyncMock()
    fake_app.start = AsyncMock()
    fake_app.updater = MagicMock()
    fake_app.updater.start_polling = AsyncMock()

    fake_builder = MagicMock()
    fake_builder.token.return_value = fake_builder
    fake_builder.build.return_value = fake_app

    monkeypatch.setattr(
        mod.Application, "builder",
        classmethod(lambda cls: fake_builder),
    )

    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    await io.run()

    fake_builder.token.assert_called_once_with("TESTTOKEN")
    assert io.app is fake_app
    fake_app.initialize.assert_awaited_once()
    fake_app.updater.start_polling.assert_awaited_once()
    # at least: panel, projects, on, off, mode, voice, engine, status,
    # callback, text msg, voice msg  == 11 handlers
    assert len(added) >= 11

    cmd_names = set()
    for h in added:
        cmds = getattr(h, "commands", None)
        if cmds:
            cmd_names |= set(cmds)
    assert {"panel", "projects", "on", "off",
            "mode", "voice", "engine", "status"} <= cmd_names
```

- [ ] **Step 13: Run test to verify it fails**

```
python -m pytest tests/test_telegram_io.py -q -k "test_run_builds_application"
```
Expected: `AttributeError: 'TelegramIO' object has no attribute 'run'` (method not yet defined).

- [ ] **Step 14: Write minimal implementation (run + handler registration)**

Add to `TelegramIO` in `src/voice_bridge/telegram_io.py` (after `_cmd_status`):

```python
    # --- lifecycle -------------------------------------------------------
    async def run(self) -> None:
        app = Application.builder().token(
            self.cfg.telegram_bot_token).build()
        self.app = app

        only_me = filters.User(user_id=self.cfg.telegram_allowed_user_id)

        app.add_handler(CommandHandler("panel", self._cmd_panel,
                                       filters=only_me))
        app.add_handler(CommandHandler("projects", self._cmd_projects,
                                       filters=only_me))
        app.add_handler(CommandHandler("on", self._cmd_on,
                                       filters=only_me))
        app.add_handler(CommandHandler("off", self._cmd_off,
                                       filters=only_me))
        app.add_handler(CommandHandler("mode", self._cmd_mode,
                                       filters=only_me))
        app.add_handler(CommandHandler("voice", self._cmd_voice,
                                       filters=only_me))
        app.add_handler(CommandHandler("engine", self._cmd_engine,
                                       filters=only_me))
        app.add_handler(CommandHandler("status", self._cmd_status,
                                       filters=only_me))
        app.add_handler(CallbackQueryHandler(self._handle_callback))
        app.add_handler(MessageHandler(
            only_me & filters.VOICE, self._handle_voice))
        app.add_handler(MessageHandler(
            only_me & filters.TEXT & ~filters.COMMAND, self._handle_text))

        await app.initialize()
        await app.start()
        await app.updater.start_polling()
```

- [ ] **Step 14: Run test to verify it passes**

```
python -m pytest tests/test_telegram_io.py -q
```
Expected: all tests pass (e.g. `23 passed`).

- [ ] **Step 15: Commit**

```
git add src/voice_bridge/telegram_io.py tests/test_telegram_io.py
git commit -m "feat(telegram_io): run() builds Application and registers handlers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Bridge wiring (main)

**Files:**
- Create: `src/voice_bridge/bridge.py`
- Test: `tests/test_bridge.py`
- Modify: none

**Interfaces:**

Consumes (from earlier tasks — use exactly these signatures):
- `config.py`: `def load_config(env: Mapping[str,str] | None = None) -> Config`; `def load_projects(path: str = 'projects.yaml') -> list[ProjectConfig]`; `def effective_voice(project: ProjectConfig, cfg: Config) -> str`
- `routing.py`: `class Store(db_path:str)` with `async def init() -> None`, `async def map_message(message_id:int, project:str) -> None`, `async def project_for_message(message_id:int) -> str | None`, `async def set_last_active(project:str) -> None`, `async def get_last_active() -> str | None`, `async def is_enabled(project:str) -> bool`
- `sanitizer.py`: `def prepare_outbound(message:str) -> tuple[str,str]` (returns `(full_text, spoken)`)
- `tts/__init__.py`: `def get_tts(cfg: Config) -> TTSBackend`; `TTSBackend.synthesize(self, text:str, voice:str) -> bytes`
- `stt.py`: `class Transcriber(model_name:str, language:str='lt')` with `async def transcribe(audio:bytes) -> str`
- `approvals.py`: `class ApprovalManager` with `def has_pending(message_id:int) -> bool`, `def resolve(message_id:int, approved:bool) -> bool`; `def parse_yes_no(text:str) -> bool | None`; `class ApprovalManager(send_question, timeout)`
- `notify_tool.py`: `def make_notify_server(on_notify) -> server`
- `sessions.py`: `class SessionManager(projects, cfg, store, on_outbound, approvals, notify_server)` with `async def start_all() -> None`, `async def deliver(project:str, text:str) -> None`, `async def stop_all() -> None`, `def project(name:str) -> ProjectConfig | None`
- `telegram_io.py`: `class TelegramIO(cfg, on_user_message, controls)` with `async def send_update(project, voice_label, text, voice_bytes) -> list[int]`, `async def send_question(project, text) -> int`, `async def run() -> None`
- types: `Config`, `ProjectConfig`, `Outbound(project:str, text:str, spoken:str)`

Produces (exact signatures this module exposes):
- `async def main() -> None`
- `async def resolve_target(msg: dict, store: 'Store') -> str | None` — pure-ish routing helper: returns the project name a non-approval inbound message should be delivered to. `reply_to` set → `store.project_for_message(reply_to)`; else → `store.get_last_active()`.
- `def make_outbound(tts, telegram, store, cfg, sessions) -> Callable[[Outbound], Awaitable[None]]` — builds the outbound closure.
- `def make_inbound(transcriber, store, approvals, sessions) -> Callable[[dict], Awaitable[None]]` — builds the inbound closure.

TDD steps:

- [ ] **Step 1: Write the failing test for `resolve_target`**

Create `tests/test_bridge.py`:

```python
import pytest

from voice_bridge.bridge import resolve_target


class FakeStore:
    def __init__(self, by_message=None, last_active=None):
        self._by_message = by_message or {}
        self._last_active = last_active

    async def project_for_message(self, message_id):
        return self._by_message.get(message_id)

    async def get_last_active(self):
        return self._last_active


@pytest.mark.asyncio
async def test_resolve_target_reply_to_maps_to_project():
    store = FakeStore(by_message={42: "qwing"}, last_active="othersapp")
    msg = {"message_id": 7, "reply_to": 42, "text": "go on", "is_voice": False, "audio": None}
    assert await resolve_target(msg, store) == "qwing"


@pytest.mark.asyncio
async def test_resolve_target_no_reply_falls_back_to_last_active():
    store = FakeStore(by_message={}, last_active="othersapp")
    msg = {"message_id": 7, "reply_to": None, "text": "go on", "is_voice": False, "audio": None}
    assert await resolve_target(msg, store) == "othersapp"


@pytest.mark.asyncio
async def test_resolve_target_reply_to_unknown_message_returns_none():
    store = FakeStore(by_message={}, last_active=None)
    msg = {"message_id": 7, "reply_to": 999, "text": "go on", "is_voice": False, "audio": None}
    assert await resolve_target(msg, store) is None


@pytest.mark.asyncio
async def test_resolve_target_no_reply_no_last_active_returns_none():
    store = FakeStore(by_message={}, last_active=None)
    msg = {"message_id": 7, "reply_to": None, "text": "go on", "is_voice": False, "audio": None}
    assert await resolve_target(msg, store) is None
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_bridge.py -q
```
Expected: collection/import error `ImportError: cannot import name 'resolve_target' from 'voice_bridge.bridge'` (or `ModuleNotFoundError: No module named 'voice_bridge.bridge'`).

- [ ] **Step 3: Write minimal implementation of `resolve_target`**

Create `src/voice_bridge/bridge.py`:

```python
"""Bridge: wire all modules together and run the main async loop."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from .approvals import ApprovalManager, make_can_use_tool, parse_yes_no
from .config import effective_voice, load_config, load_projects
from .notify_tool import make_notify_server
from .routing import Store
from .sanitizer import prepare_outbound
from .sessions import SessionManager
from .stt import Transcriber
from .telegram_io import TelegramIO
from .tts import get_tts
from .types import Outbound


async def resolve_target(msg: dict, store: Store) -> str | None:
    """Pick the project a (non-approval) inbound message goes to.

    reply_to set -> the project that owns that message; else last-active.
    """
    reply_to = msg.get("reply_to")
    if reply_to is not None:
        return await store.project_for_message(reply_to)
    return await store.get_last_active()
```

(If `Outbound` lives in `config.py` rather than a `types` module, import it from there instead; the contract lists it under `types` — use whichever module the earlier task placed it in.)

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_bridge.py -q
```
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```
git add src/voice_bridge/bridge.py tests/test_bridge.py
git commit -m "feat(bridge): resolve_target routing helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Write the failing test for `make_outbound`**

Append to `tests/test_bridge.py`:

```python
from voice_bridge.bridge import make_outbound


class FakeOutboundStore:
    def __init__(self):
        self.mapped = []
        self.last_active = []

    async def map_message(self, message_id, project):
        self.mapped.append((message_id, project))

    async def set_last_active(self, project):
        self.last_active.append(project)


class FakeTTS:
    def __init__(self, out=b"OGGDATA"):
        self.out = out
        self.calls = []

    async def synthesize(self, text, voice):
        self.calls.append((text, voice))
        return self.out


class FakeTelegram:
    def __init__(self, ids):
        self.ids = ids
        self.updates = []

    async def send_update(self, project, voice_label, text, voice_bytes):
        self.updates.append((project, voice_label, text, voice_bytes))
        return self.ids


class FakeProject:
    def __init__(self, name, voice=None):
        self.name = name
        self.voice = voice


class FakeSessions:
    def __init__(self, projects):
        self._projects = {p.name: p for p in projects}

    def project(self, name):
        return self._projects.get(name)


class FakeCfg:
    tts_voice = "nova"


@pytest.mark.asyncio
async def test_make_outbound_splits_synthesizes_sends_maps_marks_active():
    store = FakeOutboundStore()
    tts = FakeTTS(out=b"VOICE")
    telegram = FakeTelegram(ids=[101, 102])
    proj = FakeProject("qwing", voice="echo")
    sessions = FakeSessions([proj])
    cfg = FakeCfg()

    outbound = make_outbound(tts, telegram, store, cfg, sessions)
    o = Outbound(project="qwing", text="Done.\n---\n`code here`", spoken="Done.")
    await outbound(o)

    # synthesized the spoken line with the project's effective voice
    assert tts.calls == [("Done.", "echo")]
    # one send_update with full text + voice bytes
    assert telegram.updates == [("qwing", "echo", "Done.\n---\n`code here`", b"VOICE")]
    # both returned message ids mapped to the project
    assert store.mapped == [(101, "qwing"), (102, "qwing")]
    # last-active updated
    assert store.last_active == ["qwing"]


@pytest.mark.asyncio
async def test_make_outbound_tts_failure_falls_back_to_text_only():
    store = FakeOutboundStore()

    class BoomTTS:
        async def synthesize(self, text, voice):
            raise RuntimeError("tts down")

    telegram = FakeTelegram(ids=[201])
    proj = FakeProject("qwing", voice="echo")
    sessions = FakeSessions([proj])
    cfg = FakeCfg()

    outbound = make_outbound(BoomTTS(), telegram, store, cfg, sessions)
    o = Outbound(project="qwing", text="Status ok.", spoken="Status ok.")
    await outbound(o)

    # sent with voice_bytes None (text-only fallback)
    assert telegram.updates == [("qwing", "echo", "Status ok.", None)]
    assert store.mapped == [(201, "qwing")]
    assert store.last_active == ["qwing"]
```

- [ ] **Step 7: Run test to verify it fails**

```
python -m pytest tests/test_bridge.py -k make_outbound -q
```
Expected: `ImportError: cannot import name 'make_outbound' from 'voice_bridge.bridge'`.

- [ ] **Step 8: Write minimal implementation of `make_outbound`**

Append to `src/voice_bridge/bridge.py`:

```python
def make_outbound(
    tts,
    telegram: TelegramIO,
    store: Store,
    cfg,
    sessions: SessionManager,
) -> Callable[[Outbound], Awaitable[None]]:
    """Build the outbound closure.

    prepare_outbound(text) -> (full_text, spoken); synthesize spoken with the
    project's effective voice; send_update; map both ids; set_last_active.
    TTS failure -> text-only fallback (voice_bytes=None).
    """

    async def outbound(o: Outbound) -> None:
        full_text, spoken = prepare_outbound(o.text)
        proj = sessions.project(o.project)
        voice = effective_voice(proj, cfg) if proj is not None else cfg.tts_voice

        voice_bytes: bytes | None
        try:
            voice_bytes = await tts.synthesize(spoken, voice)
        except Exception:
            voice_bytes = None

        message_ids = await telegram.send_update(o.project, voice, full_text, voice_bytes)
        for mid in message_ids:
            await store.map_message(mid, o.project)
        await store.set_last_active(o.project)

    return outbound
```

Note: tests pass `spoken="Done."` and a `text` whose pre-`---` part is also `"Done."`, so `prepare_outbound(o.text)` yields the same spoken line the assertions expect. The contract specifies outbound runs `prepare_outbound` on the message text — `o.spoken` is the agent's hint but the deterministic sanitizer is the source of truth.

- [ ] **Step 9: Run test to verify it passes**

```
python -m pytest tests/test_bridge.py -k make_outbound -q
```
Expected: `2 passed`.

- [ ] **Step 10: Commit**

```
git add src/voice_bridge/bridge.py tests/test_bridge.py
git commit -m "feat(bridge): outbound closure (split, tts, send, map, last-active)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 11: Write the failing test for `make_inbound`**

Append to `tests/test_bridge.py`:

```python
from voice_bridge.bridge import make_inbound


class FakeInboundStore:
    def __init__(self, by_message=None, last_active=None):
        self._by_message = by_message or {}
        self._last_active = last_active

    async def project_for_message(self, message_id):
        return self._by_message.get(message_id)

    async def get_last_active(self):
        return self._last_active


class FakeApprovals:
    def __init__(self, pending=None):
        self._pending = set(pending or [])
        self.resolved = []

    def has_pending(self, message_id):
        return message_id in self._pending

    def resolve(self, message_id, approved):
        self.resolved.append((message_id, approved))
        return True


class FakeTranscriber:
    def __init__(self, text="transcribed"):
        self.text = text
        self.calls = []

    async def transcribe(self, audio):
        self.calls.append(audio)
        return self.text


class DeliverSessions:
    def __init__(self):
        self.delivered = []

    async def deliver(self, project, text):
        self.delivered.append((project, text))


@pytest.mark.asyncio
async def test_make_inbound_text_reply_routes_to_replied_project():
    store = FakeInboundStore(by_message={42: "qwing"}, last_active="othersapp")
    approvals = FakeApprovals(pending=[])
    transcriber = FakeTranscriber()
    sessions = DeliverSessions()

    inbound = make_inbound(transcriber, store, approvals, sessions)
    await inbound({"message_id": 7, "reply_to": 42, "text": "continue", "is_voice": False, "audio": None})

    assert transcriber.calls == []  # text -> no STT
    assert sessions.delivered == [("qwing", "continue")]
    assert approvals.resolved == []


@pytest.mark.asyncio
async def test_make_inbound_voice_is_transcribed_then_delivered():
    store = FakeInboundStore(by_message={}, last_active="qwing")
    approvals = FakeApprovals(pending=[])
    transcriber = FakeTranscriber(text="tęsk darbą")
    sessions = DeliverSessions()

    inbound = make_inbound(transcriber, store, approvals, sessions)
    await inbound({"message_id": 7, "reply_to": None, "text": "", "is_voice": True, "audio": b"OGG"})

    assert transcriber.calls == [b"OGG"]
    assert sessions.delivered == [("qwing", "tęsk darbą")]


@pytest.mark.asyncio
async def test_make_inbound_reply_to_pending_approval_resolves_yes():
    store = FakeInboundStore(by_message={}, last_active=None)
    approvals = FakeApprovals(pending=[55])
    transcriber = FakeTranscriber()
    sessions = DeliverSessions()

    inbound = make_inbound(transcriber, store, approvals, sessions)
    await inbound({"message_id": 7, "reply_to": 55, "text": "taip", "is_voice": False, "audio": None})

    assert approvals.resolved == [(55, True)]
    assert sessions.delivered == []  # not delivered as a turn


@pytest.mark.asyncio
async def test_make_inbound_pending_approval_resolves_no():
    store = FakeInboundStore(by_message={}, last_active=None)
    approvals = FakeApprovals(pending=[55])
    transcriber = FakeTranscriber()
    sessions = DeliverSessions()

    inbound = make_inbound(transcriber, store, approvals, sessions)
    await inbound({"message_id": 7, "reply_to": 55, "text": "ne", "is_voice": False, "audio": None})

    assert approvals.resolved == [(55, False)]
    assert sessions.delivered == []


@pytest.mark.asyncio
async def test_make_inbound_pending_approval_unparseable_does_not_resolve():
    store = FakeInboundStore(by_message={}, last_active=None)
    approvals = FakeApprovals(pending=[55])
    transcriber = FakeTranscriber()
    sessions = DeliverSessions()

    inbound = make_inbound(transcriber, store, approvals, sessions)
    await inbound({"message_id": 7, "reply_to": 55, "text": "maybe later", "is_voice": False, "audio": None})

    assert approvals.resolved == []
    assert sessions.delivered == []


@pytest.mark.asyncio
async def test_make_inbound_unroutable_message_is_dropped():
    store = FakeInboundStore(by_message={}, last_active=None)
    approvals = FakeApprovals(pending=[])
    transcriber = FakeTranscriber()
    sessions = DeliverSessions()

    inbound = make_inbound(transcriber, store, approvals, sessions)
    await inbound({"message_id": 7, "reply_to": None, "text": "hi", "is_voice": False, "audio": None})

    assert sessions.delivered == []
```

- [ ] **Step 12: Run test to verify it fails**

```
python -m pytest tests/test_bridge.py -k make_inbound -q
```
Expected: `ImportError: cannot import name 'make_inbound' from 'voice_bridge.bridge'`.

- [ ] **Step 13: Write minimal implementation of `make_inbound`**

Append to `src/voice_bridge/bridge.py`:

```python
def make_inbound(
    transcriber: Transcriber,
    store: Store,
    approvals: ApprovalManager,
    sessions: SessionManager,
) -> Callable[[dict], Awaitable[None]]:
    """Build the inbound closure.

    Voice -> transcribe. If reply_to has a pending approval, parse yes/no and
    resolve it (never delivered as a turn). Otherwise route via resolve_target
    and deliver the text to that project's session.
    """

    async def inbound(msg: dict) -> None:
        if msg.get("is_voice") and msg.get("audio") is not None:
            text = await transcriber.transcribe(msg["audio"])
        else:
            text = msg.get("text") or ""

        reply_to = msg.get("reply_to")
        if reply_to is not None and approvals.has_pending(reply_to):
            decision = parse_yes_no(text)
            if decision is not None:
                approvals.resolve(reply_to, decision)
            return

        # Route to a project, mutating msg's text with the transcript so
        # resolve_target sees the resolved content if it ever needs it.
        msg = {**msg, "text": text}
        project = await resolve_target(msg, store)
        if project is None:
            return
        await sessions.deliver(project, text)

    return inbound
```

- [ ] **Step 14: Run test to verify it passes**

```
python -m pytest tests/test_bridge.py -k make_inbound -q
```
Expected: `6 passed`.

- [ ] **Step 15: Commit**

```
git add src/voice_bridge/bridge.py tests/test_bridge.py
git commit -m "feat(bridge): inbound closure (stt, approval-resolve, route, deliver)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 16: Write the failing test for `main` wiring**

Append to `tests/test_bridge.py`:

```python
import voice_bridge.bridge as bridge_mod


@pytest.mark.asyncio
async def test_main_wires_components_and_starts(monkeypatch):
    calls = {"store_init": 0, "start_all": 0, "run": 0, "stop_all": 0}

    cfg = FakeCfg()
    cfg.db_path = "/tmp/ignored.db"
    cfg.whisper_model = "large-v3"
    cfg.approval_timeout = 300
    cfg.tts_backend = "openai"

    projects = [FakeProject("qwing", voice="echo")]

    class StubStore:
        def __init__(self, db_path):
            self.db_path = db_path

        async def init(self):
            calls["store_init"] += 1

    class StubTranscriber:
        def __init__(self, model_name, language="lt"):
            self.model_name = model_name

    class StubTTS:
        async def synthesize(self, text, voice):
            return b""

    class StubApprovals:
        def __init__(self, send_question, timeout):
            self.send_question = send_question
            self.timeout = timeout

    class StubSessions:
        def __init__(self, *a, **k):
            pass

        async def start_all(self):
            calls["start_all"] += 1

        async def stop_all(self):
            calls["stop_all"] += 1

    class StubTelegram:
        def __init__(self, cfg, on_user_message, controls):
            self.on_user_message = on_user_message
            self.controls = controls

        async def send_question(self, project, text):
            return 1

        async def run(self):
            calls["run"] += 1

    monkeypatch.setattr(bridge_mod, "load_config", lambda env=None: cfg)
    monkeypatch.setattr(bridge_mod, "load_projects", lambda path="projects.yaml": projects)
    monkeypatch.setattr(bridge_mod, "Store", StubStore)
    monkeypatch.setattr(bridge_mod, "Transcriber", StubTranscriber)
    monkeypatch.setattr(bridge_mod, "get_tts", lambda c: StubTTS())
    monkeypatch.setattr(bridge_mod, "ApprovalManager", StubApprovals)
    monkeypatch.setattr(bridge_mod, "make_notify_server", lambda on_notify: object())
    monkeypatch.setattr(bridge_mod, "SessionManager", StubSessions)
    monkeypatch.setattr(bridge_mod, "TelegramIO", StubTelegram)

    await bridge_mod.main()

    assert calls["store_init"] == 1
    assert calls["start_all"] == 1
    assert calls["run"] == 1
    assert calls["stop_all"] == 1
```

- [ ] **Step 17: Run test to verify it fails**

```
python -m pytest tests/test_bridge.py -k test_main_wires -q
```
Expected: `AttributeError: <module 'voice_bridge.bridge'> does not have the attribute 'main'` (or an `AttributeError`/`TypeError` from the unfinished `main`).

- [ ] **Step 18: Write minimal implementation of `main`**

Append to `src/voice_bridge/bridge.py`:

```python
async def main() -> None:
    """Construct every component, wire the closures, run until stopped."""
    cfg = load_config()
    projects = load_projects()

    store = Store(cfg.db_path)
    await store.init()

    transcriber = Transcriber(cfg.whisper_model, language="lt")
    tts = get_tts(cfg)

    # Forward declarations resolved by closure capture.
    telegram_ref: dict = {}

    async def send_question(project: str, text: str) -> int:
        return await telegram_ref["io"].send_question(project, text)

    approvals = ApprovalManager(send_question, cfg.approval_timeout)

    async def on_notify(summary: str, detail: str) -> None:
        await outbound(Outbound(project="bridge", text=summary, spoken=summary))

    notify_server = make_notify_server(on_notify)

    sessions: SessionManager | None = None

    outbound = make_outbound(tts, _LazyTelegram(telegram_ref), store, cfg, _LazySessions(lambda: sessions))

    async def on_outbound(o: Outbound) -> None:
        await outbound(o)

    sessions = SessionManager(
        projects, cfg, store, on_outbound, approvals, notify_server
    )

    inbound = make_inbound(transcriber, store, approvals, sessions)

    async def on_user_message(msg: dict) -> None:
        await inbound(msg)

    controls = _Controls(sessions, store, cfg)
    telegram = TelegramIO(cfg, on_user_message, controls)
    telegram_ref["io"] = telegram

    await sessions.start_all()
    try:
        await telegram.run()
    finally:
        await sessions.stop_all()
```

The `main` body references two tiny lazy shims (so the outbound closure can resolve `telegram` / `sessions` that are constructed after it) and a `_Controls` adapter. Add them above `main` in the same file:

```python
class _LazyTelegram:
    """Defers telegram lookup until the outbound closure actually fires."""

    def __init__(self, ref: dict):
        self._ref = ref

    async def send_update(self, project, voice_label, text, voice_bytes):
        return await self._ref["io"].send_update(project, voice_label, text, voice_bytes)


class _LazySessions:
    """Defers SessionManager lookup for project() inside the outbound closure."""

    def __init__(self, getter: Callable[[], SessionManager | None]):
        self._getter = getter

    def project(self, name: str):
        sm = self._getter()
        return sm.project(name) if sm is not None else None


class _Controls:
    """Controls protocol impl backing /panel and slash commands."""

    def __init__(self, sessions: SessionManager, store: Store, cfg):
        self._sessions = sessions
        self._store = store
        self._cfg = cfg

    async def toggle(self, project: str | None, on: bool) -> None:
        if project is None:
            for p in self._sessions._projects if hasattr(self._sessions, "_projects") else []:
                await self._sessions.set_enabled(p, on)
        else:
            await self._sessions.set_enabled(project, on)

    async def set_mode(self, project: str, mode: str) -> None:
        await self._sessions.set_mode(project, mode)

    async def set_voice(self, project: str, voice: str) -> None:
        proj = self._sessions.project(project)
        if proj is not None:
            proj.voice = voice

    async def set_engine(self, name: str) -> None:
        self._cfg.tts_backend = name

    async def snapshot(self) -> list[dict]:
        enabled = await self._store.enabled_map()
        last_active = await self._store.get_last_active()
        rows = []
        for proj in (self._sessions.project(n) for n in enabled):
            if proj is None:
                continue
            rows.append(
                {
                    "name": proj.name,
                    "enabled": enabled.get(proj.name, True),
                    "mode": proj.autonomy or self._cfg.autonomy_mode,
                    "voice": proj.voice or self._cfg.tts_voice,
                    "engine": self._cfg.tts_backend,
                    "last_active": last_active == proj.name,
                }
            )
        return rows
```

Note: `_Controls.toggle(None, ...)` iterates whatever project-name source `SessionManager` exposes; if the earlier sessions task names its registry differently, adjust the attribute. The `snapshot`/`toggle` paths are not exercised by Task 10 tests (Controls behavior is owned by the telegram_io / sessions tasks) — they are wired here only so `main` constructs a complete object. Keep `main`'s own test (Step 16) the gate for this step.

- [ ] **Step 19: Run test to verify it passes**

```
python -m pytest tests/test_bridge.py -k test_main_wires -q
```
Expected: `1 passed`.

- [ ] **Step 20: Run the full bridge test file**

```
python -m pytest tests/test_bridge.py -q
```
Expected: `13 passed`.

- [ ] **Step 21: Commit**

```
git add src/voice_bridge/bridge.py tests/test_bridge.py
git commit -m "feat(bridge): main() wiring of all components and closures

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Deployment: systemd + README

**Files:**
- Create: `systemd/voice-bridge.service`
- Create/Modify: `README.md`
- Modify (link only): `docs/superpowers/specs/2026-06-30-voice-bridge-design.md` (no edit required; referenced from README)

**Interfaces:**

Consumes (from earlier tasks — what the operator must know to run the service):
- `async def main() -> None` (bridge.py) — the process entry point; systemd runs `python -m voice_bridge.bridge`.
- Config env keys (config.py `load_config`): `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_USER_ID(int)`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TTS_BACKEND=openai|piper`, `TTS_VOICE`, `PIPER_VOICE_PATH`, `WHISPER_MODEL=large-v3`, `AUTONOMY_MODE=full|safe|ask`, `APPROVAL_TIMEOUT=300`, `DB_PATH`.
- `def load_projects(path: str = 'projects.yaml') -> list[ProjectConfig]` (config.py) — operator authors `projects.yaml`.
- Telegram commands/panel surface (telegram_io.py): `/panel`, `/projects`, `/on`, `/off`, `/status`, `/mode`, `/voice`, `/engine` — exercised by the smoke checklist.
- OGG/Opus constraint: Piper backend (tts/piper_tts.py) shells out to `ffmpeg`; README documents the system dependency.

Produces (ops artifacts — no Python API):
- `systemd/voice-bridge.service` — a `Restart=always` unit with `EnvironmentFile=` pointing at the project `.env`, running `main()`.
- `README.md` — install (venv, Piper voice, ffmpeg), config, BotFather setup, getting your Telegram user id, systemd enable/start, and a manual end-to-end smoke checklist mapping to each of the 7 spec success criteria.

This is an ops/integration task: **no pytest, no automated tests.** The verification is the manual smoke checklist, executed against a real bot once all prior tasks are merged.

---

- [ ] **Step 1: Write the systemd unit file.** Create `systemd/voice-bridge.service` with the complete content below. It is a non-templated single-instance user-runnable unit: `EnvironmentFile=` loads the chmod-600 `.env` so no secrets live in the unit; `Restart=always` satisfies the always-on hosting decision; `WorkingDirectory` is the repo root so `projects.yaml` resolves with its default relative path; `StateDirectory=voice-bridge` makes systemd create `/var/lib/voice-bridge` (matching the default `DB_PATH`) with correct ownership on start.

```ini
# systemd/voice-bridge.service
#
# Install (run as the service user, not root):
#   mkdir -p ~/.config/systemd/user
#   cp systemd/voice-bridge.service ~/.config/systemd/user/voice-bridge.service
#   # edit the three @@ paths below to match your checkout, then:
#   systemctl --user daemon-reload
#   systemctl --user enable --now voice-bridge.service
#   loginctl enable-linger "$USER"   # keep the user service running after logout
#
# Or system-wide (as root): copy to /etc/systemd/system/, set User=, then
#   systemctl daemon-reload && systemctl enable --now voice-bridge.service

[Unit]
Description=Claude Voice Bridge (Telegram <-> Claude Agent SDK)
Documentation=file:///home/home/Projects/claude-voice-bridge/README.md
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/home/Projects/claude-voice-bridge
EnvironmentFile=/home/home/Projects/claude-voice-bridge/.env
ExecStart=/home/home/Projects/claude-voice-bridge/.venv/bin/python -m voice_bridge.bridge
Restart=always
RestartSec=5
TimeoutStopSec=20
KillSignal=SIGINT

# Persistent state dir (DB_PATH default = /var/lib/voice-bridge/state.db).
# StateDirectory only applies to system-wide units; for --user units create the
# dir manually or set DB_PATH=%S/voice-bridge/state.db in .env.
StateDirectory=voice-bridge
StateDirectoryMode=0750

# Logs go to the journal: journalctl --user -u voice-bridge -f
StandardOutput=journal
StandardError=journal
SyslogIdentifier=voice-bridge

[Install]
WantedBy=default.target
```

- [ ] **Step 2: Write the README.** Create `README.md` with the complete content below — it covers prerequisites, venv install, ffmpeg + Piper voice download, `.env` and `projects.yaml` config, BotFather bot creation, obtaining your numeric Telegram user id, systemd enable/start, and the manual smoke checklist mapping each spec success criterion.

````markdown
# Claude Voice Bridge

Control long-running [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview)
sessions hands-free over Telegram. Each project agent sends you a **text** message
(full detail, code included) and a **voice** message (clean spoken summary, no code).
You reply by **voice or text**; the agent continues. Multiple projects run at once and
replies route to the right one.

Full design: [`docs/superpowers/specs/2026-06-30-voice-bridge-design.md`](docs/superpowers/specs/2026-06-30-voice-bridge-design.md).

## Requirements

- Python **3.11+**
- **ffmpeg** on `PATH` (Piper TTS pipes raw audio through ffmpeg to OGG/Opus, and
  Whisper/Telegram audio handling rely on it).
- A Telegram account and the official Telegram app on your phone.
- An Anthropic API key (the Agent SDK requires an API key, **not** a claude.ai login).
- An OpenAI API key (only if `TTS_BACKEND=openai`).
- For Piper TTS: a downloaded Lithuanian Piper voice `.onnx` model + its `.onnx.json`.

Install ffmpeg:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg   # Debian/Ubuntu
```

## Install

```bash
git clone <this-repo> claude-voice-bridge
cd claude-voice-bridge

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

`pip install -e .` installs the runtime deps declared in `pyproject.toml`:
`claude-agent-sdk`, `python-telegram-bot>=21`, `faster-whisper`, `openai`,
`piper-tts`, `pyyaml`, `aiosqlite`.

### Piper voice (only if `TTS_BACKEND=piper`)

Download a Lithuanian voice from the Piper voices repository and note the path to the
`.onnx` file (the matching `.onnx.json` must sit next to it):

```bash
sudo mkdir -p /opt/piper
# download lt_LT-*.onnx and lt_LT-*.onnx.json into /opt/piper/
```

Set `PIPER_VOICE_PATH=/opt/piper/lt_LT-....onnx` in `.env`.

### Whisper model

The first run downloads the `faster-whisper` model named by `WHISPER_MODEL`
(default `large-v3`) and caches it. With a GPU it runs on GPU automatically; otherwise
CPU. The first download is large — do it once before relying on the service.

## Configure

### 1. Create the Telegram bot (BotFather)

1. In Telegram, open a chat with **@BotFather**.
2. Send `/newbot`, choose a name and a username ending in `bot`.
3. BotFather replies with an **HTTP API token** like `123456789:AA...`. This is your
   `TELEGRAM_BOT_TOKEN`.
4. Start a chat with your new bot and send it any message (so it can message you back).

### 2. Get your numeric Telegram user id

Only this id will be allowed to drive the bot (hard whitelist). Get it with **@userinfobot**:

1. Open a chat with **@userinfobot** in Telegram.
2. Send any message; it replies with your numeric `Id`. That integer is
   `TELEGRAM_ALLOWED_USER_ID`.

### 3. `.env`

Copy and fill the example, then lock it down (it holds secrets):

```bash
cp .env.example .env
chmod 600 .env
$EDITOR .env
```

```dotenv
TELEGRAM_BOT_TOKEN=123456789:AA...
TELEGRAM_ALLOWED_USER_ID=11223344
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
TTS_BACKEND=openai            # openai|piper
TTS_VOICE=nova
PIPER_VOICE_PATH=/opt/piper/lt_LT-....onnx
WHISPER_MODEL=large-v3
AUTONOMY_MODE=safe            # full|safe|ask
APPROVAL_TIMEOUT=300
DB_PATH=/var/lib/voice-bridge/state.db
```

> `.env` is git-ignored and must be `chmod 600`. Never commit it.

### 4. `projects.yaml`

Declare the projects you want the bridge to manage. `enabled` is seeded into SQLite on
first run and persisted thereafter; `autonomy`, `voice`, `model`, and
`system_prompt_extra` are optional per-project overrides.

```yaml
projects:
  - name: qwing
    cwd: /home/home/Projects/WhisperX
    enabled: true
    autonomy: safe            # optional; overrides global AUTONOMY_MODE
    voice: nova               # optional; overrides global TTS_VOICE
    model: claude-opus-4-8    # optional
    system_prompt_extra: ""   # optional appended instructions

  - name: othersapp
    cwd: /home/home/Projects/othersapp
    enabled: false
```

## Run

### Foreground (for testing)

```bash
source .venv/bin/activate
python -m voice_bridge.bridge
```

### As a systemd service (always-on)

Edit the three absolute paths in `systemd/voice-bridge.service` to match your checkout
(`WorkingDirectory`, `EnvironmentFile`, `ExecStart` venv python), then:

```bash
mkdir -p ~/.config/systemd/user
cp systemd/voice-bridge.service ~/.config/systemd/user/voice-bridge.service
systemctl --user daemon-reload
systemctl --user enable --now voice-bridge.service
loginctl enable-linger "$USER"   # keep it running after you log out
```

For a `--user` unit the default `DB_PATH=/var/lib/voice-bridge/state.db` is not
writable; either set `DB_PATH=%h/.local/state/voice-bridge/state.db` in `.env` and
`mkdir -p ~/.local/state/voice-bridge`, or install the unit system-wide (copy to
`/etc/systemd/system/`, add `User=<you>`, then `systemctl enable --now`) so
`StateDirectory=voice-bridge` provisions `/var/lib/voice-bridge`.

Logs and status:

```bash
systemctl --user status voice-bridge
journalctl --user -u voice-bridge -f
```

## Telegram controls

| Command | Effect |
|---|---|
| `/panel` | Inline-button control board (per-project on/off, all-on/all-off, mode, voice, engine) |
| `/projects` | List projects + on/off, mode, voice, last-active |
| `/on [project]` / `/off [project]` | Enable/disable a project (no arg = all) |
| `/status [project]` | Ask a project for a quick status |
| `/mode <full\|safe\|ask> [project]` | Set autonomy globally or per project |
| `/voice list` / `/voice <name> [for <project>]` | List/set TTS voice |
| `/engine <openai\|piper>` | Switch TTS backend |

Reply routing: **swipe-reply** a specific message to target that project; a **plain
reply or quick-reply** with no quote goes to the **last-active** project.

## Manual end-to-end smoke test

Run this once after install against a real bot, phone in hand, away from the PC. Each
item maps to a success criterion in §14 of the design spec. Tick every box before
declaring the deployment good.

- [ ] **End-to-end text+voice loop (§14.1).** With the service running and at least one
  enabled project, trigger an outbound update (e.g. `/status qwing`). Confirm you
  receive **two** messages: a text message with full detail and a **voice** message
  with a spoken summary. Reply **by voice** ("kas toliau?") — confirm the agent
  continues. Reply again **by text** — confirm the agent continues. Do this entirely
  from the phone.
- [ ] **Two projects, routing (§14.2).** Enable two projects. Have both message you.
  **Swipe-reply** a message from project A — confirm the reply reaches A. Send a plain
  (no-quote) reply after project B messaged last — confirm it goes to B (last-active
  fallback). Both routing paths verified.
- [ ] **Voice carries no code (§14.3).** Trigger an update whose text contains a code
  block, a file path, a hex color (`#fff`), and a unit (`10px`). Listen to the voice
  message: it must speak **none** of those — no code, no `: 10px`-style fragments. (The
  sanitizer is also unit-tested in Task 4; this confirms it end-to-end.)
- [ ] **Live mode/voice/engine switches (§14.4).** Send `/mode full qwing`, then
  `/mode safe qwing` — confirm behavior changes. Send `/voice list`, then
  `/voice echo for qwing` — confirm the next voice message uses the new voice. Send
  `/engine piper` then `/engine openai` — confirm the engine switches without a restart.
- [ ] **Panel toggles + persistence (§14.5).** Send `/panel`. Tap a project's
  **ON/OFF** — confirm an off project goes silent (no outbound, inbound replies to it
  are rejected with a short note). Tap **ALL OFF** then **ALL ON**. **Restart the
  service** (`systemctl --user restart voice-bridge`) and send `/panel` again — confirm
  the on/off state survived the restart and toggled-on projects resume cleanly from
  their saved session.
- [ ] **Safe-mode approval, deny, timeout (§14.6).** With a project in `safe` mode, get
  it to attempt a risky op (e.g. `git push`). Confirm you receive a voice+text question
  ("… wants to run: git push …. Allow?"). (a) Reply **"ne"** — confirm the op is denied
  and the agent is told it was skipped. (b) Trigger another risky op and **do not
  reply** for longer than `APPROVAL_TIMEOUT` — confirm it auto-denies and the agent
  moves on.
- [ ] **Whitelist (§14.7).** From a **different** Telegram account (or ask someone),
  message the bot. Confirm it is **ignored** — no reply, nothing routed, nothing run.
  Confirm your own whitelisted account still works.

If every box is ticked, the deployment meets the spec success criteria.

## Troubleshooting

- **Bot never replies:** check `journalctl --user -u voice-bridge -f`; verify
  `TELEGRAM_BOT_TOKEN` and that you messaged the bot first.
- **Replies ignored:** `TELEGRAM_ALLOWED_USER_ID` must be your **numeric** id (from
  @userinfobot), and you must message from that exact account.
- **No voice / TTS errors:** verify `ffmpeg` is on `PATH`; for Piper verify
  `PIPER_VOICE_PATH` points at an `.onnx` with its `.onnx.json` beside it. On TTS
  failure the bridge falls back to text-only and logs the error.
- **DB write errors:** ensure the directory in `DB_PATH` exists and is writable by the
  service user (see the systemd `DB_PATH` note above).
````

- [ ] **Step 3: Validate the unit file syntax (sanity check, no service start).** Run `systemd-analyze verify systemd/voice-bridge.service` from the repo root. Expected: no errors (warnings about the `ExecStart`/`EnvironmentFile` paths not existing yet are acceptable on a machine without the venv/.env; the `[Unit]`/`[Service]`/`[Install]` structure must parse cleanly). If `systemd-analyze` is unavailable, instead confirm the three section headers and required keys (`ExecStart`, `EnvironmentFile`, `Restart=always`, `WantedBy`) are present by reading the file. This is a structural check, not a running service.

- [ ] **Step 4: Confirm README links and checklist completeness.** Verify `README.md` links to `docs/superpowers/specs/2026-06-30-voice-bridge-design.md` and that the smoke checklist contains exactly seven boxes, one per §14 success criterion (end-to-end loop, two-project routing, no-code voice, live switches, panel+persistence, approval/deny/timeout, whitelist). This is a documentation review, not code.

- [ ] **Step 5: Commit.** Stage and commit the ops artifacts.

```bash
git add systemd/voice-bridge.service README.md
git commit -m "ops: systemd unit + README install/config/smoke-test docs

Add Restart=always systemd unit loading secrets from .env via
EnvironmentFile, plus README covering venv install, ffmpeg + Piper voice,
.env and projects.yaml config, BotFather setup, obtaining the Telegram
user id, and a manual end-to-end smoke checklist mapped to the spec
success criteria.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
