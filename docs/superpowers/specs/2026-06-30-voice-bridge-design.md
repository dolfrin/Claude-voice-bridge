# Claude Voice Bridge — Design Spec

**Date:** 2026-06-30
**Status:** Approved design (pending written-spec review)
**Owner:** Žygimantas

## 1. Purpose

Control long-running Claude Agent SDK sessions hands-free, away from the PC, over
Telegram. While walking around the city with headphones (or driving), you:

1. Receive a Telegram update from a project agent — as **text** (full detail, code
   included) **and** a **voice message** (clean spoken summary, no code).
2. Reply by **voice or text**.
3. The agent continues working.

Multiple projects run concurrently; replies are routed to the correct project.

This is a standalone project at `/home/home/Projects/claude-voice-bridge/`, separate
from the WhisperX/Qwing repo.

## 2. Locked decisions

| Dimension | Decision |
|---|---|
| Channel | Telegram, single chat |
| Routing | quote-reply → that project; quick-reply / no-reply → last-active project |
| Backbone | Claude Agent SDK (Python, `ClaudeSDKClient`, streaming-input mode) |
| Autonomy | **Selectable** per global/project/live: `full` / `safe` / `ask` |
| TTS | Pluggable: OpenAI + Piper; engine, voice, and per-project voice all selectable |
| STT | Whisper (faster-whisper `large-v3`), Lithuanian |
| Modality | Bidirectional **text + voice**; voice carries no code |
| Auth | `ANTHROPIC_API_KEY` (Agent SDK requires API key, not claude.ai login) |
| Hosting | Always-on `systemd` service on the server (62.84.176.91) |
| Language | User speaks/writes Lithuanian; STT + TTS configured for `lt` |

## 3. Architecture

One always-on Python service, `systemd`-managed:

```
              ┌──────────────────── voice-bridge (Python, systemd) ────────────────────┐
              │                                                                          │
 Telegram ◄───┤ telegram_io   routing(SQLite)    SessionManager ── ClaudeSDKClient × N  │
 (you,        │   ▲    │          msg_id↔proj         │   ▲              (1 per project) │
  headphones) │   │    ▼                              │   │                              │
              │  TTS(OpenAI/Piper)             STT(Whisper)   notify_user (in-proc MCP)  │
              └──────────────────────────────────────────────────────────────────────────┘
```

Each module is a single-responsibility unit with a well-defined interface:

- **telegram_io** — long-polling bot; receives voice + text; sends text + voice;
  handles commands and the `/panel` inline-button control board (`callback_query`
  toggles for per-project on/off, all-on/all-off, mode, voice, engine).
- **stt** — `transcribe(ogg_bytes) -> str` via faster-whisper (VAD trim, `lt`).
- **tts/** — `TTSBackend` interface + `openai_tts` + `piper_tts`;
  `synthesize(text, voice) -> ogg_bytes`.
- **sanitizer** — `to_spoken(markdown) -> str`; strips code for the voice channel.
- **sessions** — `SessionManager`: one `ClaudeSDKClient` per project, streaming-input
  queue, lifecycle, resume.
- **routing** — SQLite-backed `message_id ↔ project` map + pending-approval map.
- **notify_tool** — in-process SDK MCP server exposing `notify_user(...)`.
- **approvals** — `canUseTool` callback wired to the Telegram voice/text loop.
- **config** — env + `projects.yaml` loading and validation.
- **bridge** — wires modules together; main async loop.

## 4. Data flow

### 4.1 Claude → you (outbound)

Triggered by (a) a session turn ending with user-facing text, or (b) the agent
calling `notify_user(...)` mid-turn.

Each outbound update becomes **two Telegram messages** in the chat:

1. **Text message** — the full content (summary + code, diffs, file paths, commands).
2. **Voice message** — the spoken summary only.

Both messages' `message_id`s are stored → the project, so you can reply to either.

**Voice/text split.** Sessions are instructed (appended system prompt) to format
user-facing messages as:

```
<one short spoken-friendly line: status / question, no code, no paths, no commands>
---
<everything technical: code blocks, diffs, file paths, commands>
```

The bridge splits on the first `---`:
- text message = the whole thing,
- voice = the part **before** `---`, passed through the sanitizer.

**Sanitizer (deterministic guarantee).** Before TTS, regardless of agent cooperation:
strip fenced ``` ``` blocks and inline `` `code` ``; drop technical inline fragments
(hex colors `#fff`, dimensions/units like `10px`/`2rem`, file paths, URLs,
`snake_case`/`camelCase`/`CONSTANT` identifiers, lone symbols). If the result is empty
or over a length cap, speak a trimmed version ending with "details in the text"
("detalės tekste"). The voice channel never dictates code or `: 10px`-style noise.

### 4.2 You → Claude (inbound)

You reply by **voice or text**:
- voice message → downloaded OGG → Whisper → text;
- text message → used as-is.

**Routing:**
- **quote-reply** (you swipe-reply a specific message) → `reply_to_message.message_id`
  → look up that project.
- **quick-reply from the notification** or a plain message with no reply → routed to
  the **last-active project** (the project that most recently messaged you). This keeps
  notification quick-reply fully hands-free; explicit quote-reply is for switching.

The resolved text is pushed into that project's streaming-input queue → the agent
resumes the conversation.

## 5. Sessions

- Projects declared in `projects.yaml`:
  ```yaml
  projects:
    - name: qwing
      cwd: /home/home/Projects/WhisperX
      enabled: true             # on/off; default true. Persisted state in SQLite
      autonomy: safe            # optional override of global AUTONOMY_MODE
      voice: nova               # optional override of global TTS_VOICE
      model: claude-opus-4-8    # optional
      system_prompt_extra: ""   # optional appended instructions
  ```
- One `ClaudeSDKClient` per project in streaming-input mode (async generator of user
  messages drained from the project's queue). The bridge awaits each assistant
  response, emits outbound messages, then waits for the next queued user turn.
- **Enabled (on/off) state per project.** When a project is **off** its session is
  held: it is not drained, emits no outbound messages, and inbound replies to it are
  rejected with a short note ("qwing is off"). When toggled **on** it resumes from its
  saved `session_id`. A global toggle flips all projects at once. The `enabled` flag is
  persisted in SQLite (survives restart) and seeded from `projects.yaml`.
- **Persistence/restart:** SQLite stores each project's `session_id` (captured from the
  `system`/`init` message), `enabled` state, the `message_id ↔ project` map, and pending
  approvals. On restart, sessions resume via the SDK `resume` option; lazily re-spawned
  on first use (only for enabled projects).
- **last-active project** is tracked for inbound fallback routing.
- Concurrency: N independent `ClaudeSDKClient` instances in one event loop (SDK
  supports multiple concurrent sessions; each is isolated by `cwd` + `session_id`).

## 6. Autonomy / permissions (selectable)

Three modes, selectable at three levels (global env default, per-project in
`projects.yaml`, live via `/mode`):

| Mode | Behavior | SDK wiring |
|---|---|---|
| `full` | Does everything, only reports | `permissionMode="bypassPermissions"` |
| `safe` | Auto-approve safe ops; **ask by voice** for risky ops | allowlist + `canUseTool` |
| `ask`  | **Ask by voice before every** tool use | `canUseTool` always prompts |

**Risky-op classification (configurable; default list):** `git push`, deploy commands,
`rm`/destructive fs, SSH/server changes, package installs, any path outside the
project `cwd`, network sends, wallet/payment operations. Everything else (read, search,
edit within `cwd`, run tests, build) is safe in `safe` mode.

**Voice approval loop.** When `canUseTool` needs a decision, the bridge formats a
question ("qwing wants to run: git push origin master. Allow?"), sends text + voice,
records a pending approval keyed by the question's `message_id`, and **blocks the tool
call** until you reply. Your reply (voice→Whisper or text) is parsed for yes/no
(Lithuanian + English: taip/ne/yes/no/davai/stop…) → `PermissionResultAllow` /
`PermissionResultDeny`.

**Timeout.** If no answer within `APPROVAL_TIMEOUT` (default 5 min) → `deny`; the agent
is told "no answer, skipped" so it can move on or pause gracefully.

## 7. STT & TTS

- **STT:** faster-whisper, model `WHISPER_MODEL` (default `large-v3`), language `lt`,
  VAD-based silence trimming. Runs on the server (GPU if available, else CPU).
- **TTS:** `TTSBackend` interface; `synthesize(text, voice) -> ogg/opus bytes`
  (Telegram voice = OGG/Opus).
  - `openai_tts` — model `gpt-4o-mini-tts`; voices `alloy|echo|fable|onyx|nova|shimmer|…`.
  - `piper_tts` — local; selectable Lithuanian voice model file.
  - Selected via `TTS_BACKEND=openai|piper`, `TTS_VOICE=<name>`.
  - **Per-project voice** via `projects.yaml` `voice:` — so each project sounds
    different and is identifiable by ear, complementing the spoken name prefix.
  - Live switch: `/voice list`, `/voice <name>` (optionally `/voice <name> for <project>`).

## 8. Telegram control: commands + button panel

### 8.1 Control panel (`/panel`) — inline keyboard buttons

`/panel` renders a live control board using Telegram **inline keyboard buttons**
(`InlineKeyboardMarkup` + `callback_query`). No separate UI app — it lives in the chat.
Tapping a button toggles state instantly and the message edits in place:

```
🟢 qwing       [ ON  ] [ safe ▾ ] [ nova ▾ ]
🔴 othersapp   [ OFF ] [ full ▾ ] [ echo ▾ ]
──────────────────────────────────────────
[ ▶ ALL ON ]   [ ⏸ ALL OFF ]   [ engine: openai ▾ ]
```

- Per-project **ON/OFF** toggle (enable/disable that project).
- Global **ALL ON / ALL OFF** toggles every project at once.
- Per-project autonomy (`safe`/`full`/`ask`) and voice cycle via their buttons.
- Global TTS engine toggle.

Buttons are the easy path when you can glance at the phone; the text commands below do
the same thing and remain available for typing or via voice intent.

### 8.2 Text commands

| Command | Effect |
|---|---|
| `/panel` | Show the inline-button control board |
| `/projects` | List projects + current on/off, mode, voice, last-active |
| `/on [project]` / `/off [project]` | Enable/disable a project (no arg = the global all-on/all-off) |
| `/status [project]` | Ask a project for a quick status |
| `/mode <full\|safe\|ask> [project]` | Set autonomy globally or per project |
| `/voice list` / `/voice <name> [for <project>]` | List/set TTS voice |
| `/engine <openai\|piper>` | Switch TTS backend |

(On/off is the single hold mechanism — there is no separate pause/resume concept.)

## 9. Security

- **Hard whitelist:** only `TELEGRAM_ALLOWED_USER_ID` may interact; all other senders
  are ignored. With `full`/bypass autonomy the bot can run arbitrary commands on the
  server, so this lock is mandatory.
- Bot token, API keys: secrets in `.env` (chmod 600), never committed.
- Optional second factor for `full` mode irreversible ops could be added later (out of
  scope v1).

## 10. Tech stack & repo layout

Python 3.11+. Dependencies: `claude-agent-sdk`, `python-telegram-bot` (v21+),
`faster-whisper`, `openai`, `piper-tts`, `pyyaml`, `aiosqlite`.

```
claude-voice-bridge/
  pyproject.toml
  .env.example
  projects.yaml
  systemd/voice-bridge.service
  src/voice_bridge/
    __init__.py
    config.py
    telegram_io.py
    stt.py
    sanitizer.py
    tts/__init__.py        # TTSBackend interface + factory
    tts/openai_tts.py
    tts/piper_tts.py
    sessions.py            # SessionManager
    routing.py             # SQLite store
    notify_tool.py         # in-process SDK MCP server
    approvals.py           # canUseTool callback + voice approval
    bridge.py              # main()
  tests/
    test_sanitizer.py
    test_routing.py
    test_approvals.py
    test_stt.py            # fixture audio
    test_tts.py
  docs/superpowers/specs/2026-06-30-voice-bridge-design.md
```

## 11. Config / env

```
TELEGRAM_BOT_TOKEN=
TELEGRAM_ALLOWED_USER_ID=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
TTS_BACKEND=openai            # openai|piper
TTS_VOICE=nova
PIPER_VOICE_PATH=/opt/piper/lt_LT-...onnx
WHISPER_MODEL=large-v3
AUTONOMY_MODE=safe            # full|safe|ask
APPROVAL_TIMEOUT=300
DB_PATH=/var/lib/voice-bridge/state.db
```

## 12. Error handling & edge cases

- STT failure / empty transcript → ask you to repeat (voice + text).
- TTS failure → fall back to text-only for that message; log.
- Unroutable inbound (no reply-to, no last-active) → ask which project (list).
- Agent session crash → report the error to you; offer `/resume`.
- Long agent silence during a turn → optional heartbeat note ("still working…").
- Telegram voice notes require a tap to play; documented as a known UX limitation
  (mitigation: text is always present to read; "raise to listen" setting).

## 13. Out of scope (v1, YAGNI)

- Real phone-call interface (Twilio / realtime voice).
- Multiple users / group chats.
- Web dashboard.
- Auto-play workarounds for Telegram voice notes.
- Cross-machine session migration (`SessionStore` adapter).

## 14. Success criteria

- From the phone, away from the PC, you can: receive a project status as text+voice,
  reply by voice and by text, and have the agent continue — verified end-to-end.
- Two projects active simultaneously; replies route correctly (quote-reply and
  last-active fallback both verified).
- Voice never dictates code or `: 10px`-style fragments (sanitizer unit-tested).
- `/mode`, `/voice`, `/engine` switch behavior live.
- `/panel` buttons toggle per-project on/off and all-on/all-off; off projects go silent
  and resume cleanly when switched back on (state survives restart).
- `safe` mode blocks a `git push` and asks by voice; `deny`/timeout both handled.
- Only the whitelisted Telegram user can drive the bot.
