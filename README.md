# Claude Voice Bridge

Control long-running [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview)
sessions hands-free over Telegram. Each project agent sends you a **text** message
(full detail, code included) and a **voice** message (clean spoken summary, no code).
You reply by **voice or text**; the agent continues. Multiple projects run at once and
replies route to the right one.

Full design: [`docs/superpowers/specs/2026-06-30-voice-bridge-design.md`](docs/superpowers/specs/2026-06-30-voice-bridge-design.md).

---

## Architecture

One always-on Python service managed by systemd. A `TelegramIO` front-end handles
inbound/outbound Telegram messages; `SessionManager` runs one Claude Agent SDK session
per project; `Store` (SQLite via aiosqlite) maps message IDs to projects for routing;
`Transcriber` (faster-whisper) converts incoming voice messages to text; the TTS layer
(OpenAI or Piper) converts outbound text to OGG/Opus voice messages; `ApprovalManager`
handles safe/ask-mode confirmation dialogs; `notify_tool` is the in-process MCP tool
that project agents call to push updates back to you.

Modules in `src/voice_bridge/`:

| Module | Role |
|---|---|
| `bridge.py` | Top-level wiring, `main()` entry point |
| `config.py` | `load_config()`, `load_projects()`, per-project overrides |
| `routing.py` | SQLite `Store`: msg-id→project, last-active, enabled flags |
| `sessions.py` | `SessionManager`: per-project Agent SDK session lifecycle |
| `telegram_io.py` | Telegram bot: polling, `/panel` inline buttons, slash commands |
| `stt.py` | `Transcriber`: faster-whisper speech-to-text |
| `tts/` | Pluggable TTS: `openai_tts.py`, `piper_tts.py`, `together_tts.py` |
| `sanitizer.py` | Strip code/paths/units from spoken text |
| `approvals.py` | Approval flow for safe/ask autonomy modes |
| `notify_tool.py` | In-process MCP tool called by agents to push updates |
| `types.py` | `Outbound` dataclass |

---

## Requirements

### System packages

- Python **3.10**
- **ffmpeg** on `PATH` (required by the Piper TTS backend to encode OGG/Opus; also
  used for Telegram audio handling)

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg python3.10 python3.10-venv
```

### Runtime Python packages (NOT installed by default in the test venv)

The test suite stubs these out; for **real operation** you must install:

| Package | Why |
|---|---|
| `faster-whisper` | STT — transcribes your voice messages |
| `piper-tts` | TTS — local Piper backend (only if `TTS_BACKEND=piper`) |

These are declared in `pyproject.toml` and installed automatically by `pip install -e .`
(see Install below).

### API keys

- A local **Claude Code login**. `ANTHROPIC_API_KEY` is optional and only needed
  if you intentionally want pay-per-token API billing.
- An **OpenAI API key** (`sk-...`) — only if `TTS_BACKEND=openai`.
- A **Together AI API key** — only if `TTS_BACKEND=together`.

### For Piper TTS (optional)

Download a Piper voice model (e.g. a Lithuanian voice from the
[Piper voices repository](https://github.com/rhasspy/piper/blob/master/VOICES.md)).
You need both the `.onnx` file and its `.onnx.json` config side-by-side:

```bash
sudo mkdir -p /opt/piper
# Download lt_LT-*.onnx and lt_LT-*.onnx.json into /opt/piper/
```

Set `PIPER_VOICE_PATH=/opt/piper/lt_LT-....onnx` in `.env`.

### Whisper model

`faster-whisper` downloads the model named by `WHISPER_MODEL` (default `large-v3`) on
first use and caches it under `~/.cache/huggingface/`. The download is several GB — run
it once before relying on the service. GPU is used automatically if available; CPU works
but is slower.

---

## Install

```bash
git clone <this-repo> claude-voice-bridge
cd claude-voice-bridge

python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

`pip install -e .` installs all runtime dependencies declared in `pyproject.toml`:
`claude-agent-sdk`, `python-telegram-bot>=21`, `faster-whisper`, `openai`,
`piper-tts`, `pyyaml`, `aiosqlite`.

---

## Configure

### 1. Create the Telegram bot (BotFather)

1. In Telegram, open a chat with **@BotFather**.
2. Send `/newbot`, choose a name and a username ending in `bot`.
3. BotFather replies with an **HTTP API token** like `123456789:AA...`. This is your
   `TELEGRAM_BOT_TOKEN`.
4. Start a chat with your new bot and send it any message (so it can message you back).

### 2. Get your numeric Telegram user id

Only this id will be allowed to drive the bot — it is the **security boundary**. Get
it with **@userinfobot**:

1. Open a chat with **@userinfobot** in Telegram.
2. Send any message; it replies with your numeric `Id`. That integer is
   `TELEGRAM_ALLOWED_USER_ID`.

> Keep `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ALLOWED_USER_ID` secret. Anyone who knows
> your token and id can drive the bot; in `full` autonomy mode the bot can run
> arbitrary shell commands on your server.

### 3. `.env`

Copy and fill the example, then lock it down (it holds secrets):

```bash
cp .env.example .env
chmod 600 .env
$EDITOR .env
```

```dotenv
# Required
TELEGRAM_BOT_TOKEN=123456789:AA...
TELEGRAM_ALLOWED_USER_ID=11223344

# TTS: choose openai, piper, or together
TTS_BACKEND=openai
TTS_VOICE=alloy
OPENAI_API_KEY=sk-...         # only needed if TTS_BACKEND=openai
TOGETHER_API_KEY=             # only needed if TTS_BACKEND=together
TOGETHER_TTS_MODEL=cartesia/sonic
TOGETHER_TTS_LANGUAGE=auto

# Optional: set only if you want Claude pay-per-token API billing.
# Leave unset to use your local Claude Code subscription login.
# ANTHROPIC_API_KEY=sk-ant-...

# Piper (only if TTS_BACKEND=piper)
PIPER_VOICE_PATH=/opt/piper/lt_LT-....onnx

# STT
WHISPER_MODEL=large-v3        # downloads on first run

# Autonomy: full, safe, or ask
AUTONOMY_MODE=safe
APPROVAL_TIMEOUT=300          # seconds; auto-deny after this

# State
DB_PATH=/var/lib/voice-bridge/state.db
```

All keys and their meaning:

| Key | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | (required) | BotFather HTTP API token |
| `TELEGRAM_ALLOWED_USER_ID` | (required) | Your numeric Telegram user id (whitelist) |
| `ANTHROPIC_API_KEY` | — | Optional; set only for pay-per-token API billing. Leave unset to use local Claude Code subscription login |
| `OPENAI_API_KEY` | — | OpenAI key; required only for `TTS_BACKEND=openai` |
| `TOGETHER_API_KEY` | — | Together AI key; required only for `TTS_BACKEND=together` |
| `TOGETHER_TTS_MODEL` | `cartesia/sonic` | Together TTS model |
| `TOGETHER_TTS_LANGUAGE` | `lt` | Together TTS language hint; use `auto` to omit it |
| `TTS_BACKEND` | `openai` | `openai`, `piper`, or `together` |
| `TTS_VOICE` | `alloy` | Voice name; for OpenAI one of `alloy/ash/ballad/cedar/coral/echo/marin/sage/shimmer/verse` |
| `PIPER_VOICE_PATH` | — | Absolute path to `.onnx` model; required for `piper` backend |
| `WHISPER_MODEL` | `large-v3` | faster-whisper model name |
| `AUTONOMY_MODE` | `safe` | `full` (run everything) / `safe` (ask for risky ops) / `ask` (ask for all) |
| `APPROVAL_TIMEOUT` | `300` | Seconds before an unanswered approval auto-denies |
| `DB_PATH` | `voice-bridge.db` | SQLite database path |
| `AUTO_DISCOVER_PROJECTS` | `false` | Add recent local VS Code/Claude projects to `/panel` at startup, disabled by default |
| `AUTO_DISCOVER_LIMIT` | `12` | Maximum auto-discovered projects to add |
| `OPEN_VSCODE_ON_ENABLE` | `false` | Run `code <project cwd>` when a project is enabled from Telegram |

> `.env` is git-ignored and must be `chmod 600`. Never commit it.

### 4. `projects.yaml`

Declare the projects you want the bridge to manage. `enabled` is seeded into SQLite on
first run and persisted thereafter (the `/panel` button or `/on`/`/off` commands
override it at runtime). All keys except `name` and `cwd` are optional overrides of the
global config.

If `AUTO_DISCOVER_PROJECTS=true`, the bridge also scans recent local VS Code and Claude
project history under `~/Projects` at startup. Discovered projects are added to the
runtime panel with `enabled: false`; entries in `projects.yaml` always win when names or
directories overlap.

```yaml
projects:
  - name: qwing
    cwd: /home/home/Projects/WhisperX
    enabled: true
    autonomy: safe            # optional; overrides global AUTONOMY_MODE
    voice: alloy               # optional; overrides global TTS_VOICE
    model: claude-opus-4-8    # optional Claude model for this project
    system_prompt_extra: ""   # optional extra instructions appended to system prompt

  - name: othersapp
    display_name: Others App       # optional label shown in Telegram
    cwd: /home/home/Projects/othersapp
    enabled: false
```

---

## Run

### Foreground (for testing)

```bash
source .venv/bin/activate
python -m voice_bridge.bridge
```

Logs go to stdout. Stop with Ctrl-C (SIGINT).

### As a systemd service (always-on)

Edit the three paths marked `@@` in `systemd/voice-bridge.service` to match your
checkout (`WorkingDirectory`, `EnvironmentFile`, and the venv `python` in `ExecStart`),
then install as a **user** service:

```bash
mkdir -p ~/.config/systemd/user
cp systemd/voice-bridge.service ~/.config/systemd/user/voice-bridge.service
# Edit paths in ~/.config/systemd/user/voice-bridge.service
systemctl --user daemon-reload
systemctl --user enable --now voice-bridge.service
loginctl enable-linger "$USER"   # keep it running after you log out
```

The sample `.env.example` uses `DB_PATH=/var/lib/voice-bridge/state.db`, which
requires `/var/lib/voice-bridge` to exist and be writable for a `--user` unit. Either:

- Set `DB_PATH=$HOME/.local/state/voice-bridge/state.db` in `.env` and
  `mkdir -p ~/.local/state/voice-bridge`, **or**
- Install system-wide: copy to `/etc/systemd/system/`, add `User=<youruser>`, then
  `systemctl daemon-reload && systemctl enable --now voice-bridge.service` — systemd's
  `StateDirectory=voice-bridge` then provisions `/var/lib/voice-bridge` automatically.

Logs and status:

```bash
systemctl --user status voice-bridge
journalctl --user -u voice-bridge -f
```

---

## Telegram controls

| Command | Effect |
|---|---|
| `/panel` | Inline-button control board (per-project on/off, all-on/all-off, mode, voice, engine) |
| `/projects` | List projects + on/off state, mode, voice, last-active |
| `/on [project]` / `/off [project]` | Enable/disable a project (no arg = all) |
| `/status [project]` | Ask a project for a quick status update |
| `/mode <full\|safe\|ask> [project]` | Set autonomy globally or per project |
| `/voice list` / `/voice <name> [for <project>]` | List available voices / set TTS voice |
| `/engine <openai\|piper\|together>` | Switch TTS backend live (no restart needed) |

Telegram turns are mirrored into each project's `.claude/voice-bridge-chat.md`
so the voice/text conversation is visible from the IDE file tree.

### Reply routing

- **Swipe-reply (quote-reply)** a specific message → that message's project receives
  your reply.
- **Plain reply** (no quote) → goes to the **last-active** project (the one that most
  recently sent you a message).

### Voice vs text

- Voice messages you send are transcribed by Whisper (Lithuanian).
- Outbound voice messages from the bridge **never contain code**, file paths, hex
  colours, or unit values — the sanitizer strips them before TTS. The text version of
  the same message retains full detail.

### Autonomy modes

| Mode | Behaviour |
|---|---|
| `full` | Agent runs all operations without asking |
| `safe` | Agent asks for confirmation before flagged risky operations (e.g. `git push`) |
| `ask` | Agent asks before every tool call |

In `safe` and `ask` modes you receive a voice+text question and reply "taip" (yes) or
"ne" (no). No reply within `APPROVAL_TIMEOUT` seconds auto-denies the operation and the
agent is told it was skipped.

---

## Tests

The test suite stubs out all heavy dependencies (faster-whisper, piper-tts, OpenAI,
Telegram). Run it with:

```bash
source .venv/bin/activate
python -m pytest -q
```

---

## Security

- The **whitelist** (`TELEGRAM_ALLOWED_USER_ID`) is the only authentication boundary.
  Messages from any other Telegram account are silently ignored.
- In **`full` autonomy mode** the agent can run arbitrary shell commands on your
  server. Use `safe` or `ask` if that is a concern.
- Keep `TELEGRAM_BOT_TOKEN` secret; anyone with the token can send arbitrary messages
  as the bot.
- `.env` must be `chmod 600` so other OS users cannot read the secrets.

---

## Troubleshooting

- **Bot never replies:** check `journalctl --user -u voice-bridge -f`; verify
  `TELEGRAM_BOT_TOKEN` and that you started a chat with the bot first.
- **Replies ignored:** `TELEGRAM_ALLOWED_USER_ID` must be your **numeric** id (from
  @userinfobot), and you must message from that exact account.
- **No voice / TTS errors:** verify `ffmpeg` is on `PATH`; for Piper verify
  `PIPER_VOICE_PATH` points at an `.onnx` with its `.onnx.json` beside it. On TTS
  failure the bridge falls back to text-only and logs the error.
- **DB write errors:** ensure the directory in `DB_PATH` exists and is writable by the
  service user (see the systemd `DB_PATH` note above).
- **Whisper slow / no GPU:** install CUDA-compatible torch before installing
  faster-whisper for GPU acceleration.

---

## Smoke test

Run this checklist once after install, phone in hand, away from the PC. Each item maps
to a success criterion in §14 of the design spec. Tick every box before declaring the
deployment good.

- [ ] **End-to-end text+voice loop (§14.1).** With the service running and at least one
  enabled project, trigger an outbound update (e.g. `/status qwing`). Confirm you
  receive **two** messages: a text message with full detail and a **voice** message
  with a spoken summary. Reply **by voice** ("kas toliau?") — confirm the agent
  continues. Reply again **by text** — confirm the agent continues. Do this entirely
  from the phone.
- [ ] **Two projects, routing (§14.2).** Enable two projects. Have both send you a
  message. **Swipe-reply** (quote-reply) a message from project A — confirm the reply
  reaches A. Send a plain (no-quote) reply after project B messaged last — confirm it
  goes to B (last-active fallback). Both routing paths verified.
- [ ] **Voice carries no code (§14.3).** Trigger an update whose text contains a code
  block, a file path, a hex colour (`#fff`), and a unit (`10px`). Listen to the voice
  message: it must speak **none** of those — no code, no `: 10px`-style fragments. (The
  sanitizer is also unit-tested separately; this confirms it end-to-end.)
- [ ] **Live mode/voice/engine switches (§14.4).** Send `/mode full qwing`, then
  `/mode safe qwing` — confirm behaviour changes. Send `/voice list`, then
  `/voice echo for qwing` — confirm the next voice message uses the new voice. Send
  `/engine piper`, `/engine together`, then `/engine openai` — confirm the engine switches without a restart.
- [ ] **Panel toggles + persistence (§14.5).** Send `/panel`. Tap a project's
  **ON/OFF** button — confirm an off project goes silent (no outbound, inbound replies
  to it are rejected with a short note). Tap **ALL OFF** then **ALL ON**. **Restart the
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
