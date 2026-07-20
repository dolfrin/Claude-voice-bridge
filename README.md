# Claude Voice Bridge

![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Telegram](https://img.shields.io/badge/telegram-bot-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)
![Claude](https://img.shields.io/badge/claude-agent_sdk-D97757?style=for-the-badge)
![Tests](https://img.shields.io/badge/tests-320_passed-2EA043?style=for-the-badge)
![License](https://img.shields.io/badge/license-PolyForm_Noncommercial-blue?style=for-the-badge)

Control long-running [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview)
coding sessions from Telegram with text, voice, files, inline buttons, and per-project
session persistence.

Claude writes full technical replies in Telegram, speaks clean summaries back to you,
accepts voice/text/file input, and keeps each project routed to the right live session.

> Free for personal and non-commercial self-hosted use. Commercial use requires
> written permission.

```text
Telegram text / voice / files
        ↓
Claude Voice Bridge
        ↓
Claude project session + local IDE history
        ↓
Telegram text + optional voice + files + buttons
```

Design notes: [`docs/DESIGN.md`](docs/DESIGN.md).

---

## Highlights

|  | Feature | What it gives you |
|---|---|---|
| 💬 | Text control | Send instructions and replies from Telegram |
| 🎤 | Voice control | Voice messages are transcribed locally with faster-whisper |
| 🔊 | Spoken replies | Claude replies with text plus a clean voice summary |
| 🧭 | Project routing | Quote-reply routes to that project; plain messages use last-active |
| 🎛 | Inline controls | `/menu`, `/projects`, `/panel`, buttons for mode, voice, on/off, stop |
| 📎 | File input | Photos, docs, archives, audio, video, and video notes go to the project inbox |
| 🎵 | Audio files | `mp3`, `m4a`, `ogg`, `wav`, etc. are transcribed and attached |
| 🎬 | Video preview | Video uploads can get a first-frame preview via `ffmpeg` |
| 🧾 | Handoff | `/handoff` shows the selected project's `.claude/voice-bridge-chat.md` history |
| ⛔ | Interrupt | `/stop`, menu Stop, or `!` prefix interrupts/restarts the session |
| 🔘 | Claude buttons | Claude can call `ask_user` to show tappable Telegram choices |
| 📤 | File delivery | Claude can call `send_file` to send project-local files back |
| 🧠 | Session resume | SDK session IDs persist in SQLite and resume after restart |
| 🛡 | Safe mode | Risky tool calls ask for Telegram approval before running |

## Telegram UX

```text
/menu
├─ 🟢 Active       active sessions
├─ 📚 All          all discovered projects
├─ 🎛 Panel        mode / voice / engine / on-off controls
├─ 🧾 Handoff      last-active project transcript
├─ ⛔ Stop         interrupt current work
└─ 🔎 Refresh      refresh local projects
```

Common patterns:

| Action | Use |
|---|---|
| Select project | Tap a project in `/projects` or `/projects_all` |
| Reply to exact project | Telegram quote-reply any bot message from that project |
| Send to current project | Send plain text or voice |
| Interrupt and replace task | Start a message with `!`, e.g. `! stop, fix tests instead` |
| Resume at PC | Open that project's `.claude/voice-bridge-chat.md` or send `/handoff` |

---

## Architecture

One always-on Python service managed by systemd:

```text
TelegramIO
  handles Telegram polling, buttons, commands, files, text, and voice

SessionManager
  owns one long-lived Claude Agent SDK session per project

Store
  persists message routing, last-active project, enabled flags, and SDK session ids

Transcriber + TTS
  local faster-whisper for inbound voice/audio; OpenAI/Piper/Together for outbound voice

Bridge MCP server
  exposes notify_user, ask_user, and send_file to each Claude project session
```

Modules in `src/voice_bridge/`:

| Module | Role |
|---|---|
| `bridge.py` | Top-level wiring, `main()` entry point |
| `config.py` | `load_config()`, `load_projects()`, per-project overrides |
| `routing.py` | SQLite `Store`: msg-id→project, last-active, enabled flags |
| `sessions.py` | `SessionManager`: per-project Agent SDK session lifecycle |
| `telegram_io.py` | Telegram bot: polling, `/menu`/`/panel` inline buttons, slash commands, file I/O |
| `stt.py` | `Transcriber`: faster-whisper speech-to-text |
| `tts/` | Pluggable TTS: `openai_tts.py`, `piper_tts.py`, `together_tts.py` |
| `sanitizer.py` | Strip code/paths/units from spoken text |
| `approvals.py` | Approval flow for safe/ask autonomy modes |
| `attachments.py` | Saves Telegram attachments, extracts archives/video preview frames |
| `notify_tool.py` | In-process MCP tools: `notify_user`, `ask_user`, `send_file` |
| `types.py` | `Outbound` dataclass |

---

## Requirements

### Platform support

Claude Voice Bridge currently works on Linux. macOS and Windows support are planned and
will be added soon.

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
| `piper-tts` | TTS — local Piper backend (for `TTS_BACKEND=piper` or English auto TTS) |

These are declared in `pyproject.toml` and installed automatically by `pip install -e .`
(see Install below).

### API keys

- A local **Claude Code login**. `ANTHROPIC_API_KEY` is optional and only needed
  if you intentionally want pay-per-token API billing.
- An **OpenAI API key** (`sk-...`) — only if `TTS_BACKEND=openai` or `auto`.
- A **Together AI API key** — only if `TTS_BACKEND=together`.

### For Piper TTS (optional)

Download a Piper voice model (for example, an English voice from the
[Piper voices repository](https://github.com/rhasspy/piper/blob/master/VOICES.md)).
You need both the `.onnx` file and its `.onnx.json` config side-by-side:

```bash
sudo mkdir -p /opt/piper
# Download en_US-*.onnx and en_US-*.onnx.json into /opt/piper/
```

Set `PIPER_VOICE_PATH=/opt/piper/en_US-....onnx` in `.env`.

### Whisper model

`faster-whisper` downloads the model named by `WHISPER_MODEL` (default `large-v3`) on
first use and caches it under `~/.cache/huggingface/`. The download is several GB — run
it once before relying on the service. GPU is used automatically if available; CPU works
but is slower.

---

## Quick Start

```bash
git clone <this-repo> claude-voice-bridge
cd claude-voice-bridge

python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env
cp projects.yaml.example projects.yaml
chmod 600 .env
$EDITOR .env
$EDITOR projects.yaml

python -m voice_bridge.bridge
```

Then open Telegram and send:

```text
/menu
```

For always-on use, install `systemd/voice-bridge.service` as a user service after the
foreground run works.

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

# TTS: choose auto, openai, piper, or together
# auto uses Piper only for English-looking text and OpenAI for everything else.
TTS_BACKEND=auto
TTS_VOICE=alloy
OPENAI_API_KEY=sk-...         # only needed if TTS_BACKEND=openai or auto
TOGETHER_API_KEY=             # only needed if TTS_BACKEND=together
TOGETHER_TTS_MODEL=cartesia/sonic
TOGETHER_TTS_LANGUAGE=auto

# Optional: set only if you want Claude pay-per-token API billing.
# Leave unset to use your local Claude Code subscription login.
# ANTHROPIC_API_KEY=sk-ant-...

# Piper (needed if TTS_BACKEND=piper, or for English voice in auto)
PIPER_VOICE_PATH=/opt/piper/en_US-....onnx

# STT
WHISPER_MODEL=large-v3        # downloads on first run

# Autonomy: full, safe, or ask
AUTONOMY_MODE=safe
APPROVAL_TIMEOUT=300          # seconds; auto-deny after this

# State
DB_PATH=/var/lib/voice-bridge/state.db

# IDE catch-up: idle minutes before a project's next turn gets a git/session
# recap prepended (see "Context sync" below)
CATCHUP_IDLE_MINUTES=10
```

All keys and their meaning:

| Key | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | (required) | BotFather HTTP API token |
| `TELEGRAM_ALLOWED_USER_ID` | (required) | Your numeric Telegram user id (whitelist) |
| `ANTHROPIC_API_KEY` | — | Optional; set only for pay-per-token API billing. Leave unset to use local Claude Code subscription login |
| `OPENAI_API_KEY` | — | OpenAI key; required only for `TTS_BACKEND=openai` or `auto` |
| `TOGETHER_API_KEY` | — | Together AI key; required only for `TTS_BACKEND=together` |
| `TOGETHER_TTS_MODEL` | `cartesia/sonic` | Together TTS model |
| `TOGETHER_TTS_LANGUAGE` | `auto` | Together TTS language hint; `auto` omits the provider hint |
| `TTS_BACKEND` | `openai` | `auto`, `openai`, `piper`, or `together` |
| `TTS_VOICE` | `alloy` | Default voice; for OpenAI one of `alloy/ash/ballad/cedar/coral/echo/marin/sage/shimmer/verse` |
| `TTS_ALERT_VOICE` | — | Optional distinct voice for approval questions and crash notices (falls back to `TTS_VOICE` if unset) |
| `PIPER_VOICE_PATH` | — | Absolute path to `.onnx` model; required for `piper` and English auto TTS |
| `WHISPER_MODEL` | `large-v3` | faster-whisper model name |
| `AUTONOMY_MODE` | `safe` | `full` (run everything) / `safe` (ask for risky ops) / `ask` (ask for all) |
| `APPROVAL_TIMEOUT` | `300` | Seconds before an unanswered approval auto-denies |
| `DB_PATH` | `voice-bridge.db` | SQLite database path |
| `AUTO_DISCOVER_PROJECTS` | `false` | Add recent local VS Code/Claude projects to `/panel` at startup, disabled by default |
| `AUTO_DISCOVER_LIMIT` | `12` | Maximum auto-discovered projects to add |
| `OPEN_VSCODE_ON_ENABLE` | `false` | Run `code <project cwd>` when a project is enabled from Telegram |
| `CLOSE_VSCODE_ON_DISABLE` | `false` | Close matching VS Code project windows via `wmctrl` when a project is disabled from Telegram |
| `CATCHUP_IDLE_MINUTES` | `10` | Idle minutes after which a project's next turn triggers an IDE catch-up (see "Context sync" below) |

> `.env` is git-ignored and must be `chmod 600`. Never commit it.

### 4. `projects.yaml`

Copy `projects.yaml.example` to `projects.yaml`, then declare the projects you want the
bridge to manage. `projects.yaml` is git-ignored because it is machine-local.
`enabled` is seeded into SQLite on first run and persisted thereafter (the `/panel`
button or `/on`/`/off` commands override it at runtime). All keys except `name` and
`cwd` are optional overrides of the global config.

If `AUTO_DISCOVER_PROJECTS=true`, the bridge also scans recent local VS Code and Claude
project history under `~/Projects` at startup. Discovered projects are added to the
runtime panel with `enabled: false`; entries in `projects.yaml` always win when names or
directories overlap.

```yaml
projects:
  - name: app
    cwd: /home/home/Projects/app
    display_name: App              # optional label shown in Telegram (default: name)
    enabled: true
    autonomy: safe                 # optional; overrides global AUTONOMY_MODE (safe|full|ask)
    voice: alloy                   # optional; overrides global TTS_VOICE
    model: claude-opus-4-8        # optional Claude model (omit for account default)
    effort: high                   # optional reasoning effort: low|medium|high|xhigh|max
    verbose: false                 # true = stream live tool-activity while working
    system_prompt_extra: ""        # optional extra instructions appended to system prompt

  - name: api
    cwd: /home/home/Projects/api
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

|  | Command | Effect |
|---|---|---|
| 🏠 | `/menu` | Main tappable menu |
| 🎛 | `/panel` | Full control board: per-project on/off, mode, voice, verbose (🔧); global ALL ON / ALL OFF, engine, 💰 Cost, 🗒 Recap |
| 🟢 | `/projects` | Active/last-active projects with select and on/off buttons |
| 📚 | `/projects_all` or `/projects all` | All known projects, including disabled ones |
| 🔎 | `/projects_refresh` | Scan recent VS Code/Claude projects and add new ones disabled |
| 🧾 | `/handoff [project]` | Show the tail of that project's `.claude/voice-bridge-chat.md`; no arg uses last-active |
| ▶️ | `/on [project]` | Enable one project, or all projects with no arg |
| ⏸ | `/off [project]` | Disable one project, or all projects with no arg |
| ⛔ | `/stop [project]` | Interrupt and restart active or named project, clearing queued work |
| 📡 | `/status [project]` | Ask a project for a quick status update |
| ℹ️ | `/info` | Show per-project model, effort, mode, voice, and verbose setting |
| 🛡 | `/mode <full\|safe\|ask> [project]` | Set autonomy globally or per project |
| 🧩 | `/effort <low\|medium\|high\|xhigh\|max> [project]` | Set reasoning effort globally or per project |
| 🔊 | `/voice list` / `/voice <name> [for <project>]` | List or set TTS voices |
| 🔧 | `/verbose [on\|off] [project]` | Toggle live tool-activity streaming (default off); omit on/off to enable |
| 🧠 | `/engine <auto\|openai\|piper\|together>` | Switch TTS backend live |
| 🗒 | `/recap` | Show what changed across all projects while you were away |
| 💰 | `/cost` | Show per-project and total token and cost usage |
| ♾ | `/policies` / `/policies clear [project]` | List, or revoke (all / one project's), the always-allow grants |

Telegram turns are mirrored into each project's `.claude/voice-bridge-chat.md`
so the voice/text conversation is visible from the IDE file tree.

### Interrupts and queueing

Each project has its own queue. If you send multiple turns while Claude is still
working, the bridge reports `Queued: N.` and processes them in order. To break the
current run:

- Send `/stop` to interrupt the last-active project.
- Send `/stop app` to interrupt a specific project.
- Tap `Stop` in `/menu`.
- Prefix a message with `!` to interrupt and immediately send the rest of that message,
  for example `! stop that, fix the tests instead`.

### Attachments and files

You can send files directly to the bot:

- Photos/screenshots are saved into the target project's `.claude/voice-bridge-inbox/`
  and Claude is prompted to inspect the visible UI/text.
- Documents are saved into the same inbox; ZIP/TAR archives are extracted safely.
- Audio files are transcribed with faster-whisper and also saved as files.
- Video/video-note files are saved, and the bridge tries to extract a first preview
  frame with `ffmpeg`.

Claude can also send files back to Telegram by calling the `send_file` MCP tool with a
project-local path. Paths outside the project directory are denied.

### Reply routing

- **Swipe-reply (quote-reply)** a specific message → that message's project receives
  your reply.
- **Plain reply** (no quote) → goes to the **last-active** project (the one that most
  recently sent you a message).

### Voice vs text

- Voice messages you send are transcribed by local faster-whisper with language
  auto-detection. This does not use OpenAI credits.
- Audio files you send are also transcribed by faster-whisper before they are passed to
  Claude.
- Outbound voice messages from the bridge **never contain code**, file paths, hex
  colours, or unit values — the sanitizer strips them before TTS. The text version of
  the same message retains full detail.
- With `TTS_BACKEND=auto`, English-looking output uses local Piper when
  `PIPER_VOICE_PATH` is configured; non-English or uncertain output uses OpenAI TTS. Set
  `/engine openai`, `/engine piper`, or `/engine together` to force a backend.

### Claude MCP tools

Every project session gets an in-process MCP server named `bridge` with these tools:

| Tool | Effect |
|---|---|
| `notify_user` | Send a short status/question to Telegram; summary can be spoken |
| `ask_user` | Ask a Telegram question with tappable choices and return the selected label to Claude |
| `send_file` | Send a project-local file back to Telegram as photo/audio/video/document |

### Autonomy modes

| Mode | Behaviour |
|---|---|
| `full` | Agent runs all operations without asking |
| `safe` | Agent asks for confirmation before flagged risky operations (e.g. `git push`) |
| `ask` | Agent asks before every tool call |

In `safe` and `ask` modes you receive a voice+text question showing the command or diff
and three inline buttons: **✅ Leisti** (Allow once), **❌ Neleisti** (Deny), and
**✅♾ Visada leisti** (Always allow). Tap a button or reply "yes" / "no" by text. No reply
within `APPROVAL_TIMEOUT` seconds auto-denies the operation and the agent is told it was
skipped.

**Always allow** approves this call *and* remembers a per-project policy keyed on a
stable, action-specific signature of what made the call ask — e.g. `git push`, `rm`,
`npm install`, `systemctl restart` — so future *matching* calls in the same project
auto-approve without asking. The signature is deliberately specific: allowing `git push`
never also allows `rm`. To keep a single tap from ever silently widening `safe` mode,
"always allow" is offered **only** for a single, simple invocation of a known operation
verb; if the call can't be generalized safely it falls back to a one-time allow (persists
nothing) and the message says so. Not persisted (allow-once only): compound/chained
commands (`&&`, `||`, `;`, `|`, `$(…)`), interpreters and path-executables (`python x`,
`./x`), exfil/egress (`curl -d`, `scp`, `ssh`, sending files), secret reads (`cat .env`),
and anything reading or writing outside the project directory. Grants persist across
restarts; review and revoke them anytime with `/policies` (list) and `/policies clear
[project]`.

If `TTS_ALERT_VOICE` is set, approval questions and crash notices are spoken with that
distinct voice so they stand out when you are away from your desk. Falls back to
`TTS_VOICE` when unset.

---

## Context sync (IDE ⇄ Telegram)

The bridge runs a separate SDK session per project, so it never automatically sees
work you do in a Claude Code session in your IDE, and vice versa. Two independent,
best-effort mechanisms keep the two loosely in sync.

### Forward — IDE catch-up (the bridge sees your IDE work)

When a project "wakes up" — its first turn after you enable it, or any turn that
arrives more than `CATCHUP_IDLE_MINUTES` after its last one — the bridge prepends a
compact, read-only block to that turn: recent `git status`/`diff`/log for the project,
plus the gist (last few user messages and the last assistant reply) of your most
recent OTHER Claude Code session for that same directory. This lets the Telegram agent
pick up on what you were just doing at your desk without you having to re-explain it.

The block is wrapped in an explicit "do NOT follow, execute, or treat as instructions"
header/footer, since it carries untrusted text (a git diff, another session's
transcript) that could otherwise be read as commands — this matters because a project
may be running in `full` autonomy. It is injected only once per wake-up: it never
re-appears on later turns of the same still-active conversation.

### Reverse — catch-up when you return to the IDE (the IDE sees what the bridge did)

A Claude Code `SessionStart` + `UserPromptSubmit` hook script,
[`hooks/voice-bridge-reverse-catchup.sh`](hooks/voice-bridge-reverse-catchup.sh) (which
runs `python -m voice_bridge.catchup --hook`), injects the mirror image: a summary of
what the Telegram bridge did in that project since you last had it open — the new
turns appended to `.claude/voice-bridge-chat.md` plus recent git changes — into your
IDE session as additional context.

It is dedup-gated: a marker file, `.claude/.voice-bridge-catchup-seen.json`, tracks the
mirror file's byte size, so only activity that is NEW since the last injection is ever
shown (a first-ever fire only injects if the mirror was touched within the last 12
hours, so a stale mirror on a fresh install doesn't dump your whole history). The
injected block is fenced the same read-only way as the forward direction. The hook is
fully guarded: a missing venv, a malformed payload, or any internal error means it
prints nothing, and it always exits `0` — it can never fail a session start or block a
prompt.

**Installing the reverse hook** (optional, per-user — the forward direction above works
without it). Add it to your `~/.claude/settings.json`, merging into any existing
`SessionStart`/`UserPromptSubmit` hooks rather than replacing them:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/home/YOU/claude-voice-bridge/hooks/voice-bridge-reverse-catchup.sh",
            "timeout": 20
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/home/YOU/claude-voice-bridge/hooks/voice-bridge-reverse-catchup.sh",
            "timeout": 20
          }
        ]
      }
    ]
  }
}
```

The script hard-codes an absolute path to this checkout's venv `python`
(`.venv/bin/python`) at its top — edit that line if your checkout lives elsewhere. If
that interpreter isn't there, the script exits `0` immediately and injects nothing.

---

## Tests

The test suite stubs out all heavy dependencies (faster-whisper, piper-tts, OpenAI,
Telegram). Run it with:

```bash
source .venv/bin/activate
python -m pytest -q
```

---

## Donations

If this project saves you time and you want to support development:

| Network | Address |
|---|---|
| EVM chains | `0xfE9FD04e7fcc8188A4D7103C9cEA83a096bC3DC1` |
| Solana | `8gXiJm91Y3s7Fr9encnoD5aCgnGAX59fsfy8K5VKMPHq` |

EVM address works for EVM-compatible currencies and networks.

---

## License

Claude Voice Bridge is released under the PolyForm Noncommercial License 1.0.0.

You may use, modify, and self-host this project for personal, educational, research,
and other non-commercial purposes.

You may not sell this software, offer it as a hosted service, include it in a paid
product, use it inside a commercial product, or otherwise use it commercially without
prior written permission from the author.

For commercial licensing, contact the repository owner or open a GitHub issue.

See [`LICENSE`](LICENSE) and [`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md).

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
  enabled project, trigger an outbound update (e.g. `/status app`). Confirm you
  receive **two** messages: a text message with full detail and a **voice** message
  with a spoken summary. Reply **by voice** ("what is next?") — confirm the agent
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
- [ ] **Live mode/voice/engine switches (§14.4).** Send `/mode full app`, then
  `/mode safe app` — confirm behavior changes. Send `/voice list`, then
  `/voice echo for app` — confirm the next voice message uses the new voice. Send
  `/engine auto`, `/engine piper`, `/engine together`, then `/engine openai` — confirm the engine switches without a restart.
- [ ] **Panel toggles + persistence (§14.5).** Send `/panel`. Tap a project's
  **ON/OFF** button — confirm an off project goes silent (no outbound, inbound replies
  to it are rejected with a short note). Tap **ALL OFF** then **ALL ON**. **Restart the
  service** (`systemctl --user restart voice-bridge`) and send `/panel` again — confirm
  the on/off state survived the restart and toggled-on projects resume cleanly from
  their saved session.
- [ ] **Menu + handoff.** Send `/menu`; tap **Active**, **All**, **Panel**,
  **Handoff**, and **Stop**. Confirm each button edits the Telegram message with the
  expected view or status. Send `/handoff app` and confirm it shows the tail of
  `app`'s project-local `.claude/voice-bridge-chat.md`.
- [ ] **Interrupt.** Start a longer task, then send `/stop app`; confirm the project
  reports it was interrupted and accepts a new turn. Start another long task and send a
  message beginning with `!`; confirm the old work is interrupted and the new text is
  delivered without the `!`.
- [ ] **Attachments.** Send a screenshot/photo with a caption; confirm the target
  project receives a prompt with a `.claude/voice-bridge-inbox/...` path. Send a ZIP or
  TAR and confirm it is extracted under the inbox. Send an audio file and confirm its
  transcript is included in the prompt. Send a video and confirm a preview frame is
  created when `ffmpeg` can read it.
- [ ] **Claude file delivery + ask_user.** Ask Claude to generate a small file and send
  it back; confirm it arrives in Telegram. Ask Claude to choose between options using
  `ask_user`; confirm Telegram shows tappable buttons and Claude receives the selected
  value.
- [ ] **Safe-mode approval, deny, timeout (§14.6).** With a project in `safe` mode, get
  it to attempt a risky op (e.g. `git push`). Confirm you receive a voice+text question
  ("... wants to run: git push .... Allow?"). (a) Reply **"no"** — confirm the op is denied
  and the agent is told it was skipped. (b) Trigger another risky op and **do not
  reply** for longer than `APPROVAL_TIMEOUT` — confirm it auto-denies and the agent
  moves on.
- [ ] **Whitelist (§14.7).** From a **different** Telegram account (or ask someone),
  message the bot. Confirm it is **ignored** — no reply, nothing routed, nothing run.
  Confirm your own whitelisted account still works.

If every box is ticked, the deployment meets the spec success criteria.
