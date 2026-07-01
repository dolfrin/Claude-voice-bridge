# Design Notes

Claude Voice Bridge is a single always-on Python service that connects Telegram to
long-running Claude Agent SDK sessions.

## Goals

- Control multiple Claude coding sessions from a phone.
- Support both text and voice input.
- Keep project conversations resumable from the local IDE.
- Route replies to the correct project without manual command syntax for every turn.
- Allow Claude to ask questions, send files, and notify the user through Telegram.
- Keep dangerous operations gated by safe/ask approval modes.

## Runtime Shape

```text
Telegram user
  -> TelegramIO
  -> bridge routing
  -> SessionManager
  -> Claude Agent SDK session
  -> bridge MCP tools
  -> TelegramIO
  -> Telegram user
```

## Main Components

| Component | Responsibility |
|---|---|
| `TelegramIO` | Telegram polling, commands, inline buttons, inbound/outbound files |
| `SessionManager` | One Claude SDK client and queue per project |
| `Store` | SQLite routing state, last-active project, enabled flags, session ids |
| `Transcriber` | Local faster-whisper transcription for voice/audio |
| `TTS` | OpenAI, Piper, Together, or automatic backend routing |
| `ApprovalManager` | Safe/ask tool approval flow |
| `attachments` | Project-local inbox, archive extraction, video preview frames |
| `notify_tool` | In-process MCP tools for `notify_user`, `ask_user`, and `send_file` |

## Project Routing

- Quote-replying a bot message routes the turn to the project that produced that
  message.
- Plain messages route to the last-active project.
- `/projects` and `/projects_all` provide explicit project selection buttons.
- Disabled projects are not written to silently; the bot asks whether to enable and
  send the pending message.

## Persistence

- SQLite stores Telegram message-to-project mappings, enabled flags, last-active
  project, and Claude SDK session ids.
- Each project gets its own transcript mirror at:

```text
<project cwd>/.claude/voice-bridge-chat.md
```

`/handoff` reads that project-local transcript. With no argument, it uses the
last-active project.

## Attachments

Incoming Telegram files are saved under:

```text
<project cwd>/.claude/voice-bridge-inbox/
```

ZIP/TAR archives are extracted safely under the same inbox. Audio files are
transcribed and still kept as files. Video files try to generate a first-frame preview
when `ffmpeg` is available.

## Interrupts

Each project has an independent queue. `/stop`, the menu Stop button, and messages
prefixed with `!` interrupt and restart the target project session, clearing queued
work before the next turn is delivered.

## Security Model

- Telegram user id allowlisting is the primary access boundary.
- `full` autonomy allows Claude to run tools without asking.
- `safe` asks for risky operations.
- `ask` asks for every tool call.
- `send_file` only allows paths inside the active project directory.
