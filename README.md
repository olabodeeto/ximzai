# Ximz

A fully local, always-on voice assistant for macOS. Say **"Hello Ximz"**, ask a
question or give an instruction, and Ximz answers out loud.

Everything runs on your Mac: speech recognition, the language model and the
voice. No cloud APIs, and no audio leaves the machine.

## How it works

```
[always listening] → "Hello Ximz" → "Yes?" → record until you stop talking
→ speech-to-text → LLM + tools → spoken answer → back to listening
```

If you say it all in one go ("Hello Ximz, what time is it?"), Ximz skips the
"Yes?" and answers straight away. After answering it keeps listening for a few
seconds, so you can follow up without the wake phrase.

| Stage | Built with |
| --- | --- |
| Voice activity detection | Silero VAD (the only thing running while idle) |
| Wake phrase + speech-to-text | `mlx-whisper` on the Apple GPU |
| Language model | Ollama (`qwen2.5:3b` on 8 GB, `qwen2.5:7b` on 16 GB+) |
| Voice | Kokoro neural TTS, with macOS `say` as fallback |
| Overlay | PyObjC floating sound-wave window + menu-bar icon |
| Autostart | launchd LaunchAgent |

## What it can do

- Answer questions and chat, in short spoken replies
- Tell the time and date
- Battery level, volume up/down/set
- Open and quit apps (quitting asks for a spoken "yes" first)
- Weather (Open-Meteo)
- Read your Calendar events and Reminders, and add reminders
- Remember things about you across sessions (stored locally in
  `~/Library/Application Support/ximz/memory.json`), and forget them on request

## Requirements

- macOS on Apple Silicon (built on a MacBook Air M2)
- Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) (`brew install ollama`) with a tool-capable model:

  ```bash
  ollama pull qwen2.5:3b
  ```

## Setup

```bash
uv sync                              # install dependencies
uv run python -m ximz.main           # run once in the foreground
```

The first run from Terminal is when macOS asks for microphone access. Allow it;
background processes never get the prompt.

To try it without the wake phrase:

```bash
uv run python -m ximz.main --push-to-talk   # press Enter to talk
uv run python -m ximz.main --no-gui         # no overlay
```

## Run at login

```bash
./scripts/install_agent.sh     # install the LaunchAgent, download voice models, add the `ximz` command
./scripts/uninstall_agent.sh   # remove it all
```

Then control it with:

```bash
ximz status | start | stop | restart | logs
start ximz / stop ximz         # zsh shortcuts; stop lasts until next login
```

Logs go to `~/Library/Logs/ximz/`. After changing code or `config.yaml`, run
`ximz restart`.

## Configuration

All settings live in [`config.yaml`](config.yaml): models, voice, wake-phrase
spellings, VAD thresholds, enabled tool groups, the system prompt and more.
Pick model sizes to match your RAM (see the comment at the top of the file).

If Whisper keeps misspelling names you use often, add them to
`stt.vocabulary`.

## Development

```bash
uv run pytest                        # tests
uv run python scripts/gui_demo.py    # cycle the overlay through every state
```

Code lives in `src/ximz/`. Each stage (wake word, STT, LLM, TTS) sits behind a
small interface so it can be swapped. Tools are registered in
`src/ximz/tools/__init__.py`: each one is a function, a JSON schema and a risk
level (`safe` or `confirm`).

## Privacy

Audio is processed in memory and never written to disk unless you set
`debug.save_audio: true`. Memory facts are stored locally with `chmod 600`.
While running, the only network call is the optional weather lookup.
