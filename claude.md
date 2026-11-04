# Ximz — Local Voice Assistant

Ximz is a fully local, always-on voice assistant for macOS (MacBook Air M2).
It starts automatically at login, listens for the wake phrase "Hello Ximz",
transcribes the spoken instruction, reasons with a local LLM (via Ollama),
optionally runs tools, and answers out loud.

Everything runs on-device. No cloud APIs, no audio leaves the machine.

## Core loop

```
[always listening] → "Hello Ximz" → spoken "Yes?" → record until silence (VAD)
→ speech-to-text → agent loop (Ollama LLM + tools) → text-to-speech → back to listening
```

If the request comes in the same breath ("Hello Ximz, what time is it?"), the greeting is
skipped and the request is answered directly. A floating overlay (`gui.py`) shows the state
while Ximz is active, with an animated line wave while speaking; it subscribes to `events.py`
and the voice loop never depends on it.

## Tech stack

- **Language:** Python 3.11+ (audio/ML ecosystem is far stronger than Node here)
- **Package manager:** `uv` (`uv sync`, `uv run`)
- **Wake phrase:** "Hello Ximz". v1 (user's choice) = Silero VAD gates short speech bursts
  (0.4–8s), which are transcribed by the same mlx-whisper model and fuzzy-matched as
  "<greeting> <name>" (`wakeword.py`; Whisper hears "Ximz" as Sims/Zims, so spellings live in
  `wake.names`). Long speech is never transcribed. A custom openWakeWord model may replace it
  later behind the same `WakeWord` interface.
- **Voice activity detection:** Silero VAD. It ends recording when the user stops speaking.
- **Speech-to-text:** `mlx-whisper`, which uses Apple's GPU via MLX. Do NOT use faster-whisper,
  which runs CPU-only on Mac.
- **LLM:** Ollama at `http://localhost:11434`, using a tool-capable model (qwen2.5 / llama3.1)
- **Text-to-speech:** macOS `say` for v1 (female voice: Samantha), then Kokoro in a later phase
- **GUI:** PyObjC (AppKit + Quartz). AppKit owns the main thread; the voice loop runs on a worker thread.
- **Audio I/O:** `sounddevice`, `soundfile`, `numpy`
- **Autostart:** launchd LaunchAgent (`~/Library/LaunchAgents/com.featherport.ximz.plist`)

## Hardware constraints

- MacBook Air M2, fanless, with unified memory shared by STT + LLM + TTS.
- Model sizes are chosen by RAM and set in `config.yaml`, never hardcoded:
  - 8GB: `qwen2.5:3b`, `whisper-base.en-mlx`
  - 16GB+: `qwen2.5:7b`, `whisper-small.en-mlx`
- Idle state must be cheap. Only VAD runs continuously (Whisper only on short speech bursts). Load STT/LLM lazily
  or keep them warm, but never run heavy inference while idle.

## Project structure

```
ximz/
├── CLAUDE.md
├── pyproject.toml
├── config.yaml              # models, voice, wake word, thresholds, tool toggles
├── src/ximz/
│   ├── main.py              # entry point; runs the main loop
│   ├── config.py            # loads/validates config.yaml
│   ├── audio.py             # mic stream, recording, playback, chimes
│   ├── wakeword.py          # openWakeWord listener
│   ├── vad.py               # Silero VAD end-of-speech detection
│   ├── stt.py               # mlx-whisper transcription
│   ├── tts.py               # TTS interface; `say` backend first, Kokoro later
│   ├── agent.py             # Ollama tool-calling loop
│   ├── memory.py            # conversation history + persistent user facts
│   └── tools/
│       ├── __init__.py      # tool registry (schema + function)
│       ├── system.py        # time, date, open apps, volume, battery
│       └── ...              # one file per tool group
├── models/                  # custom wake word model (.onnx), gitignored if large
├── scripts/
│   ├── install_agent.sh     # installs + loads the launchd plist
│   └── uninstall_agent.sh
├── launchd/com.featherport.ximz.plist
└── tests/
```

## Key design rules

1. **Each stage is swappable.** STT, TTS, LLM and wake word sit behind small interfaces,
   so a backend can be replaced without touching the loop.
2. **Agent loop** follows the standard Ollama tool-calling pattern: send messages + tool schemas,
   execute any `tool_calls`, append results as `role: "tool"`, and repeat until a plain answer comes back.
   Hard cap of 8 steps per instruction.
3. **Spoken output only.** The system prompt tells the model to reply in 1–3 short spoken
   sentences, with no markdown, lists or code. TTS input is stripped of any formatting before speaking.
4. **Latency matters.**
   - Stream the LLM response and send each completed sentence to TTS immediately.
   - Keep the Ollama model warm (`keep_alive`).
   - Target under 2s from end of speech to first audio on 16GB.
5. **Tool safety.**
   - Every tool declares `risk: safe | confirm`.
   - `confirm` tools (deleting files, sending messages, running shell commands, purchases)
     require Ximz to ask aloud and get a spoken "yes" first.
   - Validate all tool arguments; never trust raw model output.
   - No arbitrary shell execution in v1.
6. **Never crash the daemon.**
   - Catch and log errors per interaction, say a short apology, and return to listening.
   - If Ollama is down, say so and retry with backoff.
7. **Privacy.** Audio is processed in memory and not saved to disk unless `debug.save_audio: true`.

## Autostart (launchd)

- A LaunchAgent (not a LaunchDaemon) runs as the user at login, which gives mic access.
- Set `RunAtLoad: true` and `KeepAlive: true` so it restarts if it crashes.
- Logs go to `~/Library/Logs/ximz/ximz.log` and `ximz.err.log`.
- `ProgramArguments` points to the absolute path of the venv's Python running `-m ximz.main`.
- Ollama must also start at login. The Ollama Mac app can launch at login; if not, the
  install script adds it.
- **Microphone permission:** macOS will not show a permission prompt for a background
  process. Run Ximz once from Terminal first (`uv run python -m ximz.main`) and allow mic
  access, then install the agent. If launchd still has no mic access, the fallback is to
  package Ximz as a minimal `.app` bundle with `NSMicrophoneUsageDescription` and launch
  that from launchd.

## Commands

```bash
uv sync                                  # install deps
uv run python -m ximz.main               # run in foreground (dev)
uv run python -m ximz.main --push-to-talk  # dev mode: press Enter instead of wake word
uv run pytest                            # tests
uv run python scripts/gui_demo.py        # cycle the overlay through every state
./scripts/install_agent.sh               # enable autostart + `ximz` command + shell shortcuts
./scripts/uninstall_agent.sh             # disable autostart, remove command + shortcuts
start ximz / stop ximz                   # zsh shortcuts (stop lasts until next login)
ximz status | restart | logs             # ~/.local/bin/ximz → scripts/ximz
```

After changing code or config, run `ximz restart` so the agent picks it up. The agent runs
with `HF_HUB_OFFLINE=1`; if `stt.model` changes, rerun `install_agent.sh` to download it.

## Build phases

Work through these in order, getting each one working end-to-end before starting the next.

1. ✅ **Push-to-talk MVP.** Enter key → record → mlx-whisper → Ollama (no tools) → `say`.
2. ✅ **VAD.** Record until silence instead of a fixed duration.
3. ✅ **Wake phrase.** Always listening for "Hello Ximz" (Whisper check), spoken "Yes?" on wake.
4. ✅ **Autostart.** launchd agent, logging, crash recovery, Ollama health check, `start/stop ximz`.
   (Also done early: GUI overlay.)
5. **Tools.** Registry plus system tools (time, date, open app, volume, battery, weather), with confirm flow.
6. **Streaming + interruption.** Sentence-chunked TTS; stop speaking when the user talks.
7. **Custom "Hello Ximz" wake word.** Optional: train with openWakeWord's notebook and swap it
   in for the Whisper check if idle cost or false triggers become a problem.
8. **Better voice.** Kokoro TTS backend.
9. **Memory.** Persistent user facts loaded into the system prompt.

## Conventions

- Type hints everywhere; small, focused modules.
- All settings live in `config.yaml`, with no magic numbers in code.
- Use the `logging` module, never `print`, outside dev mode.
- New tools: one function + one JSON schema + a `risk` level, registered in `tools/__init__.py`, with a test.
- Keep dependencies minimal; prefer Apple Silicon–optimized libraries (MLX) where available.
