#!/usr/bin/env bash
# Install Ximz as a login agent, add the `ximz` command, and the
# `start ximz` / `stop ximz` shell shortcuts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.featherport.ximz"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/ximz"
PYTHON="$ROOT/.venv/bin/python"
BIN_DIR="$HOME/.local/bin"
ZSHRC="$HOME/.zshrc"
MARK_BEGIN="# >>> ximz shortcuts >>>"
MARK_END="# <<< ximz shortcuts <<<"

echo "→ Syncing dependencies"
(cd "$ROOT" && uv sync --quiet)

echo "→ Downloading the Kokoro voice model if missing (~350 MB, once)"
KOKORO_DIR="$ROOT/models/kokoro"
mkdir -p "$KOKORO_DIR"
for f in voices-v1.0.bin kokoro-v1.0.onnx; do
  if [[ ! -f "$KOKORO_DIR/$f" ]]; then
    curl -fL --progress-bar -o "$KOKORO_DIR/$f.part" \
      "https://huggingface.co/fastrtc/kokoro-onnx/resolve/main/$f"
    mv "$KOKORO_DIR/$f.part" "$KOKORO_DIR/$f"
  fi
done

echo "→ Downloading the speech model (once; the agent runs offline)"
(cd "$ROOT" && uv run python -c "
import numpy as np
from ximz.config import load_config
from ximz.stt import create_stt
create_stt(load_config().stt).transcribe(np.zeros(8000, dtype=np.float32))
" 2>/dev/null)

echo "→ Writing $PLIST_DST"
mkdir -p "$LOG_DIR" "$(dirname "$PLIST_DST")"
sed -e "s|__PYTHON__|$PYTHON|g" -e "s|__WORKDIR__|$ROOT|g" -e "s|__LOG_DIR__|$LOG_DIR|g" \
  "$ROOT/launchd/$LABEL.plist" > "$PLIST_DST"
plutil -lint "$PLIST_DST" >/dev/null

echo "→ Installing the ximz command to $BIN_DIR/ximz"
mkdir -p "$BIN_DIR"
chmod +x "$ROOT/scripts/ximz"
ln -sf "$ROOT/scripts/ximz" "$BIN_DIR/ximz"

echo "→ Adding 'start ximz' / 'stop ximz' to $ZSHRC"
touch "$ZSHRC"
if ! grep -qF "$MARK_BEGIN" "$ZSHRC"; then
  cat >> "$ZSHRC" <<EOF

$MARK_BEGIN
export PATH="$BIN_DIR:\$PATH"
# Only "start ximz" / "stop ximz" are handled; anything else falls through.
start() { if [[ "\$1" == ximz ]]; then ximz start; elif whence -p start >/dev/null; then command start "\$@"; else echo "start: unknown target '\$1'" >&2; return 1; fi; }
stop()  { if [[ "\$1" == ximz ]]; then ximz stop;  elif whence -p stop  >/dev/null; then command stop  "\$@"; else echo "stop: unknown target '\$1'" >&2; return 1; fi; }
$MARK_END
EOF
fi

echo "→ Making sure Ollama starts at login"
if command -v brew >/dev/null && brew list ollama >/dev/null 2>&1; then
  brew services list | grep -qE '^ollama +started' || brew services start ollama
elif [[ -d /Applications/Ollama.app ]]; then
  echo "  Ollama.app found — make sure it's in System Settings → General → Login Items."
else
  echo "  WARNING: Ollama not found. Install it: brew install ollama" >&2
fi

echo "→ (Re)starting the agent"
"$ROOT/scripts/ximz" stop >/dev/null
"$ROOT/scripts/ximz" start

cat <<EOF

Done. Ximz now starts at every login.
  start ximz / stop ximz     (open a new terminal first)
  ximz status | restart | logs
If macOS asks for microphone access for "python", click Allow.
EOF
