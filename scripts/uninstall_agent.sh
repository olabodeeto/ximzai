#!/usr/bin/env bash
# Stop Ximz and remove it from login, plus the ximz command and shell shortcuts.
set -euo pipefail

LABEL="com.featherport.ximz"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
ZSHRC="$HOME/.zshrc"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$PLIST" "$HOME/.local/bin/ximz"
if [[ -f "$ZSHRC" ]]; then
  # Delete the marked block added by install_agent.sh.
  sed -i '' '/# >>> ximz shortcuts >>>/,/# <<< ximz shortcuts <<</d' "$ZSHRC"
fi
echo "Ximz removed from login. Logs remain in ~/Library/Logs/ximz."
