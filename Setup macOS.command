#!/bin/zsh
set -eu
cd -- "${0:A:h}"
python_bin=""
for candidate in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
  if command -v "$candidate" >/dev/null 2>&1; then
    python_bin="$(command -v "$candidate")"
    break
  fi
done
if [[ -z "$python_bin" ]]; then
  print 'Install native Python 3.12 first: brew install python@3.12'
  print 'Then reopen this file. Full instructions: MACOS.md'
  read '?Press Enter to close...'
  exit 1
fi
if "$python_bin" run.py setup --codex; then
  print 'Ready. Open Training Studio.command or Play Maze.command.'
else
  print 'Setup did not complete. See the error above and MACOS.md.'
  read '?Press Enter to close...'
  exit 1
fi
read '?Press Enter to close...'
