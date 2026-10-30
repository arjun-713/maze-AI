#!/bin/zsh
set -eu
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print 'Run Setup macOS.command first. Do not copy a Linux .venv to this Mac.'
  read '?Press Enter to close...'
  exit 1
fi
if ! .venv/bin/python run.py game; then
  read '?Press Enter to close...'
  exit 1
fi
