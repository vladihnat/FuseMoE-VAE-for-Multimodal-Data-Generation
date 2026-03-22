#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

if [[ -d "$REPO_ROOT/tests" ]]; then
  cd "$REPO_ROOT"
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  cd "$REPO_ROOT"
fi

if [[ ! -d "tests" ]]; then
  echo "Could not find tests/ directory from: $REPO_ROOT"
  exit 1
fi

if [[ -d "tests/pytests" ]]; then
  echo "Running pytest suite from tests/pytests ..."
  pytest tests/pytests "$@"
else
  echo "Running pytest suite from tests/ ..."
  pytest tests "$@"
fi
