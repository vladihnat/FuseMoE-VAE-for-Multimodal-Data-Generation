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

mapfile -t SMOKE_FILES < <(
  find tests -type f -name "smoke*.py" 2>/dev/null | sort
)

if [[ ${#SMOKE_FILES[@]} -eq 0 ]]; then
  echo "No smoke test scripts found under tests/."
  exit 1
fi

echo "Found ${#SMOKE_FILES[@]} smoke script(s)."
echo

for smoke_file in "${SMOKE_FILES[@]}"; do
  echo "============================================================"
  echo "Running: $smoke_file"
  echo "============================================================"
  python "$smoke_file"
  echo
done

echo "All smoke scripts completed successfully."
