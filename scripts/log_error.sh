#!/usr/bin/env bash
# log_error.sh — Pull the latest Kaggle kernel log and write to errors/latest_error.txt.
# Usage: bash scripts/log_error.sh <kernel-id>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BIN="$SCRIPT_DIR/../.venv/Scripts"
[[ -f "$VENV_BIN/kaggle" || -f "$VENV_BIN/kaggle.exe" ]] && export PATH="$VENV_BIN:$PATH"
export PYTHONUTF8=1

ENV_FILE="$SCRIPT_DIR/pipeline.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

KAGGLE_USER="${KAGGLE_USER:-yehiasamir}"
KERNEL_SLUG="${KERNEL_SLUG:-isic2018-xai-evaluation}"
KERNEL_ID="${1:-${KERNEL_ID:-${KAGGLE_USER}/${KERNEL_SLUG}}}"
ERROR_FILE="errors/latest_error.txt"

if [[ -z "$KERNEL_ID" ]]; then
  echo "Usage: $0 <kaggle-user/kernel-slug>" >&2
  exit 1
fi

mkdir -p errors

echo "── Fetching log for ${KERNEL_ID}..."

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

kaggle kernels output "$KERNEL_ID" -p "$TMP_DIR" --quiet 2>/dev/null || true

LOG_FILE=$(find "$TMP_DIR" -name "*.log" | head -1)

if [[ -n "$LOG_FILE" ]]; then
  cp "$LOG_FILE" "$ERROR_FILE"
else
  {
    echo "=== Kaggle kernel status: $(date) ==="
    kaggle kernels status "$KERNEL_ID"
    echo ""
    echo "(No .log file returned. Paste the full Kaggle error output below this line.)"
  } > "$ERROR_FILE"
fi

echo "── Written to ${ERROR_FILE}"
