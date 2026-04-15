#!/usr/bin/env bash
# wait_for_kernel.sh — Poll Kaggle until the kernel finishes.
# Usage: bash scripts/wait_for_kernel.sh <kernel-id>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/pipeline.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

KAGGLE_USER="${KAGGLE_USER:-yehiasamir}"
KERNEL_SLUG="${KERNEL_SLUG:-isic2018-xai-evaluation}"
KERNEL_ID="${1:-${KERNEL_ID:-${KAGGLE_USER}/${KERNEL_SLUG}}}"
if [[ -z "$KERNEL_ID" ]]; then
  echo "Usage: $0 <kaggle-user/kernel-slug>" >&2
  exit 1
fi

POLL_INTERVAL="${POLL_INTERVAL:-30}"
MAX_WAIT="${MAX_WAIT:-7200}"

elapsed=0
echo "── Polling ${KERNEL_ID} every ${POLL_INTERVAL}s (max ${MAX_WAIT}s)..."

while true; do
  status=$(kaggle kernels status "$KERNEL_ID" 2>&1 | tail -1)
  echo "  [$(date '+%H:%M:%S')] $status"

  if echo "$status" | grep -qiE "complete|error|cancel"; then
    echo ""
    echo "── Kernel finished: $status"
    if echo "$status" | grep -qi "error"; then
      echo "Run 'make log-error' to capture the failure."
      exit 1
    fi
    exit 0
  fi

  sleep "$POLL_INTERVAL"
  elapsed=$((elapsed + POLL_INTERVAL))

  if [[ $elapsed -ge $MAX_WAIT ]]; then
    echo "  [$(date '+%H:%M:%S')] Still running after ${MAX_WAIT}s — continuing to wait..." >&2
    MAX_WAIT=$((MAX_WAIT + 3600))
  fi
done
