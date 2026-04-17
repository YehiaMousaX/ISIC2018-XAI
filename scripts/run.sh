#!/usr/bin/env bash
# run.sh — Full experiment loop in one command.
# Usage: bash scripts/run.sh ["optional note about what you changed"]
#
# What it does:
#   1. Commits any notebook changes and pushes to GitHub
#   2. Pushes the notebook to Kaggle and triggers a run
#   3. Waits until the run finishes (prints progress every 30s)
#   4. Pulls results into outputs/YYYY-MM-DD_HH-MM/
#   5. Prints the evaluation report so you see results immediately
#   6. If it errored, writes the log to errors/latest_error.txt
set -euo pipefail
export PYTHONUTF8=1

# Prefer the .venv kaggle over any globally installed version
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BIN="$SCRIPT_DIR/../.venv/Scripts"
if [[ -f "$VENV_BIN/kaggle" || -f "$VENV_BIN/kaggle.exe" ]]; then
    export PATH="$VENV_BIN:$PATH"
fi

ENV_FILE="$SCRIPT_DIR/pipeline.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

KAGGLE_USER="${KAGGLE_USER:-yehiasamir}"
KERNEL_SLUG="${KERNEL_SLUG:-isic2018-xai-evaluation}"
KERNEL_ID="${KERNEL_ID:-${KAGGLE_USER}/${KERNEL_SLUG}}"
NOTEBOOK="${NOTEBOOK:-XAI_Evaluation_Pipeline_Kaggle.ipynb}"
NOTE="${1:-}"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M)
OUT_DIR="outputs/${TIMESTAMP}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
MAX_WAIT="${MAX_WAIT:-36000}"

PY_SOURCE="${NOTEBOOK%.ipynb}.py"

# ── 0. Sync .py → .ipynb ─────────────────────────────────────
echo ""
echo "━━━ [0/4] Syncing .py → .ipynb ━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ -f "$PY_SOURCE" ]]; then
  jupytext --to notebook --set-kernel python3 \
           --output "$NOTEBOOK" "$PY_SOURCE"
  echo "✓ ${NOTEBOOK} updated from ${PY_SOURCE}"
else
  echo "No .py source found (${PY_SOURCE}) — using existing notebook."
fi

if [[ ! -f "$NOTEBOOK" ]]; then
  echo "Notebook not found: $NOTEBOOK" >&2
  echo "Set NOTEBOOK in scripts/pipeline.env or create the file first." >&2
  exit 1
fi

# ── 1. Commit & push to GitHub ────────────────────────────────
echo ""
echo "━━━ [1/4] Saving to GitHub ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
git add "$NOTEBOOK" SETUP.md scripts/SETUP.md scripts/pipeline.env 2>/dev/null || true

if git diff --cached --quiet; then
  echo "No changes to commit — pushing as-is."
else
  MSG="experiment: ${TIMESTAMP}"
  [[ -n "$NOTE" ]] && MSG="${MSG} — ${NOTE}"
  git commit -m "$MSG"
fi

git push
echo "✓ GitHub up to date."

# ── 2. Push to Kaggle ─────────────────────────────────────────
echo ""
echo "━━━ [2/4] Pushing to Kaggle ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash scripts/push_to_kaggle.sh
echo "✓ Kernel queued."

# ── 3. Wait for run ───────────────────────────────────────────
echo ""
echo "━━━ [3/4] Waiting for Kaggle run ━━━━━━━━━━━━━━━━━━━━━━━━"
elapsed=0
while true; do
  STATUS=$(kaggle kernels status "$KERNEL_ID" 2>/dev/null | tail -1)
  echo "  [$(date '+%H:%M:%S')] $STATUS"

  if echo "$STATUS" | grep -qiE "complete"; then
    echo "✓ Run complete."
    break
  fi

  if echo "$STATUS" | grep -qiE "error|cancel"; then
    echo ""
    echo "✗ Run failed. Pulling error log..."
    bash scripts/log_error.sh "$KERNEL_ID" 2>/dev/null || true
    echo ""
    echo "━━━ ERROR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python3 -c "
import json, sys
raw = open('errors/latest_error.txt').read()
try:
    entries = json.loads(raw)
    for e in entries:
        if e.get('stream_name') == 'stderr' and any(k in e['data'] for k in ['Error','error','Traceback','Exception']):
            print(e['data'], end='')
except Exception:
    print(raw)
" 2>/dev/null || cat errors/latest_error.txt
    echo ""
    echo "Full log saved to errors/latest_error.txt"
    echo "Tell Claude: 'New error in errors/latest_error.txt — fix it.'"
    exit 1
  fi

  sleep "$POLL_INTERVAL"
  elapsed=$((elapsed + POLL_INTERVAL))
  [[ $elapsed -ge $MAX_WAIT ]] && { echo "Timeout." >&2; exit 2; }
done

# ── 4. Pull results ───────────────────────────────────────────
echo ""
echo "━━━ [4/4] Pulling results ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
mkdir -p "$OUT_DIR"
MAX_PULL_ATTEMPTS=5
pull_attempt=0
until kaggle kernels output "$KERNEL_ID" -p "$OUT_DIR" --quiet; do
  pull_attempt=$((pull_attempt + 1))
  if [[ $pull_attempt -ge $MAX_PULL_ATTEMPTS ]]; then
    echo "✗ Pull failed after ${MAX_PULL_ATTEMPTS} attempts." >&2
    exit 1
  fi
  echo "  Pull attempt ${pull_attempt} failed (IncompleteRead?), retrying in 15s..."
  sleep 15
done
echo "✓ Results saved to ${OUT_DIR}/"

echo ""
find "$OUT_DIR" -maxdepth 2 | sort | sed "s|$OUT_DIR/||" | head -40

# Print the evaluation report if it exists
REPORT=$(find "$OUT_DIR" -name "evaluation_report_*.txt" | head -1)
if [[ -n "$REPORT" ]]; then
  echo ""
  echo "━━━ RESULTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  cat "$REPORT"
fi

# Save a note alongside the results
if [[ -n "$NOTE" ]]; then
  echo "$NOTE" > "$OUT_DIR/note.txt"
fi

# ── Snapshot: notebook + git info ─────────────────────────────
cp "$NOTEBOOK" "$OUT_DIR/notebook.ipynb"
git log -1 --format="commit %H%nauthor %an%ndate   %ai%nsubject %s" > "$OUT_DIR/git_commit.txt"
git diff HEAD~1 HEAD -- "$NOTEBOOK" \
  | python -c "
import sys, json, re
patch = sys.stdin.read()
# Extract only +/- lines from cells (skip JSON boilerplate)
lines = [l for l in patch.splitlines()
         if re.match(r'^[+-]', l) and not re.match(r'^(---|\+\+\+|--- a|\\+\\+\\+ b)', l)]
print('\n'.join(lines))
" > "$OUT_DIR/notebook_diff.txt" 2>/dev/null || true

echo ""
echo "━━━ Done ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Run: $TIMESTAMP"
[[ -n "$NOTE" ]] && echo "Note: $NOTE"
echo "Output: $OUT_DIR/"
