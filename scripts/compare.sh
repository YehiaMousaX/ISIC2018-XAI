#!/usr/bin/env bash
# compare.sh — Print a summary table of all past experiment results.
# Usage: bash scripts/compare.sh
set -euo pipefail

echo ""
echo "━━━ Experiment Results ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-20s %-28s %-10s %-8s  %s\n" "Run" "Report" "Rows" "Size" "Note"
echo "────────────────────────────────────────────────────────────────────────────────────────"

for dir in outputs/*/; do
  [[ -d "$dir" ]] || continue

  run=$(basename "$dir")
  note=$(cat "$dir/note.txt" 2>/dev/null || echo "—")
  report=$(find "$dir" -type f \( -name "evaluation_report_*.txt" -o -name "*summary*.csv" -o -name "*metrics*.csv" \) | head -1)

  if [[ -z "$report" ]]; then
    printf "%-20s %-28s %-10s %-8s  %s\n" "$run" "—" "—" "—" "$note"
    continue
  fi

  report_name=$(basename "$report")
  row_count=$(wc -l < "$report" 2>/dev/null || echo "—")
  size_kb=$(du -k "$report" | awk '{print $1"KB"}')

  printf "%-20s %-28s %-10s %-8s  %s\n" \
    "$run" "$report_name" "$row_count" "$size_kb" "$note"
done

echo ""
