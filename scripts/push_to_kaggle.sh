#!/usr/bin/env bash
# push_to_kaggle.sh — Push the notebook to Kaggle as a kernel.
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
KERNEL_TITLE="${KERNEL_TITLE:-ISIC2018 XAI Evaluation Pipeline}"
NOTEBOOK="${NOTEBOOK:-XAI_Evaluation_Pipeline_Kaggle.ipynb}"
DATASET_SOURCES="${DATASET_SOURCES:-sani84/isic-2018-classification}"
ENABLE_GPU="${ENABLE_GPU:-true}"
ENABLE_TPU="${ENABLE_TPU:-false}"
ENABLE_INTERNET="${ENABLE_INTERNET:-true}"
MACHINE_SHAPE="${MACHINE_SHAPE:-NvidiaTeslaT4}"
META_DIR="scripts/kernel-meta"

if [[ ! -f "$NOTEBOOK" ]]; then
  echo "Notebook not found: $NOTEBOOK" >&2
  echo "Set NOTEBOOK in scripts/pipeline.env or create the file first." >&2
  exit 1
fi

DATASET_JSON="[]"
if [[ -n "$DATASET_SOURCES" ]]; then
  DATASET_JSON=$(echo "$DATASET_SOURCES" | awk -F',' '
    BEGIN { printf("[") }
    {
      for (i = 1; i <= NF; i++) {
        gsub(/^[ \t]+|[ \t]+$/, "", $i)
        if (length($i) > 0) {
          if (c++ > 0) printf(", ")
          printf("\"%s\"", $i)
        }
      }
    }
    END { printf("]") }
  ')
fi

echo "── Preparing kernel metadata..."
mkdir -p "$META_DIR"

cat > "$META_DIR/kernel-metadata.json" <<EOF
{
  "id": "${KAGGLE_USER}/${KERNEL_SLUG}",
  "title": "${KERNEL_TITLE}",
  "code_file": "../../${NOTEBOOK}",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": ${ENABLE_GPU},
  "enable_tpu": ${ENABLE_TPU},
  "enable_internet": ${ENABLE_INTERNET},
  "machine_shape": "${MACHINE_SHAPE}",
  "dataset_sources": ${DATASET_JSON},
  "competition_sources": [],
  "kernel_sources": []
}
EOF

echo "── Pushing kernel: ${KAGGLE_USER}/${KERNEL_SLUG}"
kaggle kernels push -p "$META_DIR"
echo "── Done. Monitor with: kaggle kernels status ${KAGGLE_USER}/${KERNEL_SLUG}"
