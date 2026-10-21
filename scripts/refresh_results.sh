#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${TERRALENS_PYTHON:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then
    TERRALENS_PYTHON="$PWD/.venv/bin/python"
  elif [[ -x venv/bin/python ]]; then
    TERRALENS_PYTHON="$PWD/venv/bin/python"
  else
    TERRALENS_PYTHON="$(command -v python3)"
  fi
fi

echo "==> Running experiments..."
"$TERRALENS_PYTHON" experiments/run_all.py
"$TERRALENS_PYTHON" experiments/summarize_results.py

echo "==> Generating markdown + figures..."
"$TERRALENS_PYTHON" scripts/generate_results_md.py
"$TERRALENS_PYTHON" scripts/make_figures.py

echo "✅ Done. See:"
echo " - outputs/results/RESULTS.md"
echo " - outputs/figures/"
