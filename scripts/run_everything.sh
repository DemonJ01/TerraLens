#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash scripts/run_everything.sh [YEAR] [--skip-downloads]"
  echo "Build real city datasets, train models, and cache dashboard predictions."
  echo "Downloads Census data and OSM extracts; uses osmium-tool or GDAL's OSM driver."
  echo "--skip-downloads uses existing raw data/features instead."
  exit 0
fi

if [[ -z "${TERRALENS_PYTHON:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then
    TERRALENS_PYTHON="$PWD/.venv/bin/python"
  elif [[ -x venv/bin/python ]]; then
    TERRALENS_PYTHON="$PWD/venv/bin/python"
  else
    TERRALENS_PYTHON="$(command -v python3)"
  fi
fi
export TERRALENS_PYTHON
YEAR=2021
SKIP_DOWNLOADS=false
for argument in "$@"; do
  if [[ "$argument" == "--skip-downloads" ]]; then
    SKIP_DOWNLOADS=true
  elif [[ "$argument" =~ ^[0-9]{4}$ ]]; then
    YEAR="$argument"
  else
    echo "Unknown argument: $argument" >&2
    exit 1
  fi
done

PREPARATION_FLAGS=(--year "$YEAR" --models xgboost,extra_trees,hist_gb,ridge)
if [[ "$SKIP_DOWNLOADS" == true ]]; then
  PREPARATION_FLAGS+=(--skip-downloads)
fi
for city in nyc chicago san_francisco; do
  "$TERRALENS_PYTHON" scripts/prepare_city.py --city "$city" "${PREPARATION_FLAGS[@]}"
done

if [[ -f data/processed/nyc_dataset_311.parquet ]]; then
  "$TERRALENS_PYTHON" scripts/cache_predictions.py --city nyc --feature_sets osm_311
fi

bash scripts/refresh_results.sh

"$TERRALENS_PYTHON" scripts/make_interactive_maps.py --city nyc --model xgboost --feature_set osm_only
if [[ -f data/processed/nyc_dataset_311.parquet ]]; then
  "$TERRALENS_PYTHON" scripts/make_interactive_maps.py --city nyc --model xgboost --feature_set osm_311
fi

mkdir -p docs/figures
cp -f outputs/figures/*.png docs/figures/ 2>/dev/null || true

echo "✅ Done. Open:"
echo " - docs/maps/*.html"
echo " - docs/figures/*.png"
