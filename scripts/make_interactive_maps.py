#!/usr/bin/env python3
"""Export any cached TerraLens prediction layer to an interactive HTML map."""

from __future__ import annotations

import argparse
import sys
from html import escape
from pathlib import Path

from branca.element import Element

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from terralens.data.dashboard import load_predictions  # noqa: E402 - script bootstrap
from terralens.utils.maps import build_prediction_map  # noqa: E402 - script bootstrap

MODELS = ["ridge", "xgboost", "extra_trees", "hist_gb", "random_forest", "mlp", "elasticnet"]
TARGETS = [
    "median_income",
    "median_rent",
    "median_home_value",
    "poverty_rate",
    "unemployment_rate",
    "bachelors_plus_rate",
]
LAYERS = {"prediction": "y_pred", "error": "error", "ground_truth": "y_true"}


def export_map(
    cached: Path,
    output: Path,
    title: str,
    layer: str = "prediction",
    mode: str = "Cluster",
    clip: bool = True,
) -> Path:
    """Render the same validated rows and map behavior as the live dashboard."""
    predictions = load_predictions(cached)
    map_view = build_prediction_map(predictions, LAYERS[layer], mode, clip=clip)
    map_view.get_root().header.add_child(Element(f"<title>{escape(title)}</title>"))
    output.parent.mkdir(parents=True, exist_ok=True)
    map_view.save(str(output))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True, help="City key used in the cached filename")
    parser.add_argument("--model", default="xgboost", choices=MODELS)
    parser.add_argument(
        "--feature_set", "--feature-set", default="osm_only", choices=["osm_only", "osm_311"]
    )
    parser.add_argument("--target", default="median_income", choices=TARGETS)
    parser.add_argument("--layer", default="prediction", choices=LAYERS)
    parser.add_argument("--mode", default="Cluster", choices=["Cluster", "Points", "Heatmap"])
    parser.add_argument("--no-clip", action="store_true", help="Use the complete value range")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "outputs/preds")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "docs/maps")
    args = parser.parse_args()

    if not args.city or any(char in args.city for char in ("/", "\\")) or args.city in {".", ".."}:
        parser.error("--city must be a city key, without path separators")
    filename = f"{args.city}__{args.model}__{args.feature_set}__{args.target}.parquet"
    cached = args.cache_dir / filename
    if not cached.is_file():
        parser.error(
            f"Missing cached predictions: {cached}. Generate this city/model/target cache first."
        )
    # Preserve the established filenames used by links to the default income maps.
    measure = "income" if args.target == "median_income" else args.target
    suffix = (
        ""
        if args.layer == "prediction" and args.mode == "Cluster"
        else f"_{args.layer}_{args.mode.lower()}"
    )
    output = (
        args.output_dir / f"{args.city}_{args.model}_{args.feature_set}_{measure}{suffix}_map.html"
    )
    title = (
        f"TerraLens | {args.city.upper()} | {args.target.replace('_', ' ')} | "
        f"{args.layer.replace('_', ' ')}"
    )
    try:
        export_map(cached, output, title, args.layer, args.mode, clip=not args.no_clip)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"Could not export map: {exc}")
    print(f"Wrote interactive TerraLens map: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
