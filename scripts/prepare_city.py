#!/usr/bin/env python3
"""Download public city inputs, build measured features, and cache held-out predictions."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def preparation_steps(city: str, year: int, models: str, skip_downloads=False, root=ROOT):
    census = root / "data/raw/census"
    features = root / "data/features" / f"{city}_geo_features.parquet"
    osm = root / "data/raw/osm" / f"{city}.osm.pbf"
    tracts = census / f"{city}_tracts_{year}.gpkg"
    targets = census / f"{city}_targets_{year}.parquet"
    if skip_downloads:
        required = [tracts, targets, *([] if features.is_file() else [osm])]
        missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
        if missing:
            raise ValueError("Missing local inputs: " + ", ".join(missing))
    steps = []
    if not tracts.is_file():
        steps.append(["fetch_census_tracts_multi.py", "--cities", city, "--year", str(year)])
    if not targets.is_file():
        steps.append(["download_acs_targets.py", "--city", city, "--year", str(year)])
    if not features.is_file():
        if not osm.is_file():
            steps.append(["download_osm_extract.py", "--city", city])
        steps.append(
            [
                "compute_osm_features_from_pbf.py",
                "--city",
                city,
                "--pbf",
                str(osm),
                "--tracts",
                str(tracts),
                "--output",
                str(features),
            ]
        )
    steps.extend(
        [
            [
                "build_dataset_multi_city.py",
                "--city",
                city,
                "--year",
                str(year),
                "--output",
                f"data/processed/{city}_dataset.parquet",
            ],
            [
                "cache_predictions.py",
                "--city",
                city,
                "--models",
                models,
                "--feature_sets",
                "osm_only",
                "--targets",
                "all",
            ],
        ]
    )
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=["nyc", "chicago", "san_francisco"], default="nyc")
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--models", default="ridge,extra_trees,hist_gb")
    parser.add_argument("--skip-downloads", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show steps without running them")
    args = parser.parse_args()
    try:
        steps = preparation_steps(args.city, args.year, args.models, args.skip_downloads)
    except ValueError as exc:
        parser.error(str(exc))
    for step in steps:
        command = [sys.executable, str(ROOT / "scripts" / step[0]), *step[1:]]
        print("Running:", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        print("Ready. Open TerraLens and choose Reload results.")


if __name__ == "__main__":
    main()
