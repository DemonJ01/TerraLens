#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from pathlib import Path as _Path

import yaml

# Ensure repo root is on sys.path when running as a script
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from scripts.utils.census_api import CensusAPI  # noqa: E402 - script bootstrap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fetch_census_data")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch ACS median income labels (tract level)")
    parser.add_argument("--city", required=True, help="City code (nyc, chicago, san_francisco)")
    parser.add_argument(
        "--year", type=int, default=2021, help="ACS 5-year vintage year (default: 2021)"
    )
    parser.add_argument("--output", required=True, help="Output directory (e.g., data/raw/census/)")
    parser.add_argument("--api-key", default=None, help="Optional Census API key")
    parser.add_argument("--force", action="store_true", help="Overwrite if file exists")
    args = parser.parse_args()

    config = yaml.safe_load(Path("configs/cities.yaml").read_text())
    cities = config.get("cities", config)
    if "sf" in cities and "san_francisco" not in cities:
        cities["san_francisco"] = cities["sf"]
    if args.city not in cities:
        logger.error(f"Unknown city '{args.city}'. Valid: {list(cities.keys())}")
        return 1

    cfg = cities[args.city]
    state_fips = str(cfg.get("state_fips", cfg.get("fips_code", ""))).zfill(2)
    counties = cfg.get("counties", cfg.get("county_fips", []))
    if not counties:
        logger.error("No county_fips configured for this city.")
        return 1

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.city}_income_{args.year}.csv"

    if out_path.exists() and not args.force:
        logger.info(f"File exists: {out_path}")
        logger.info("Use --force to overwrite.")
        return 0

    logger.info(f"Fetching ACS median income for {cfg['name']} ({args.city}) year={args.year}")
    client = CensusAPI(api_key=args.api_key)

    try:
        df = client.median_income_by_counties(
            year=args.year, state_fips=state_fips, counties=counties
        )
    except Exception as e:
        logger.exception(f"Failed fetching Census data: {e}")
        return 1

    df.to_csv(out_path, index=False)
    logger.info(f"✅ Saved {len(df)} rows -> {out_path}")

    missing = int(df["median_income"].isna().sum())
    logger.info(f"Missing income: {missing}")
    logger.info(
        f"Income stats: min={df['median_income'].min()} max={df['median_income'].max()} "
        f"median={df['median_income'].median()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
