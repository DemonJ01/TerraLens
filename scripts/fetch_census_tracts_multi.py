#!/usr/bin/env python3
"""
Fetch census tract geometries for multiple cities (TIGER/Line).

Important: TRACT files are STATE-BASED in TIGER/Line (not county-based).
We download tl_<year>_<STATE>_tract.zip then filter to the desired counties.

Usage:
  python3 scripts/fetch_census_tracts_multi.py --cities nyc chicago sf --year 2021
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import is_zipfile

import geopandas as gpd
import requests
import yaml

TIGER_BASE = "https://www2.census.gov/geo/tiger"


def load_config() -> dict:
    cfg_path = Path("configs/cities.yaml")
    if not cfg_path.exists():
        raise SystemExit("configs/cities.yaml not found.")
    return yaml.safe_load(cfg_path.read_text())


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(out_path.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with partial.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        if not is_zipfile(partial):
            raise RuntimeError(
                "Census returned an invalid TIGER archive. No tract cache was saved."
            )
        partial.replace(out_path)
    finally:
        partial.unlink(missing_ok=True)


def fetch_state_tracts(year: int, state_fips: str, cache_dir: Path) -> Path:
    """
    Download the state-based tract zip (tl_<year>_<STATE>_tract.zip).
    """
    state_fips = str(state_fips).zfill(2)
    url = f"{TIGER_BASE}/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
    out_zip = cache_dir / f"tl_{year}_{state_fips}_tract.zip"
    if out_zip.is_file() and is_zipfile(out_zip):
        return out_zip

    print(f"  Downloading: {url}")
    download(url, out_zip)
    return out_zip


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch TIGER tract geometries for multiple cities.")
    ap.add_argument("--cities", nargs="+", required=True)
    ap.add_argument("--year", type=int, default=2021)
    args = ap.parse_args()

    cfg = load_config()
    cities = cfg.get("cities", {})
    if "sf" in cities and "san_francisco" not in cities:
        cities["san_francisco"] = cities["sf"]

    out_dir = Path("data/raw/census")
    cache_dir = out_dir / "_tiger_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for city in args.cities:
        if city not in cities:
            raise SystemExit(f"Unknown city '{city}'. Check configs/cities.yaml")

        meta = cities[city]
        state_fips = str(meta["state_fips"]).zfill(2)
        counties = [str(c).zfill(3) for c in meta.get("counties", [])]

        print(f"\n=== {city.upper()} tracts (TIGER {args.year}) ===")
        zpath = fetch_state_tracts(args.year, state_fips, cache_dir)

        # Read the shapefile directly from the zip
        gdf = gpd.read_file(f"zip://{zpath}")

        # Standard TIGER columns: STATEFP, COUNTYFP, TRACTCE, GEOID
        if "GEOID" not in gdf.columns:
            raise SystemExit(f"{city}: GEOID column missing in TIGER tract file.")

        # Filter to requested counties (if provided)
        if counties:
            if "COUNTYFP" not in gdf.columns:
                raise SystemExit(f"{city}: COUNTYFP missing; cannot filter counties.")
            before = len(gdf)
            gdf = gdf[gdf["COUNTYFP"].astype(str).str.zfill(3).isin(counties)].copy()
            print(f"  Filtered counties {counties}: {before} -> {len(gdf)} tracts")

        gdf["tract_id"] = gdf["GEOID"].astype(str)

        out_path = out_dir / f"{city}_tracts_{args.year}.gpkg"
        gdf.to_file(out_path, driver="GPKG")
        print(f"✅ Saved: {out_path} (rows={len(gdf)})")


if __name__ == "__main__":
    main()
