#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import requests

# Ensure repo root on sys.path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from terralens.utils.config import load_yaml  # noqa: E402 - script bootstrap


def download_zip(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, help="nyc, chicago, san_francisco")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--output", default="data/raw/census", help="Output directory")
    args = ap.parse_args()

    config = load_yaml("configs/cities.yaml")
    cities = config.get("cities", config)
    if "sf" in cities and "san_francisco" not in cities:
        cities["san_francisco"] = cities["sf"]
    if args.city not in cities:
        raise SystemExit(f"Unknown city '{args.city}'. Valid: {list(cities.keys())}")

    cfg = cities[args.city]
    state_fips = str(cfg.get("state_fips", cfg.get("fips_code", ""))).zfill(2)
    counties = [str(c).zfill(3) for c in cfg.get("counties", cfg.get("county_fips", []))]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.city}_tracts_{args.year}.gpkg"

    # ✅ Correct TIGER/Line pattern: state-level tract zip
    url = (
        f"https://www2.census.gov/geo/tiger/TIGER{args.year}/TRACT/"
        f"tl_{args.year}_{state_fips}_tract.zip"
    )
    print(f"Downloading: {url}")

    zbytes = download_zip(url)

    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        # find the .shp inside the zip
        shp_name = [n for n in zf.namelist() if n.endswith(".shp")]
        if not shp_name:
            raise RuntimeError("No .shp found in TIGER tract zip")
        shp_name = shp_name[0]

        # geopandas can read from zip via vsizip if path-like; easiest is extract to temp dir
        extract_dir = out_dir / f"_{args.city}_tiger_extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(extract_dir)

    shp = list(extract_dir.glob("*.shp"))
    if not shp:
        raise RuntimeError(f"No .shp found after extracting: {extract_dir}")
    gdf = gpd.read_file(shp[0])

    # Filter to only requested counties (NYC uses 5 counties)
    gdf = gdf[gdf["COUNTYFP"].astype(str).str.zfill(3).isin(counties)].copy()

    # Standardize tract id
    gdf["tract_id"] = gdf["GEOID"].astype(str)
    gdf = gdf[["tract_id", "STATEFP", "COUNTYFP", "TRACTCE", "NAME", "geometry"]].reset_index(
        drop=True
    )

    # Save to GeoPackage (robust on macOS)
    gdf.to_file(out_path, layer="tracts", driver="GPKG")
    print(f"✅ Saved {len(gdf)} tracts -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
