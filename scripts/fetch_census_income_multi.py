#!/usr/bin/env python3
"""
Fetch ACS 5-year median household income for multiple cities.

- Variable: B19013_001E (median household income)
- Dataset: ACS 5-year (acs/acs5)
- Output: data/raw/census/{city}_income_{year}.parquet with columns:
    tract_id, median_income

Usage:
  python3 scripts/fetch_census_income_multi.py --cities chicago sf --year 2021
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests
import yaml


def load_config() -> dict:
    cfg_path = Path("configs/cities.yaml")
    if not cfg_path.exists():
        raise SystemExit("configs/cities.yaml not found.")
    return yaml.safe_load(cfg_path.read_text())


def fetch_county(year: int, state_fips: str, county_fips: str) -> pd.DataFrame:
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "B19013_001E,NAME",
        "for": "tract:*",
        "in": f"state:{state_fips}+county:{county_fips}",
    }
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df.rename(columns={"B19013_001E": "median_income"}, inplace=True)
    df["median_income"] = pd.to_numeric(df["median_income"], errors="coerce")
    df["tract_id"] = (
        df["state"].astype(str).str.zfill(2)
        + df["county"].astype(str).str.zfill(3)
        + df["tract"].astype(str).str.zfill(6)
    )
    return df[["tract_id", "median_income", "NAME"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", required=True)
    ap.add_argument("--year", type=int, default=2021)
    args = ap.parse_args()

    cfg = load_config()
    cities = cfg.get("cities", {})
    if "sf" in cities and "san_francisco" not in cities:
        cities["san_francisco"] = cities["sf"]

    out_dir = Path("data/raw/census")
    out_dir.mkdir(parents=True, exist_ok=True)

    for city in args.cities:
        if city not in cities:
            raise SystemExit(f"Unknown city '{city}'. Check configs/cities.yaml")

        meta = cities[city]
        state_fips = str(meta["state_fips"]).zfill(2)
        counties = [str(c).zfill(3) for c in meta.get("counties", [])]
        if not counties:
            raise SystemExit(f"{city}: no counties configured in configs/cities.yaml")

        print(f"\n=== {city.upper()} income (ACS {args.year}) ===")
        parts = []
        for cty in counties:
            print(f"  Fetching county {state_fips}{cty} ...")
            parts.append(fetch_county(args.year, state_fips, cty))

        df = pd.concat(parts, ignore_index=True)
        before = len(df)

        df["median_income"] = pd.to_numeric(df["median_income"], errors="coerce")

        # Drop NaN + invalid/sentinel + non-positive values
        df = df.dropna(subset=["median_income"]).copy()
        df = df[df["median_income"] > 0].copy()
        df = df[df["median_income"] != -666666666].copy()

        print(f"  Dropped invalid/missing income: {before} -> {len(df)}")

        out_path = out_dir / f"{city}_income_{args.year}.parquet"
        df[["tract_id", "median_income"]].to_parquet(out_path, index=False)
        print(f"✅ Saved: {out_path} (rows={len(df)})")
        print(f"   income: min={df['median_income'].min():.0f} max={df['median_income'].max():.0f}")


if __name__ == "__main__":
    main()
