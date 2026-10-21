#!/usr/bin/env python3
"""
Download tract-level ACS targets for a city/year and save to parquet.

Writes:
  data/raw/census/{city}_targets_{year}.parquet

Targets (columns):
  - median_income
  - median_rent
  - median_home_value
  - poverty_rate
  - unemployment_rate
  - bachelors_plus_rate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

CITY_TO_COUNTY_STATE = {
    "nyc": [
        ("005", "36"),
        ("047", "36"),
        ("061", "36"),
        ("081", "36"),
        ("085", "36"),
    ],  # Bronx, Kings, New York, Queens, Richmond
    "chicago": [("031", "17")],  # Cook County
    "san_francisco": [("075", "06")],  # San Francisco County
}

# ACS 5-year profile / subject / detailed tables
# We use ACS5 "B" tables for tract detail.
ACS_VARS = {
    "median_income": "B19013_001E",
    "median_rent": "B25064_001E",
    "median_home_value": "B25077_001E",
    "poverty_num": "B17001_002E",
    "poverty_den": "B17001_001E",
    "unemp_num": "B23025_005E",
    "unemp_den": "B23025_003E",
    "ba_plus_num": "B15003_022E",  # Bachelor's
    "ba_plus_num2": "B15003_023E",  # Master's
    "ba_plus_num3": "B15003_024E",  # Professional
    "ba_plus_num4": "B15003_025E",  # Doctorate
    "ba_plus_den": "B15003_001E",
}

SENTINEL_BAD = {-666666666, -555555555, -333333333}


def fetch_county_tracts(year: int, state_fips: str, county_fips: str) -> pd.DataFrame:
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    get_vars = [
        "NAME",
        "state",
        "county",
        "tract",
        ACS_VARS["median_income"],
        ACS_VARS["median_rent"],
        ACS_VARS["median_home_value"],
        ACS_VARS["poverty_num"],
        ACS_VARS["poverty_den"],
        ACS_VARS["unemp_num"],
        ACS_VARS["unemp_den"],
        ACS_VARS["ba_plus_num"],
        ACS_VARS["ba_plus_num2"],
        ACS_VARS["ba_plus_num3"],
        ACS_VARS["ba_plus_num4"],
        ACS_VARS["ba_plus_den"],
    ]
    url = (
        f"{base}?get={','.join(get_vars)}"
        f"&for=tract:*&in=state:{state_fips}%20county:{county_fips}"
    )
    with urlopen(url) as r:
        data = json.loads(r.read().decode("utf-8"))

    header = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=header)

    # numeric cast
    num_cols = [c for c in df.columns if c.endswith("E")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # tract_id = state+county+tract (11 digits)
    df["tract_id"] = (
        df["state"].astype(str) + df["county"].astype(str) + df["tract"].astype(str)
    ).astype(str)
    return df


def clean_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # replace sentinel bad values with NaN
    for k in ["median_income", "median_rent", "median_home_value"]:
        col = ACS_VARS[k]
        df[col] = df[col].where(~df[col].isin(SENTINEL_BAD), np.nan)

    # top-coding sometimes appears (e.g., 250001), we keep it as valid
    # drop impossible negatives
    for k in ["median_income", "median_rent", "median_home_value"]:
        col = ACS_VARS[k]
        df[col] = df[col].where(df[col].isna() | (df[col] >= 0), np.nan)

    # rates
    pov = pd.to_numeric(df[ACS_VARS["poverty_num"]], errors="coerce") / pd.to_numeric(
        df[ACS_VARS["poverty_den"]], errors="coerce"
    )
    unemp = pd.to_numeric(df[ACS_VARS["unemp_num"]], errors="coerce") / pd.to_numeric(
        df[ACS_VARS["unemp_den"]], errors="coerce"
    )
    ba_plus_num = (
        pd.to_numeric(df[ACS_VARS["ba_plus_num"]], errors="coerce")
        + pd.to_numeric(df[ACS_VARS["ba_plus_num2"]], errors="coerce")
        + pd.to_numeric(df[ACS_VARS["ba_plus_num3"]], errors="coerce")
        + pd.to_numeric(df[ACS_VARS["ba_plus_num4"]], errors="coerce")
    )
    ba_plus = ba_plus_num / pd.to_numeric(df[ACS_VARS["ba_plus_den"]], errors="coerce")

    out = pd.DataFrame(
        {
            "tract_id": df["tract_id"].astype(str),
            "median_income": df[ACS_VARS["median_income"]],
            "median_rent": df[ACS_VARS["median_rent"]],
            "median_home_value": df[ACS_VARS["median_home_value"]],
            "poverty_rate": pov,
            "unemployment_rate": unemp,
            "bachelors_plus_rate": ba_plus,
        }
    )

    # keep sane ranges
    for c in ["poverty_rate", "unemployment_rate", "bachelors_plus_rate"]:
        out[c] = out[c].where(out[c].isna() | ((out[c] >= 0) & (out[c] <= 1)), np.nan)

    # require at least income for Phase-1 compatibility (and most work)
    out = out.dropna(subset=["median_income"]).copy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=sorted(CITY_TO_COUNTY_STATE.keys()))
    ap.add_argument("--year", type=int, default=2021)
    args = ap.parse_args()

    parts = []
    for county_fips, state_fips in CITY_TO_COUNTY_STATE[args.city]:
        parts.append(fetch_county_tracts(args.year, state_fips=state_fips, county_fips=county_fips))
    raw = pd.concat(parts, ignore_index=True)

    targets = clean_targets(raw)

    out_path = Path(f"data/raw/census/{args.city}_targets_{args.year}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(out_path, index=False)

    cols = [c for c in targets.columns if c != "tract_id"]
    print(f"✅ Wrote ACS targets: {out_path} (rows={len(targets)}, cols={len(targets.columns)})")
    print(f"   columns: {['tract_id'] + cols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
