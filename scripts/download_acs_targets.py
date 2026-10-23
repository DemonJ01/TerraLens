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
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from zipfile import is_zipfile

import numpy as np
import pandas as pd
import requests

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


def download_archive(url: str, destination: Path) -> Path:
    """Download the public Census bulk archive without requiring API credentials."""
    if destination.is_file() and is_zipfile(destination):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        output.write(block)
    except requests.RequestException:
        raise RuntimeError("Could not download the Census bulk archive. Try again later.") from None
    if not is_zipfile(partial):
        raise RuntimeError("Census returned an invalid bulk archive. No targets were written.")
    partial.replace(destination)
    return destination


def fetch_bulk_tracts(city: str, year: int, cache_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read the same ACS detailed-table estimates from official TIGER/ACS files."""
    import pyogrio

    parts, urls = [], []
    states = {state for _, state in CITY_TO_COUNTY_STATE[city]}
    for state in sorted(states):
        filename = f"ACS_{year}_5YR_TRACT_{state}.gdb.zip"
        url = f"https://www2.census.gov/geo/tiger/TIGER_DP/{year}ACS/{filename}"
        archive = download_archive(url, cache_dir / filename)
        urls.append(url)
        source = "/vsizip/" + str(archive.resolve())
        layers = pyogrio.list_layers(source)[:, 0]
        by_subject = {}
        for variable in ACS_VARS.values():
            table, column = re.fullmatch(r"(B\d+)_(\d+)E", variable).groups()
            by_subject.setdefault(table[1:3], {})[f"{table}e{int(column)}"] = variable
        merged = None
        for subject, columns in by_subject.items():
            candidates = [layer for layer in layers if layer.startswith(f"X{subject}_")]
            if len(candidates) != 1:
                raise RuntimeError(f"ACS archive is missing subject table X{subject}.")
            fields = pyogrio.read_info(source, layer=candidates[0])["fields"]
            if not {"GEOID", *columns}.issubset(fields):
                raise RuntimeError(f"ACS archive is missing required fields in {candidates[0]}.")
            frame = pyogrio.read_dataframe(
                source, layer=candidates[0], columns=["GEOID", *columns], read_geometry=False
            ).rename(columns=columns)
            frame["tract_id"] = frame["GEOID"].astype(str).str.extract(r"(\d{11})$")[0]
            prefixes = [state + county for county, st in CITY_TO_COUNTY_STATE[city] if st == state]
            frame = frame.loc[frame.tract_id.str[:5].isin(prefixes)].drop(columns="GEOID")
            if frame.tract_id.isna().any() or frame.tract_id.duplicated().any():
                raise RuntimeError("ACS archive contains invalid or duplicate tract IDs.")
            merged = (
                frame
                if merged is None
                else merged.merge(frame, on="tract_id", how="outer", validate="one_to_one")
            )
        parts.append(merged)
    return pd.concat(parts, ignore_index=True), urls


def fetch_county_tracts(
    year: int, state_fips: str, county_fips: str, api_key: str | None = None
) -> pd.DataFrame:
    key = (api_key or os.environ.get("CENSUS_API_KEY", "")).strip()
    if not key:
        raise RuntimeError(
            "Census data queries require an API key. "
            "Set CENSUS_API_KEY before running this script. "
            "Request a free key at https://api.census.gov/data/key_signup.html."
        )
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    get_vars = [
        "NAME",
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
    params = {
        "get": ",".join(get_vars),
        "for": "tract:*",
        "in": f"state:{str(state_fips).zfill(2)} county:{str(county_fips).zfill(3)}",
        "key": key,
    }
    try:
        response = requests.get(base, params=params, timeout=60)
    except requests.RequestException:
        # Request exceptions can contain the URL and its API key.
        raise RuntimeError(
            "Could not reach the Census API. Check the connection and try again."
        ) from None
    if response.status_code in {401, 403}:
        raise RuntimeError(
            "Census rejected the API key. Check that CENSUS_API_KEY contains an activated key."
        )
    if not response.ok:
        raise RuntimeError(
            f"Census API request failed (HTTP {response.status_code}). "
            "Check the dataset year and CENSUS_API_KEY, then try again."
        )
    try:
        data = response.json()
    except ValueError:
        # Authentication and maintenance pages sometimes use HTTP 200.
        raise RuntimeError(
            "Census returned an unexpected response. Check that CENSUS_API_KEY is activated "
            "and that the dataset year is available."
        ) from None

    required = set(get_vars) | {"state", "county", "tract"}
    if (
        not isinstance(data, list)
        or not data
        or not isinstance(data[0], list)
        or not all(isinstance(name, str) for name in data[0])
        or not required.issubset(data[0])
        or len(set(data[0])) != len(data[0])
        or not all(isinstance(row, list) and len(row) == len(data[0]) for row in data[1:])
    ):
        raise RuntimeError("Census returned an invalid data table. No target file was written.")

    header = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=header)

    # numeric cast
    num_cols = [c for c in df.columns if c.endswith("E")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # tract_id = state+county+tract (11 digits)
    df["tract_id"] = (
        df["state"].astype(str).str.zfill(2)
        + df["county"].astype(str).str.zfill(3)
        + df["tract"].astype(str).str.zfill(6)
    ).astype(str)
    return df


def clean_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for variable in ACS_VARS.values():
        values = pd.to_numeric(df[variable], errors="coerce")
        df[variable] = values.where(np.isfinite(values) & (values >= 0))

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

    # Income is required by the training and evaluation scripts.
    out = out.dropna(subset=["median_income"]).copy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=sorted(CITY_TO_COUNTY_STATE.keys()))
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument(
        "--source",
        choices=["auto", "census", "tiger-gdb"],
        default="auto",
        help="Use the Census API with CENSUS_API_KEY, or public bulk archives without a key",
    )
    ap.add_argument("--cache-dir", type=Path, default=Path("data/raw/census/_acs_cache"))
    args = ap.parse_args()

    parts = []
    source = args.source
    if source == "auto":
        source = "census" if os.environ.get("CENSUS_API_KEY", "").strip() else "tiger-gdb"
    try:
        if source == "tiger-gdb":
            raw, urls = fetch_bulk_tracts(args.city, args.year, args.cache_dir)
        else:
            for county_fips, state_fips in CITY_TO_COUNTY_STATE[args.city]:
                parts.append(
                    fetch_county_tracts(args.year, state_fips=state_fips, county_fips=county_fips)
                )
            raw = pd.concat(parts, ignore_index=True)
            urls = [f"https://api.census.gov/data/{args.year}/acs/acs5"]
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None

    targets = clean_targets(raw)
    if targets.empty:
        raise SystemExit("No usable ACS target rows were found. No target file was written.")

    out_path = Path(f"data/raw/census/{args.city}_targets_{args.year}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(out_path, index=False)
    out_path.with_suffix(".source.json").write_text(
        json.dumps(
            {
                "source": source,
                "urls": urls,
                "year": args.year,
                "release": f"ACS {args.year} five-year estimates",
                "city": args.city,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "rows": len(targets),
                "variables": ACS_VARS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    cols = [c for c in targets.columns if c != "tract_id"]
    print(f"✅ Wrote ACS targets: {out_path} (rows={len(targets)}, cols={len(targets.columns)})")
    print(f"   columns: {['tract_id'] + cols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
