#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

NYC_311_ENDPOINT = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"


def fetch_page(limit: int, offset: int, where: str) -> list:
    params = {
        "$select": "unique_key,created_date,complaint_type,descriptor,latitude,longitude",
        "$where": where,
        "$limit": str(limit),
        "$offset": str(offset),
    }
    r = requests.get(NYC_311_ENDPOINT, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12, help="How many months back to fetch")
    ap.add_argument("--output", default="data/raw/nyc311", help="Output directory")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"nyc311_last_{args.months}mo.parquet"
    if out_path.exists() and not args.force:
        print(f"✅ Exists (skip): {out_path}")
        return 0

    # Only rows with coordinates + last N months
    # Socrata prefers concrete timestamps over SQL date functions
    start = (datetime.utcnow() - timedelta(days=30 * args.months)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start_iso = start.strftime("%Y-%m-%dT%H:%M:%S.000")
    where = (
        f"created_date >= '{start_iso}' "
        "AND latitude IS NOT NULL AND longitude IS NOT NULL "
        "AND latitude != '0' AND longitude != '0'"
    )

    limit = 50000
    offset = 0
    rows = []
    while True:
        page = fetch_page(limit, offset, where)
        if not page:
            break
        rows.extend(page)
        offset += limit
        print(f"Fetched {len(rows)} rows...")

        # safety stop (~2M rows max)
        if offset > 2000000:
            break

    df = pd.DataFrame(
        rows,
        columns=[
            "unique_key",
            "created_date",
            "complaint_type",
            "descriptor",
            "latitude",
            "longitude",
        ],
    )
    # Coerce numeric coords
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "complaint_type"])

    df.to_parquet(out_path, index=False)
    print(f"✅ Saved NYC 311 -> {out_path} (rows={len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
