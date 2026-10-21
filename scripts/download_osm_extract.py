#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

CITY_TO_URL = {
    # Geofabrik extracts (stable + widely used in research)
    "nyc": "https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf",
    "chicago": "https://download.geofabrik.de/north-america/us/illinois-latest.osm.pbf",
    # NorCal is much smaller than full California and covers SF well
    "san_francisco": (
        "https://download.geofabrik.de/north-america/us/california/" "norcal-latest.osm.pbf"
    ),
}


def download(url: str, out_path: Path, force: bool = False) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"OK already exists: {out_path}")
        return out_path

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = out_path.with_suffix(out_path.suffix + ".part")

        with (
            tmp.open("wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    tmp.rename(out_path)
    print(f"✅ Downloaded: {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=sorted(CITY_TO_URL.keys()))
    ap.add_argument("--output", default="data/raw/osm", help="Output directory")
    ap.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = ap.parse_args()

    url = CITY_TO_URL[args.city]
    out_dir = Path(args.output)
    out_path = out_dir / f"{args.city}.osm.pbf"
    download(url, out_path, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
