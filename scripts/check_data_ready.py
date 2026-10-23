#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local package import works when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from terralens.utils.status import (  # noqa: E402 - script bootstrap
    DataChecker,
    DataStatus,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check data availability for a city")
    parser.add_argument("--city", required=True, help="City code (nyc, chicago, san_francisco)")
    args = parser.parse_args()

    checker = DataChecker()
    status = checker.check_all(args.city)

    print(f"\nData Status for {args.city.upper()}")
    print("=" * 50)
    for k, v in status.items():
        icon = (
            "OK"
            if v == DataStatus.AVAILABLE
            else ("PARTIAL" if v == DataStatus.PARTIAL else "MISSING")
        )
        label = "311 data (optional)" if k == "safety" else k
        print(f"{icon} {label:<20} : {v.value}")
    print("=" * 50)

    geo_ready = checker.is_ready_geo_only(args.city)
    print(f"\nGeo-only runnable: {'✅ YES' if geo_ready else '❌ NO'}")

    return 0 if geo_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
