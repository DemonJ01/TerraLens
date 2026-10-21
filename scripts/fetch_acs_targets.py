#!/usr/bin/env python3
"""
Thin wrapper to run download_acs_targets.py without requiring scripts/ to be a package.
Usage:
  python scripts/fetch_acs_targets.py --city nyc --year 2021
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    target = Path(__file__).parent / "download_acs_targets.py"
    if not target.exists():
        raise SystemExit(f"❌ Missing: {target}")
    # Make relative imports (if any) behave sanely
    sys.path.insert(0, str(Path.cwd()))
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
