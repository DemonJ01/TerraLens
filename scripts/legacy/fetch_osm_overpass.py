#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict

import osmnx as ox
import pandas as pd
from osmnx._errors import InsufficientResponseError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fetch_osm_features")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_points(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    required = {"lat", "lon", "tract_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must include {required}, got {set(df.columns)}")
    return df[["tract_id", "lat", "lon"]].copy()


def _entropy_from_counts(counts: pd.Series) -> float:
    import math

    counts = counts[counts > 0]
    if counts.empty:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * p.apply(lambda x: 0.0 if x <= 0 else math.log(x))).sum())


def _pick_category_column(gdf: pd.DataFrame) -> str | None:
    for c in ["amenity", "shop", "leisure", "tourism"]:
        if c in gdf.columns:
            return c
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch OSM features around sampled points (fast, cached, resumable)."
    )
    parser.add_argument("--dataset", required=True, help="Parquet with lat/lon points")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--radius_m", type=int, default=500, help="Query radius in meters")
    parser.add_argument("--max_points", type=int, default=300, help="Limit number of points")
    parser.add_argument("--sleep_s", type=float, default=0.1, help="Sleep between queries")
    parser.add_argument(
        "--checkpoint_every", type=int, default=25, help="Save progress every N points"
    )
    parser.add_argument("--resume", action="store_true", help="Resume if partial output exists")
    parser.add_argument("--force", action="store_true", help="Overwrite output")
    args = parser.parse_args()

    out_path = Path(args.output)
    _ensure_dir(out_path.parent)

    # OSMnx settings
    ox.settings.requests_timeout = 120
    ox.settings.log_console = False
    ox.settings.use_cache = True

    tags = {"amenity": True, "shop": True, "leisure": True, "tourism": True}

    pts = _load_points(Path(args.dataset)).head(args.max_points).copy()

    # Resume logic
    existing: Dict[str, Dict[str, Any]] = {}
    if out_path.exists() and args.resume and not args.force:
        try:
            prev = pd.read_parquet(out_path)
            for r in prev.itertuples(index=False):
                existing[str(r.tract_id)] = {
                    "tract_id": r.tract_id,
                    "poi_count_500m": float(r.poi_count_500m),
                    "poi_entropy": float(r.poi_entropy),
                }
            logger.info(f"Resuming: loaded {len(existing)} rows from {out_path}")
        except Exception as e:
            logger.warning(f"Could not resume from {out_path}: {e}")

    if out_path.exists() and not args.resume and not args.force:
        logger.info(f"Output exists: {out_path}. Use --resume or --force.")
        return 0

    rows = list(existing.values())
    done_ids = set(existing.keys())

    failures = 0
    start = time.time()
    logger.info(f"Querying OSM around up to {len(pts)} points, radius={args.radius_m}m (cached).")

    for i, r in enumerate(pts.itertuples(index=False), 1):
        tract_id = str(r.tract_id)
        if tract_id in done_ids:
            continue

        lat = float(r.lat)
        lon = float(r.lon)

        try:
            gdf = ox.features_from_point((lat, lon), dist=args.radius_m, tags=tags).reset_index()

            # If no features, that is OK (poi_count = 0)
            poi_count = float(len(gdf))
            cat_col = _pick_category_column(gdf)

            if poi_count == 0 or cat_col is None:
                poi_entropy = 0.0
            else:
                vc = gdf[cat_col].fillna("unknown").astype(str).value_counts()
                poi_entropy = _entropy_from_counts(vc)

            rows.append(
                {"tract_id": tract_id, "poi_count_500m": poi_count, "poi_entropy": poi_entropy}
            )
            done_ids.add(tract_id)

        except InsufficientResponseError:
            # "No matching features" → not an error for us
            rows.append({"tract_id": tract_id, "poi_count_500m": 0.0, "poi_entropy": 0.0})
            done_ids.add(tract_id)

        except KeyboardInterrupt:
            logger.warning("Interrupted by user. Saving progress...")
            pd.DataFrame(rows).drop_duplicates(subset=["tract_id"]).to_parquet(
                out_path, index=False
            )
            raise

        except Exception as e:
            failures += 1
            rows.append({"tract_id": tract_id, "poi_count_500m": 0.0, "poi_entropy": 0.0})
            done_ids.add(tract_id)
            logger.warning(f"Failed tract_id={tract_id}: {repr(e)}")

        if len(done_ids) % args.checkpoint_every == 0:
            pd.DataFrame(rows).drop_duplicates(subset=["tract_id"]).to_parquet(
                out_path, index=False
            )
            elapsed = time.time() - start
            logger.info(
                f"Checkpoint: {len(done_ids)}/{len(pts)} done | "
                f"failures={failures} | elapsed={elapsed:.1f}s"
            )

        time.sleep(args.sleep_s)

    feat = pd.DataFrame(rows).drop_duplicates(subset=["tract_id"])
    feat.to_parquet(out_path, index=False)
    logger.info(f"✅ Saved geo features -> {out_path} rows={len(feat)} failures={failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
