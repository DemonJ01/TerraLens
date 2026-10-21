#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def entropy_from_counts(counts: pd.Series) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts.astype(float) / total
    # avoid log(0)
    p = p[p > 0]
    return float(-(p * (p.apply(math.log))).sum())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # EPSG:3857 is fine for NYC-scale lengths; keeps dependencies simple.
    return gdf.to_crs(epsg=3857)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["nyc", "chicago", "san_francisco"])
    ap.add_argument(
        "--pbf", default=None, help="Input .osm.pbf (defaults to data/raw/osm/{city}_clip.osm.pbf)"
    )
    ap.add_argument(
        "--tracts",
        default=None,
        help="Input tracts .gpkg (defaults to data/raw/census/{city}_tracts_2021.gpkg)",
    )
    ap.add_argument(
        "--workdir", default="data/raw/osm/derived", help="Where to write intermediate files"
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output parquet path (defaults to data/features/{city}_osm_features.parquet)",
    )
    ap.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    args = ap.parse_args()

    city = args.city
    pbf = Path(args.pbf) if args.pbf else Path("data/raw/osm") / f"{city}_clip.osm.pbf"
    if not pbf.exists() and args.pbf is None:
        pbf = Path("data/raw/osm") / f"{city}.osm.pbf"
    if not pbf.exists():
        raise SystemExit(
            f"Missing PBF: {pbf}. Run scripts/download_osm_extract.py --city {city} first."
        )

    tracts_path = (
        Path(args.tracts) if args.tracts else Path("data/raw/census") / f"{city}_tracts_2021.gpkg"
    )
    if not tracts_path.exists():
        raise SystemExit(f"Missing tracts file: {tracts_path}")

    out_path = (
        Path(args.output) if args.output else Path("data/features") / f"{city}_osm_features.parquet"
    )
    ensure_dir(out_path.parent)

    workdir = Path(args.workdir) / city
    ensure_dir(workdir)

    roads_pbf = workdir / "roads.osm.pbf"
    pois_pbf = workdir / "pois.osm.pbf"
    roads_geojson = workdir / (
        "roads.geojsonseq" if (workdir / "roads.geojsonseq").exists() else "roads.geojson"
    )
    pois_geojson = workdir / (
        "pois.geojsonseq" if (workdir / "pois.geojsonseq").exists() else "pois.geojson"
    )

    if out_path.exists() and not args.force:
        print(f"✅ Output already exists: {out_path} (use --force to recompute)")
        return 0

    if shutil.which("osmium") is None:
        raise SystemExit(
            "osmium-tool is required to process PBF files. Install it before running this command."
        )

    # 1) Filter PBF locally (FAST) using osmium-tool
    if (not roads_pbf.exists()) or args.force:
        run(["osmium", "tags-filter", str(pbf), "w/highway", "-o", str(roads_pbf), "--overwrite"])

    if (not pois_pbf.exists()) or args.force:
        # nodes with POI-like tags
        run(
            [
                "osmium",
                "tags-filter",
                str(pbf),
                "n/amenity",
                "n/shop",
                "n/leisure",
                "n/tourism",
                "-o",
                str(pois_pbf),
                "--overwrite",
            ]
        )

    # 2) Export to GeoJSON for Python processing
    if (not roads_geojson.exists()) or args.force:
        run(
            [
                "osmium",
                "export",
                str(roads_pbf),
                "-o",
                str(roads_geojson),
                "--overwrite",
                "--geometry-types=linestring",
            ]
        )

    if (not pois_geojson.exists()) or args.force:
        run(
            [
                "osmium",
                "export",
                str(pois_pbf),
                "-o",
                str(pois_geojson),
                "--overwrite",
                "--geometry-types=point",
            ]
        )

    # 3) Load tracts + features
    tracts = gpd.read_file(tracts_path)  # auto-detect first layer
    if "tract_id" not in tracts.columns:
        raise SystemExit("tracts layer missing tract_id column (expected standardized tract_id).")

    tracts = tracts[["tract_id", "geometry"]].copy()
    tracts = tracts[tracts.geometry.notnull()].copy()
    tracts = tracts.set_geometry("geometry")

    pois = gpd.read_file(pois_geojson)
    roads = gpd.read_file(roads_geojson)

    # Some exports may include null geometries; drop them
    pois = pois[pois.geometry.notnull()].copy()
    roads = roads[roads.geometry.notnull()].copy()

    # 4) POI category normalization
    # Try to build a single category column using the strongest available tag
    def first_nonnull(row, keys):
        for k in keys:
            if k in row and pd.notna(row[k]) and str(row[k]).strip() != "":
                return str(row[k])
        return "unknown"

    tag_priority = ["amenity", "shop", "leisure", "tourism"]
    # osmium export puts tags as regular columns when present
    pois["poi_type"] = pois.apply(lambda r: first_nonnull(r, tag_priority), axis=1)

    # 5) Spatial joins + aggregations (work in metric CRS for road lengths)
    tr_m = to_metric(tracts)
    pois_m = to_metric(pois)
    roads_m = to_metric(roads)

    # POI counts per tract
    poi_join = gpd.sjoin(
        pois_m[["poi_type", "geometry"]],
        tr_m[["tract_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    poi_count = poi_join.groupby("tract_id").size().rename("poi_count").reset_index()

    # POI entropy per tract
    ent_rows = []
    for tract_id, grp in poi_join.groupby("tract_id"):
        c = grp["poi_type"].value_counts()
        ent_rows.append({"tract_id": tract_id, "poi_entropy": entropy_from_counts(c)})
    poi_entropy = pd.DataFrame(ent_rows, columns=["tract_id", "poi_entropy"])

    # Road length per tract (intersection + length)
    # Intersect can be heavy but is still orders faster than Overpass for NYC if run once.
    roads_small = roads_m[["geometry"]].copy()
    # Give each road an id to track
    roads_small["rid"] = range(len(roads_small))

    inter = gpd.overlay(
        roads_small,
        tr_m[["tract_id", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    inter["seg_len_m"] = inter.geometry.length
    road_len = inter.groupby("tract_id")["seg_len_m"].sum().rename("road_length_m").reset_index()

    # 6) Merge into final feature table
    out = tracts[["tract_id"]].copy()
    out = out.merge(poi_count, on="tract_id", how="left")
    out = out.merge(poi_entropy, on="tract_id", how="left")
    out = out.merge(road_len, on="tract_id", how="left")

    out = out.fillna({"poi_count": 0.0, "poi_entropy": 0.0, "road_length_m": 0.0})
    out.to_parquet(out_path, index=False)

    print(f"✅ Wrote OSM tract features: {out_path} (rows={len(out)}, cols={len(out.columns)})")
    print(
        f"   poi_count: min={out.poi_count.min()} max={out.poi_count.max()} "
        f"mean={out.poi_count.mean():.2f}"
    )
    print(
        f"   road_length_m: min={out.road_length_m.min():.1f} "
        f"max={out.road_length_m.max():.1f} mean={out.road_length_m.mean():.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
