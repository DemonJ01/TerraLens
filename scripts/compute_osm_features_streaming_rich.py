#!/usr/bin/env python3
"""
NYC-safe streaming OSM feature builder (geojsonseq -> tract-level features)

Design goals:
- Works for NYC-scale OSM (streaming; low memory).
- Produces richer, interpretable features:
  - road_length_m (total) + road length by road class
  - poi_count (total) + POI counts by category + entropy
  - tract_area_km2 + densities (poi per km^2, road km per km^2)

Approximation (intentional for speed):
- Assign each road LineString to the tract containing its midpoint.
  (Good baseline, avoids expensive polygon intersection.)
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, shape

ROAD_CLASSES = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "service",
    "other",
]

POI_CATEGORIES = [
    "food",
    "retail",
    "health",
    "education",
    "transit",
    "parks",
    "civic",
    "hospitality",
    "entertainment",
    "other",
]

POI_CATEGORY_MAP = {
    # food & drink
    ("amenity", "restaurant"): "food",
    ("amenity", "cafe"): "food",
    ("amenity", "fast_food"): "food",
    ("amenity", "bar"): "food",
    ("amenity", "pub"): "food",
    # retail
    ("shop", "*"): "retail",
    # health
    ("amenity", "hospital"): "health",
    ("amenity", "clinic"): "health",
    ("amenity", "doctors"): "health",
    ("amenity", "pharmacy"): "health",
    ("amenity", "dentist"): "health",
    # education
    ("amenity", "school"): "education",
    ("amenity", "college"): "education",
    ("amenity", "university"): "education",
    ("amenity", "kindergarten"): "education",
    ("amenity", "library"): "education",
    # transit
    ("amenity", "bus_station"): "transit",
    ("amenity", "ferry_terminal"): "transit",
    ("public_transport", "*"): "transit",
    ("railway", "station"): "transit",
    ("railway", "subway_entrance"): "transit",
    # parks/leisure/tourism
    ("leisure", "*"): "parks",
    ("tourism", "*"): "parks",
    # civic
    ("amenity", "police"): "civic",
    ("amenity", "fire_station"): "civic",
    ("amenity", "townhall"): "civic",
    ("amenity", "courthouse"): "civic",
    ("amenity", "post_office"): "civic",
    # hospitality
    ("tourism", "hotel"): "hospitality",
    ("tourism", "hostel"): "hospitality",
    ("tourism", "motel"): "hospitality",
    # entertainment
    ("amenity", "cinema"): "entertainment",
    ("amenity", "theatre"): "entertainment",
    ("amenity", "nightclub"): "entertainment",
}


def road_bucket(highway: Optional[str]) -> str:
    if not highway:
        return "other"
    hv = str(highway).lower()
    return hv if hv in ROAD_CLASSES else "other"


def poi_bucket(props: Dict[str, Any]) -> str:
    for key in ["amenity", "shop", "leisure", "tourism", "public_transport", "railway"]:
        v = props.get(key)
        if not v:
            continue
        v = str(v).lower()
        if (key, v) in POI_CATEGORY_MAP:
            return POI_CATEGORY_MAP[(key, v)]
        if (key, "*") in POI_CATEGORY_MAP:
            return POI_CATEGORY_MAP[(key, "*")]
        if key in ["leisure", "tourism"]:
            return "parks"
        if key == "shop":
            return "retail"
        return "other"
    return "other"


def entropy_from_counts(counts: Dict[str, float]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return float(ent)


def find_tract_id(point: Point, tracts_ll: gpd.GeoDataFrame) -> Optional[str]:
    # fast bbox candidates via spatial index
    sidx = tracts_ll.sindex
    cand_idx = list(sidx.intersection(point.bounds))
    if not cand_idx:
        return None
    cand = tracts_ll.iloc[cand_idx]
    # contains check
    hits = cand[cand.geometry.contains(point)]
    if hits.empty:
        return None
    return str(hits.iloc[0]["tract_id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracts", required=True)
    ap.add_argument("--roads", required=True, help="roads.geojsonseq")
    ap.add_argument("--pois", required=True, help="pois.geojsonseq")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tracts = gpd.read_file(args.tracts)
    if "tract_id" not in tracts.columns:
        raise SystemExit("❌ tracts file missing 'tract_id' column")

    # compute area in km^2
    tracts_proj = tracts.to_crs("EPSG:3857")
    tracts["tract_area_km2"] = (tracts_proj.geometry.area / 1e6).astype(float)

    # keep a lat/lon version for point-in-polygon
    tracts_ll = tracts.to_crs("EPSG:4326")[["tract_id", "geometry", "tract_area_km2"]].copy()
    _ = tracts_ll.sindex  # build spatial index

    # initialize accumulators
    road_total_m = {tid: 0.0 for tid in tracts_ll["tract_id"].astype(str)}
    road_by_class = {tid: {k: 0.0 for k in ROAD_CLASSES} for tid in road_total_m.keys()}

    poi_total = {tid: 0.0 for tid in road_total_m.keys()}
    poi_by_cat = {tid: {k: 0.0 for k in POI_CATEGORIES} for tid in road_total_m.keys()}

    # --- roads ---
    roads_path = Path(args.roads)
    if not roads_path.exists():
        raise SystemExit(f"❌ roads file not found: {roads_path}")

    with roads_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            feat = json.loads(line)
            geom = feat.get("geometry")
            if not geom:
                continue
            props = feat.get("properties", {}) or {}

            g = shape(geom)
            if g.is_empty:
                continue

            # midpoint assignment
            mid = g.interpolate(0.5, normalized=True) if hasattr(g, "interpolate") else g.centroid
            if mid is None or mid.is_empty:
                mid = g.centroid
            mid = Point(mid.x, mid.y)

            tid = find_tract_id(mid, tracts_ll)
            if tid is None or tid not in road_total_m:
                continue

            # length in meters (project only this geometry)
            g_m = gpd.GeoSeries([g], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
            seg_m = float(g_m.length)

            road_total_m[tid] += seg_m
            cls = road_bucket(props.get("highway"))
            road_by_class[tid][cls] += seg_m

    # --- pois ---
    pois_path = Path(args.pois)
    if not pois_path.exists():
        raise SystemExit(f"❌ pois file not found: {pois_path}")

    with pois_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            feat = json.loads(line)
            geom = feat.get("geometry")
            if not geom:
                continue
            props = feat.get("properties", {}) or {}

            g = shape(geom)
            if g.is_empty:
                continue

            pnt = g if isinstance(g, Point) else g.centroid
            tid = find_tract_id(pnt, tracts_ll)
            if tid is None or tid not in poi_total:
                continue

            poi_total[tid] += 1.0
            cat = poi_bucket(props)
            poi_by_cat[tid][cat] += 1.0

    # build output
    rows = []
    for _, r in tracts_ll.iterrows():
        tid = str(r["tract_id"])
        area_km2 = float(r.get("tract_area_km2", 0.0))

        cat_counts = poi_by_cat[tid]
        ent = entropy_from_counts(cat_counts)

        road_m = float(road_total_m[tid])
        poi_c = float(poi_total[tid])

        row = {
            "tract_id": tid,
            "poi_count": poi_c,
            "poi_entropy": ent,
            "road_length_m": road_m,
            "tract_area_km2": area_km2,
            "poi_density_per_km2": float(poi_c / area_km2) if area_km2 > 0 else 0.0,
            "road_km_per_km2": float((road_m / 1000.0) / area_km2) if area_km2 > 0 else 0.0,
        }

        for k in ROAD_CLASSES:
            row[f"road_len_{k}_m"] = float(road_by_class[tid][k])

        for k in POI_CATEGORIES:
            row[f"poi_{k}_count"] = float(cat_counts[k])

        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"✅ Wrote RICH OSM tract features: {out_path} (rows={len(df)}, cols={len(df.columns)})")
    print("   sample cols:", list(df.columns)[:18])


if __name__ == "__main__":
    main()
