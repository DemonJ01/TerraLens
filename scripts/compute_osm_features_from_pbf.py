#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import union_all

FEATURE_SCHEMA_VERSION = 2
POI_GROUPS = (
    "education",
    "healthcare",
    "food_drink",
    "retail",
    "finance",
    "transport",
    "entertainment",
    "tourism",
    "religion",
    "civic",
    "services",
    "other",
)
ROAD_GROUPS = ("motorway", "arterial", "residential", "service", "pedestrian", "cycle", "other")
LAND_GROUPS = ("residential", "commercial", "industrial", "retail", "green_space", "water")


def polygonal_parts(geometry):
    """Keep repaired polygon area while discarding collapsed lines/points."""
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [
        part for component in getattr(geometry, "geoms", []) for part in polygonal_parts(component)
    ]


def poi_group(value: str) -> str:
    """Fixed OSM tag groups; categories never depend on census labels."""
    key, _, tag = str(value).partition(":")
    if key == "shop":
        return "retail"
    if key == "tourism":
        return "tourism"
    if key == "leisure":
        return "entertainment"
    if tag in {
        "school",
        "college",
        "university",
        "kindergarten",
        "library",
        "language_school",
        "music_school",
    }:
        return "education"
    if tag in {
        "hospital",
        "clinic",
        "doctors",
        "dentist",
        "pharmacy",
        "nursing_home",
        "veterinary",
    }:
        return "healthcare"
    if tag in {
        "restaurant",
        "cafe",
        "fast_food",
        "bar",
        "pub",
        "food_court",
        "ice_cream",
        "biergarten",
    }:
        return "food_drink"
    if tag in {"bank", "atm", "bureau_de_change"}:
        return "finance"
    if tag in {
        "parking",
        "parking_entrance",
        "parking_space",
        "fuel",
        "charging_station",
        "bus_station",
        "bicycle_parking",
        "bicycle_rental",
        "car_rental",
        "taxi",
        "ferry_terminal",
    }:
        return "transport"
    if tag in {"cinema", "theatre", "nightclub", "casino", "arts_centre", "community_centre"}:
        return "entertainment"
    if tag in {"place_of_worship", "monastery"}:
        return "religion"
    if tag in {
        "police",
        "fire_station",
        "post_office",
        "townhall",
        "courthouse",
        "social_facility",
    }:
        return "civic"
    if tag in {
        "toilets",
        "drinking_water",
        "recycling",
        "waste_disposal",
        "shelter",
        "bench",
        "telephone",
        "public_bath",
    }:
        return "services"
    return "other"


def road_group(value) -> str:
    tag = str(value).split(";")[0]
    if tag in {"motorway", "motorway_link", "trunk", "trunk_link"}:
        return "motorway"
    if tag in {
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
    }:
        return "arterial"
    if tag in {"residential", "living_street", "unclassified"}:
        return "residential"
    if tag in {"service", "track"}:
        return "service"
    if tag in {"footway", "path", "steps", "pedestrian", "bridleway", "corridor"}:
        return "pedestrian"
    if tag == "cycleway":
        return "cycle"
    return "other"


def land_group(row) -> str | None:
    landuse = str(row.get("landuse", ""))
    natural = str(row.get("natural", ""))
    leisure = str(row.get("leisure", ""))
    if landuse in {"residential", "commercial", "industrial", "retail"}:
        return landuse
    if natural in {"water", "bay", "wetland"} or landuse in {"reservoir", "basin"}:
        return "water"
    if (
        landuse
        in {
            "forest",
            "grass",
            "meadow",
            "recreation_ground",
            "village_green",
            "allotments",
            "cemetery",
            "orchard",
        }
        or natural in {"wood", "grassland", "scrub", "heath"}
        or leisure in {"park", "garden", "golf_course", "nature_reserve", "recreation_ground"}
    ):
        return "green_space"
    return None


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
    return gdf.to_crs(gdf.estimate_utm_crs())


def poi_type_from_tags(value) -> str | None:
    """Read GDAL's OSM hstore tags without treating untagged nodes as POIs."""
    if not isinstance(value, str):
        return None
    matches = dict(re.findall(r'"(amenity|shop|leisure|tourism)"=>"((?:[^"\\]|\\.)*)"', value))
    for key in ("amenity", "shop", "leisure", "tourism"):
        if matches.get(key) and matches[key] not in {"no", "false"}:
            return key + ":" + json.loads('"' + matches[key] + '"')
    return None


def load_osm_gdal(pbf: Path, tracts: gpd.GeoDataFrame):
    """Read PBF via the GDAL OSM driver already supplied with Pyogrio."""
    import pyogrio

    if "OSM" not in pyogrio.list_drivers():
        raise SystemExit("Install osmium-tool or a Pyogrio build with the GDAL OSM driver.")
    bbox = tuple(tracts.to_crs(4326).total_bounds)
    print("Reading roads with GDAL...", flush=True)
    roads = pyogrio.read_dataframe(
        pbf, layer="lines", columns=["highway"], bbox=bbox, where="highway IS NOT NULL"
    )
    print("Reading tagged point locations with GDAL...", flush=True)
    where = " OR ".join(
        f"other_tags LIKE '%\"{key}\"=>%'" for key in ("amenity", "shop", "leisure", "tourism")
    )
    pois = pyogrio.read_dataframe(
        pbf, layer="points", columns=["other_tags"], bbox=bbox, where=where
    )
    pois["poi_type"] = pois["other_tags"].map(poi_type_from_tags)
    pois = pois.dropna(subset=["poi_type", "geometry"])[["poi_type", "geometry"]]
    roads = roads.dropna(subset=["highway", "geometry"])
    print(f"Loaded {len(roads):,} road segments and {len(pois):,} POI nodes.", flush=True)
    return roads, pois


def load_osm_polygons_gdal(pbf: Path, tracts: gpd.GeoDataFrame):
    """Read actual mapped building footprints and land-cover polygons."""
    import pyogrio

    print("Reading building and land-cover polygons with GDAL...", flush=True)
    columns = ["building", "landuse", "natural", "leisure"]
    polygons = pyogrio.read_dataframe(
        pbf,
        layer="multipolygons",
        columns=columns,
        bbox=tuple(tracts.to_crs(4326).total_bounds),
        where="building IS NOT NULL OR landuse IS NOT NULL OR natural IS NOT NULL OR leisure IS NOT NULL",
    )
    polygons = polygons.dropna(subset=["geometry"])
    print(f"Loaded {len(polygons):,} building/land-cover polygons.", flush=True)
    return polygons


def aggregate_features(tracts, roads, pois, polygons=None) -> pd.DataFrame:
    """Measure target-independent OSM features in the tracts' common UTM CRS.

    POI and building counts use points strictly within a tract. Road lengths and
    polygon coverage are clipped to boundaries. Land-cover polygons are unioned
    within each category before measuring, avoiding double-counted coverage.
    Zero describes no mapped object, not proof that no real-world object exists.
    """
    metric_crs = tracts.estimate_utm_crs()
    tr_m = tracts.to_crs(metric_crs)[["tract_id", "geometry"]].copy()
    tr_m["tract_id"] = tr_m["tract_id"].astype(str)
    roads_m = roads.to_crs(metric_crs)
    pois_m = pois.to_crs(metric_crs)
    roads_m["road_group"] = roads_m.get("highway", pd.Series("unknown", index=roads_m.index)).map(
        road_group
    )
    pois_m["poi_group"] = pois_m["poi_type"].map(poi_group)
    poi_join = gpd.sjoin(pois_m[["poi_type", "geometry"]], tr_m, how="inner", predicate="within")
    poi_join["poi_group"] = poi_join["poi_type"].map(poi_group)
    counts = poi_join.groupby("tract_id").size().rename("poi_count")
    entropies = (
        poi_join.groupby("tract_id")["poi_type"]
        .agg(lambda values: entropy_from_counts(values.value_counts()))
        .rename("poi_entropy")
    )
    print("Clipping road segments to tract boundaries...", flush=True)
    clipped = gpd.overlay(
        roads_m[["road_group", "geometry"]], tr_m, how="intersection", keep_geom_type=False
    )
    clipped["road_length_m"] = clipped.geometry.length
    lengths = clipped.groupby("tract_id")["road_length_m"].sum()
    out = tr_m.set_index("tract_id").drop(columns="geometry")
    out["tract_area_km2"] = tr_m.set_index("tract_id").geometry.area / 1e6
    out = out.join([counts, entropies, lengths]).astype(float).fillna(0)
    area = out["tract_area_km2"].where(out["tract_area_km2"] > 0)
    out["poi_density_per_km2"] = out["poi_count"].div(area).fillna(0)
    out["road_density_km_per_km2"] = out["road_length_m"].div(1000 * area).fillna(0)
    out["tract_perimeter_km"] = tr_m.set_index("tract_id").geometry.length / 1000
    out["tract_compactness"] = (
        (4 * math.pi * out["tract_area_km2"] / out["tract_perimeter_km"].pow(2).replace(0, np.nan))
        .fillna(0)
        .clip(0, 1)
    )
    for group in POI_GROUPS:
        group_counts = poi_join.loc[poi_join["poi_group"] == group].groupby("tract_id").size()
        out[f"poi_{group}_count"] = group_counts.reindex(out.index, fill_value=0).astype(float)
        out[f"poi_{group}_density_per_km2"] = out[f"poi_{group}_count"].div(area).fillna(0)
    total_length = out["road_length_m"].where(out["road_length_m"] > 0)
    for group in ROAD_GROUPS:
        group_length = (
            clipped.loc[clipped["road_group"] == group].groupby("tract_id")["road_length_m"].sum()
        )
        out[f"road_{group}_length_m"] = group_length.reindex(out.index, fill_value=0).astype(float)
        out[f"road_{group}_density_km_per_km2"] = (
            out[f"road_{group}_length_m"].div(1000 * area).fillna(0)
        )
        out[f"road_{group}_fraction"] = out[f"road_{group}_length_m"].div(total_length).fillna(0)

    building_cols = [
        "building_count",
        "building_density_per_km2",
        "building_coverage_fraction",
        "building_mean_footprint_m2",
        "building_max_footprint_m2",
    ]
    for column in building_cols:
        out[column] = 0.0
    for group in LAND_GROUPS:
        out[f"land_{group}_fraction"] = 0.0
    if polygons is not None and len(polygons):
        polygons_m = polygons.to_crs(metric_crs).copy()
        # Real OSM extracts contain occasional self-intersecting multipolygons.
        # Repair them before unions, not just inside the later overlay call.
        polygons_m["geometry"] = polygons_m.geometry.make_valid()
        collections = polygons_m.geom_type == "GeometryCollection"
        polygons_m.loc[collections, "geometry"] = polygons_m.loc[collections].geometry.map(
            lambda geometry: union_all(polygonal_parts(geometry))
        )
        polygons_m = polygons_m.loc[~polygons_m.geometry.is_empty].copy()
        building_tag = polygons_m.get("building", pd.Series(index=polygons_m.index, dtype="object"))
        buildings = polygons_m.loc[
            building_tag.notna() & ~building_tag.isin(["no", "false"])
        ].copy()
        if len(buildings):
            print("Measuring building footprints...", flush=True)
            centers = buildings[["geometry"]].copy()
            centers["footprint_m2"] = centers.geometry.area
            centers["geometry"] = centers.geometry.representative_point()
            building_join = gpd.sjoin(centers, tr_m, how="inner", predicate="within")
            grouped = building_join.groupby("tract_id")["footprint_m2"]
            out["building_count"] = grouped.size().reindex(out.index, fill_value=0).astype(float)
            out["building_density_per_km2"] = out["building_count"].div(area).fillna(0)
            out["building_mean_footprint_m2"] = grouped.mean().reindex(out.index, fill_value=0)
            out["building_max_footprint_m2"] = grouped.max().reindex(out.index, fill_value=0)
            # Overlapping mapped footprints are unioned before measuring coverage.
            # Clip first to avoid unioning unrelated buildings from the PBF bbox.
            building_inter = gpd.overlay(
                buildings[["geometry"]], tr_m, how="intersection", keep_geom_type=False
            )
            covered = building_inter.dissolve(by="tract_id").geometry.area
            out["building_coverage_fraction"] = (
                covered.reindex(out.index, fill_value=0).div(area * 1e6).fillna(0).clip(0, 1)
            )
        land_columns = [
            column for column in ("landuse", "natural", "leisure") if column in polygons_m
        ]
        # Most source polygons are buildings without any land-cover tag. Avoid
        # running Python category parsing on millions of such footprints.
        candidates = polygons_m[land_columns].notna().any(axis=1)
        land = polygons_m.loc[candidates].copy()
        land["land_group"] = land.apply(land_group, axis=1) if len(land) else None
        land = land.dropna(subset=["land_group"])[["land_group", "geometry"]]
        if len(land):
            print("Measuring unioned land-cover areas...", flush=True)
            land = land.dissolve(by="land_group").reset_index()
            land_inter = gpd.overlay(land, tr_m, how="intersection", keep_geom_type=False)
            land_inter["area_m2"] = land_inter.geometry.area
            for group in LAND_GROUPS:
                covered = (
                    land_inter.loc[land_inter["land_group"] == group]
                    .groupby("tract_id")["area_m2"]
                    .sum()
                )
                out[f"land_{group}_fraction"] = (
                    covered.reindex(out.index, fill_value=0).div(area * 1e6).fillna(0).clip(0, 1)
                )
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite measured OSM feature values")
    return pd.DataFrame(out).reset_index()


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

    tracts = gpd.read_file(tracts_path)
    if "tract_id" not in tracts:
        raise SystemExit("Tracts layer must contain tract_id")
    tracts = tracts[["tract_id", "geometry"]].dropna(subset=["geometry"])
    if shutil.which("osmium") is None:
        roads, pois = load_osm_gdal(pbf, tracts)
    else:
        if (not roads_pbf.exists()) or args.force:
            run(
                [
                    "osmium",
                    "tags-filter",
                    str(pbf),
                    "w/highway",
                    "-o",
                    str(roads_pbf),
                    "--overwrite",
                ]
            )
        if (not pois_pbf.exists()) or args.force:
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
        for filtered, exported, geometry_type in [
            (roads_pbf, roads_geojson, "linestring"),
            (pois_pbf, pois_geojson, "point"),
        ]:
            if not exported.exists() or args.force:
                run(
                    [
                        "osmium",
                        "export",
                        str(filtered),
                        "-o",
                        str(exported),
                        "--overwrite",
                        f"--geometry-types={geometry_type}",
                    ]
                )
        roads = gpd.read_file(roads_geojson).dropna(subset=["geometry"])
        pois = gpd.read_file(pois_geojson).dropna(subset=["geometry"])

        def first_tag(row):
            for key in ("amenity", "shop", "leisure", "tourism"):
                value = row.get(key)
                if pd.notna(value) and str(value).strip() and value not in {"no", "false"}:
                    return f"{key}:{value}"
            return None

        pois["poi_type"] = pois.apply(first_tag, axis=1)
        pois = pois.dropna(subset=["poi_type"])

    import pyogrio

    if "OSM" in pyogrio.list_drivers():
        polygons = load_osm_polygons_gdal(pbf, tracts)
    else:
        # Support installations with osmium but without GDAL's optional OSM driver.
        polygon_pbf = workdir / "building_landcover.osm.pbf"
        polygon_json = workdir / "building_landcover.geojson"
        if not polygon_pbf.exists() or args.force:
            tags = [
                f"{kind}/{tag}"
                for kind in ("w", "r")
                for tag in ("building", "landuse", "natural", "leisure")
            ]
            run(["osmium", "tags-filter", str(pbf), *tags, "-o", str(polygon_pbf), "--overwrite"])
        if not polygon_json.exists() or args.force:
            run(
                [
                    "osmium",
                    "export",
                    str(polygon_pbf),
                    "-o",
                    str(polygon_json),
                    "--overwrite",
                    "--geometry-types=polygon",
                ]
            )
        polygons = gpd.read_file(polygon_json).dropna(subset=["geometry"])
    out = aggregate_features(tracts, roads, pois, polygons)
    out.to_parquet(out_path, index=False)
    metadata = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "city": city,
        "osm_pbf": str(pbf),
        "tract_geometry": str(tracts_path),
        "metric_crs": str(tracts.estimate_utm_crs()),
        "rows": len(out),
        "feature_columns": [column for column in out if column != "tract_id"],
        "source_objects": {
            "roads": len(roads),
            "poi_nodes": len(pois),
            "building_landcover_polygons": len(polygons),
        },
        "definitions": {
            "counts": "Tagged POI nodes and building representative points strictly within tract boundaries.",
            "lengths": "Mapped highway geometry clipped to tract boundaries; meters in local UTM CRS.",
            "areas": "Mapped building and per-category land-cover polygon unions clipped to tract boundaries.",
            "zero_values": "No mapped object in the source extract, not proof of real-world absence.",
            "land_cover": "Fixed categories may overlap across categories; each category is unioned before measuring.",
        },
    }
    out_path.with_suffix(".source.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote measured OSM features: {out_path} ({len(out):,} tracts, {len(out.columns)-1} features)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
