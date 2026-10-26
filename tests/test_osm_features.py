import numpy as np
import pandas as pd
import pytest

from scripts.compute_osm_features_from_pbf import (
    aggregate_features,
    load_osm_gdal,
    poi_type_from_tags,
)


def test_poi_tag_parsing_ignores_unrelated_nodes_and_keeps_priority():
    assert poi_type_from_tags(None) is None
    assert poi_type_from_tags('"name"=>"Example"') is None
    assert poi_type_from_tags('"amenity"=>"no"') is None
    assert poi_type_from_tags('"shop"=>"books","amenity"=>"cafe"') == "amenity:cafe"
    assert poi_type_from_tags('"shop"=>"books, gifts"') == "shop:books, gifts"


def test_gdal_osm_reader_selects_real_highways_and_tagged_pois(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    source = tmp_path / "tiny.osm"
    source.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="TerraLens tests">
<node id="1" lat="40.700" lon="-74.000"><tag k="amenity" v="cafe"/></node>
<node id="2" lat="40.701" lon="-74.001"><tag k="name" v="Not a POI"/></node>
<node id="3" lat="40.702" lon="-74.002"><tag k="shop" v="books"/></node>
<node id="4" lat="40.703" lon="-74.003"/>
<way id="10"><nd ref="1"/><nd ref="4"/><tag k="highway" v="residential"/></way>
<way id="11"><nd ref="2"/><nd ref="3"/><tag k="waterway" v="stream"/></way>
</osm>""",
        encoding="utf-8",
    )
    tracts = gpd.GeoDataFrame(
        {"tract_id": ["36061000100"]}, geometry=[box(-74.01, 40.69, -73.99, 40.71)], crs=4326
    )
    roads, pois = load_osm_gdal(source, tracts)
    assert roads.highway.tolist() == ["residential"]
    assert sorted(pois.poi_type) == ["amenity:cafe", "shop:books"]
    out = aggregate_features(tracts, roads, pois)
    assert out.poi_count.iloc[0] == 2
    assert out.poi_entropy.iloc[0] == pytest.approx(np.log(2))
    assert 300 < out.road_length_m.iloc[0] < 500
    assert np.isfinite(out.select_dtypes(include="number").to_numpy()).all()


def test_empty_pois_do_not_prevent_road_features():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString, box

    tracts = gpd.GeoDataFrame(
        {"tract_id": ["one"]}, geometry=[box(-74.01, 40.7, -74, 40.71)], crs=4326
    )
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(-74.009, 40.705), (-74.001, 40.705)])], crs=4326
    )
    pois = gpd.GeoDataFrame({"poi_type": pd.Series(dtype="str")}, geometry=[], crs=4326)
    out = aggregate_features(tracts, roads, pois)
    assert out.poi_count.iloc[0] == 0 and out.poi_entropy.iloc[0] == 0
    assert out.road_length_m.iloc[0] > 0


def test_rich_features_use_clipped_lengths_and_union_polygon_coverage():
    gpd = pytest.importorskip('geopandas')
    from shapely.geometry import LineString, Point, box
    tracts = gpd.GeoDataFrame({'tract_id': ['left', 'right']}, geometry=[box(500000, 4500000, 500100, 4500100), box(500100, 4500000, 500200, 4500100)], crs=32618)
    roads = gpd.GeoDataFrame({'highway': ['residential', 'cycleway']}, geometry=[LineString([(500000, 4500050), (500200, 4500050)]), LineString([(500025, 4500000), (500025, 4500100)])], crs=32618)
    pois = gpd.GeoDataFrame({'poi_type': ['amenity:school', 'shop:books']}, geometry=[Point(500050, 4500050), Point(500150, 4500050)], crs=32618)
    polygons = gpd.GeoDataFrame({'building': ['yes', 'yes', None, None], 'landuse': [None, None, 'residential', 'residential']}, geometry=[box(500025, 4500025, 500075, 4500075), box(500025, 4500025, 500075, 4500075), box(500000, 4500000, 500100, 4500100), box(500000, 4500000, 500100, 4500100)], crs=32618)
    out = aggregate_features(tracts, roads, pois, polygons).set_index('tract_id')
    assert out.loc['left', 'road_length_m'] == pytest.approx(200, rel=0.01)
    assert out.loc['right', 'road_length_m'] == pytest.approx(100, rel=0.01)
    assert out.loc['left', 'road_residential_fraction'] == pytest.approx(0.5, rel=0.01)
    assert out.loc['left', 'poi_education_count'] == 1
    assert out.loc['right', 'poi_retail_count'] == 1
    assert out.loc['left', 'building_coverage_fraction'] == pytest.approx(0.25, rel=0.01)
    assert out.loc['left', 'land_residential_fraction'] == pytest.approx(1, rel=0.01)
    assert out.loc['right', 'land_residential_fraction'] == 0
    assert np.isfinite(out.to_numpy()).all()


def test_invalid_real_world_polygon_is_repaired_before_union():
    gpd = pytest.importorskip('geopandas')
    from shapely.geometry import Polygon, box
    tracts = gpd.GeoDataFrame({'tract_id': ['one']}, geometry=[box(500000, 4500000, 500100, 4500100)], crs=32618)
    roads = gpd.GeoDataFrame({'highway': pd.Series(dtype='str')}, geometry=[], crs=32618)
    pois = gpd.GeoDataFrame({'poi_type': pd.Series(dtype='str')}, geometry=[], crs=32618)
    bowtie = Polygon([(500000,4500000), (500100,4500100), (500000,4500100), (500100,4500000), (500000,4500000)])
    assert not bowtie.is_valid
    polygons = gpd.GeoDataFrame({'building': ['yes'], 'landuse': ['residential']}, geometry=[bowtie], crs=32618)
    out = aggregate_features(tracts, roads, pois, polygons).set_index('tract_id')
    assert out.loc['one','building_coverage_fraction'] == pytest.approx(0.5, rel=0.01)
    assert out.loc['one','land_residential_fraction'] == pytest.approx(0.5, rel=0.01)
    assert out.loc['one','road_length_m'] == 0
    assert np.isfinite(out.to_numpy()).all()
