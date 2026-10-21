import sqlite3

import pandas as pd

from terralens.utils.status import DataChecker, DataStatus


def write_table(root, relative, data):
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(data)
    if destination.suffix == ".csv":
        frame.to_csv(destination, index=False)
    else:
        frame.to_parquet(destination, index=False)
    return destination


def write_tracts(root, city="nyc", year=2022):
    path = root / "raw" / "census" / f"{city}_tracts_{year}.gpkg"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE gpkg_contents (table_name TEXT, data_type TEXT)")
        db.execute("INSERT INTO gpkg_contents VALUES ('tracts', 'features')")
        db.execute("CREATE TABLE tracts (tract_id TEXT)")
        db.execute("INSERT INTO tracts VALUES ('36061000100')")


def test_census_recognizes_current_parquet_and_requested_year(tmp_path):
    checker = DataChecker(tmp_path)
    write_table(
        tmp_path,
        "raw/census/nyc_income_2022.parquet",
        {
            "tract_id": ["36061000100"],
            "median_income": [60_000],
        },
    )
    assert checker.check_census("nyc", 2022) == DataStatus.AVAILABLE
    assert checker.check_census("nyc", 2021) == DataStatus.MISSING


def test_census_legacy_csv_and_corrupt_files(tmp_path):
    checker = DataChecker(tmp_path)
    path = write_table(tmp_path, "raw/census/nyc_income_2021.csv", {"wrong": [1]})
    assert checker.check_census("nyc") == DataStatus.PARTIAL
    pd.DataFrame({"tract_id": ["36061000100"], "median_income": [60_000]}).to_csv(path, index=False)
    assert checker.check_census("nyc") == DataStatus.AVAILABLE
    path.write_text("")
    assert checker.check_census("nyc") == DataStatus.PARTIAL


def test_osm_only_recognizes_city_specific_artifacts(tmp_path):
    checker = DataChecker(tmp_path)
    osm_dir = tmp_path / "raw/osm"
    osm_dir.mkdir(parents=True)
    (osm_dir / "chicago.osm.pbf").write_bytes(b"extract")
    assert checker.check_osm("nyc") == DataStatus.MISSING
    assert checker.check_osm("chicago") == DataStatus.PARTIAL
    write_table(
        tmp_path,
        "features/nyc_geo_features.parquet",
        {
            "tract_id": ["36061000100"],
            "poi_count": [1],
            "road_length_m": [0.0],
        },
    )
    assert checker.check_osm("nyc") == DataStatus.AVAILABLE


def test_tracts_check_requested_vintage_and_invalid_content(tmp_path):
    checker = DataChecker(tmp_path)
    write_tracts(tmp_path)
    assert checker.check_tracts("nyc", 2022) == DataStatus.AVAILABLE
    assert checker.check_tracts("nyc", 2021) == DataStatus.MISSING
    (tmp_path / "raw/census/nyc_tracts_2021.gpkg").write_bytes(b"not a geopackage")
    assert checker.check_tracts("nyc", 2021) == DataStatus.PARTIAL


def test_generic_smoke_or_synthetic_city_dataset_is_not_source_data(tmp_path):
    checker = DataChecker(tmp_path)
    sample = {
        "tract_id": ["nyc_dummy_0001"],
        "median_income": [50_000],
        "lat": [40.7],
        "lon": [-74],
    }
    write_table(tmp_path, "processed/smoke_test.parquet", sample)
    assert checker.check_dataset("nyc") == DataStatus.MISSING
    write_table(tmp_path, "processed/nyc_dataset.parquet", sample)
    assert checker.check_dataset("nyc") == DataStatus.PARTIAL
    assert not checker.is_ready_geo_only("nyc")
    sample["tract_id"] = ["36061000100"]
    write_table(tmp_path, "processed/nyc_dataset.parquet", sample)
    assert checker.check_dataset("nyc") == DataStatus.AVAILABLE
    assert checker.is_ready_geo_only("nyc")
    assert not checker.is_ready_geo_only("chicago")


def test_raw_geo_readiness_requires_geometry_income_and_osm(tmp_path):
    checker = DataChecker(str(tmp_path))
    write_table(
        tmp_path,
        "raw/census/nyc_acs_targets_2022.parquet",
        {
            "tract_id": ["36061000100"],
            "median_income": [60_000],
        },
    )
    assert not checker.is_ready_geo_only("nyc", 2022)
    write_tracts(tmp_path)
    write_table(
        tmp_path,
        "features/nyc_geo_features.parquet",
        {
            "tract_id": ["36061000100"],
            "poi_count": [1],
        },
    )
    assert checker.is_ready_geo_only("nyc", 2022)
    assert not checker.is_ready_geo_only("nyc", 2021)


def test_current_nyc_311_download_location_is_recognized(tmp_path):
    checker = DataChecker(tmp_path)
    write_table(
        tmp_path,
        "raw/nyc311/nyc311_last_12mo.parquet",
        {
            "latitude": [40.7],
            "longitude": [-74],
            "complaint_type": ["Noise"],
        },
    )
    assert checker.check_safety("nyc") == DataStatus.AVAILABLE
    assert checker.check_safety("chicago") == DataStatus.MISSING
