"""Regression checks for data isolation and the real prediction-cache pipeline."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.cross_city_eval import infer_feature_cols
from scripts.cache_predictions import clean_xy, select_features

ROOT = Path(__file__).resolve().parents[1]


def fixture_dataset(rows=60):
    rng = np.random.default_rng(11)
    road = rng.uniform(100, 1000, rows)
    return pd.DataFrame(
        {
            "tract_id": [f"tract-{i}" for i in range(rows)],
            "city": "nyc",
            "lat": 40.6 + rng.random(rows) * 0.2,
            "lon": -74 + rng.random(rows) * 0.1,
            "NAME": "tract name",
            "road_length_m": road,
            "poi_count": rng.integers(0, 100, rows),
            "complaint_count": rng.integers(0, 100, rows),
            "median_income": 30000 + 40 * road + rng.normal(0, 800, rows),
            "median_rent": rng.uniform(800, 2400, rows),
            "poverty_rate": rng.uniform(0, 0.5, rows),
        }
    )


def run_script(relative_script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / relative_script), *map(str, args)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_features_exclude_labels_metadata_and_311_from_osm_baseline():
    df = fixture_dataset()
    osm = select_features(df, "osm_only")
    assert {"road_length_m", "poi_count", "lat", "lon"}.issubset(osm)
    assert not {
        "median_income",
        "median_rent",
        "poverty_rate",
        "complaint_count",
        "NAME",
    }.intersection(osm)
    enriched = select_features(df, "osm_311")
    assert set(osm).issubset(enriched)
    assert "complaint_count" in enriched
    assert not {"median_rent", "poverty_rate", "complaint_count"}.intersection(
        infer_feature_cols(df)
    )


def test_invalid_targets_are_removed_before_model_fitting():
    features = pd.DataFrame({"road": [1, np.inf, 3, 4]})
    target = pd.Series([50, np.inf, np.nan, 80])
    X, y = clean_xy(features, target)
    assert X.index.tolist() == [0, 3]
    assert y.tolist() == [50, 80]
    assert np.isfinite(X.to_numpy()).all()


def test_cache_cli_trains_on_fixture_and_replaces_sample_manifest(tmp_path):
    data_dir, out_dir = tmp_path / "processed", tmp_path / "preds"
    data_dir.mkdir()
    out_dir.mkdir()
    df = fixture_dataset()
    df.to_parquet(data_dir / "nyc_dataset.parquet", index=False)
    pd.DataFrame(
        [
            {
                "city": "nyc_sample",
                "model": "demo",
                "feature_set": "osm_only",
                "target": "median_income",
                "path": "unused-sample.parquet",
            }
        ]
    ).to_csv(out_dir / "manifest.csv", index=False)
    run_script(
        "scripts/cache_predictions.py",
        "--city",
        "nyc",
        "--models",
        "ridge",
        "--targets",
        "median_income",
        "--data-dir",
        data_dir,
        "--out-dir",
        out_dir,
        # Ten bins cannot fit in a 3-row test set: this used to fail stratification.
        "--test_size",
        "0.05",
    )
    manifest = pd.read_csv(out_dir / "manifest.csv")
    assert manifest["city"].tolist() == ["nyc"]
    pred = pd.read_parquet(manifest.iloc[0]["path"])
    assert len(pred) == 3
    assert np.isfinite(pred[["y_true", "y_pred", "error"]].to_numpy()).all()
    assert np.allclose(pred["error"], pred["y_pred"] - pred["y_true"])
    metadata = json.loads(next(out_dir.glob("*.metrics.json")).read_text())
    assert metadata["evaluation"] == "random_holdout"
    assert metadata["train_rows"] == 57 and metadata["test_rows"] == 3
    assert metadata["test_fraction"] == 0.05


def test_cross_city_summary_preserves_both_target_transforms(tmp_path):
    data_dir, out_dir = tmp_path / "processed", tmp_path / "results"
    data_dir.mkdir()
    fixture_dataset().to_parquet(data_dir / "nyc_dataset.parquet", index=False)
    arguments = (
        "--train",
        "nyc",
        "--test",
        "nyc",
        "--model",
        "ridge",
        "--data-dir",
        data_dir,
        "--out-dir",
        out_dir,
    )
    run_script("experiments/cross_city_eval.py", *arguments)
    run_script("experiments/cross_city_eval.py", *arguments, "--no-log-target")
    summary = pd.read_csv(out_dir / "cross_city_summary.csv")
    assert len(summary) == 2
    assert set(summary["log_target"]) == {True, False}
    assert len(list(out_dir.glob("cross_city_*.json"))) == 2


@pytest.mark.parametrize(
    "script", ["experiments/train_income.py", "experiments/train_income_v2.py"]
)
def test_income_trainers_accept_current_city_feature_schema(tmp_path, script):
    data_path = tmp_path / "dataset.parquet"
    out_dir = tmp_path / "training"
    fixture_dataset().to_parquet(data_path, index=False)
    run_script(script, "--data", data_path, "--model", "ridge", "--output", out_dir)
    metrics = json.loads((out_dir / "results/income_metrics.json").read_text())
    assert metrics["n_test"] == 12
    assert np.isfinite([metrics["mae"], metrics["rmse"], metrics["r2"]]).all()


def test_city_builder_accepts_targets_without_income_file_and_preserves_missing_labels(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    census = tmp_path / "data/raw/census"
    features = tmp_path / "data/features"
    census.mkdir(parents=True)
    features.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"tract_id": ["one", "two"]},
        geometry=[box(-74.02, 40.7, -74.01, 40.71), box(-74.0, 40.72, -73.99, 40.73)],
        crs="EPSG:4326",
    ).to_file(census / "nyc_tracts_2021.gpkg", driver="GPKG")
    pd.DataFrame(
        {
            "tract_id": ["one", "two"],
            "median_income": [50000, 70000],
            "median_rent": [np.nan, 1700],
        }
    ).to_parquet(census / "nyc_targets_2021.parquet", index=False)
    pd.DataFrame({"tract_id": ["one"], "poi_count": [12]}).to_parquet(
        features / "nyc_geo_features.parquet",
        index=False,
    )
    output = tmp_path / "dataset.parquet"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_dataset_multi_city.py"),
            "--city",
            "nyc",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    result = pd.read_parquet(output).set_index("tract_id")
    assert pd.isna(result.loc["one", "median_rent"])
    assert result.loc["two", "poi_count"] == 0
    assert np.isfinite(result[["lat", "lon"]].to_numpy()).all()
    source = json.loads(output.with_suffix(".source.json").read_text())
    assert source["acs_year"] == 2021
    assert source["acs_period"] == "2017–2021"


def test_single_city_preparation_reuses_inputs_and_keeps_downloads_optional(tmp_path):
    from scripts.prepare_city import preparation_steps

    first = preparation_steps("nyc", 2021, "ridge", root=tmp_path)
    assert len(first) == 6
    assert first[0][0] == "fetch_census_tracts_multi.py"
    assert first[1][0] == "download_acs_targets.py"
    assert first[2][0] == "download_osm_extract.py"
    assert first[-1][-1] == "all"
    with pytest.raises(ValueError, match="Missing local inputs"):
        preparation_steps("nyc", 2021, "ridge", skip_downloads=True, root=tmp_path)
    for relative in (
        "raw/census/nyc_tracts_2021.gpkg",
        "raw/census/nyc_targets_2021.parquet",
        "features/nyc_geo_features.parquet",
    ):
        path = tmp_path / "data" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "data/features/nyc_geo_features.source.json").write_text(
        json.dumps({"tract_geometry": "data/raw/census/nyc_tracts_2021.gpkg"})
    )
    reused = preparation_steps("nyc", 2021, "ridge", skip_downloads=True, root=tmp_path)
    assert [step[0] for step in reused] == ["build_dataset_multi_city.py", "cache_predictions.py"]
