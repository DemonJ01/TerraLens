import json
from pathlib import Path

import pytest

from scripts.prepare_city import preparation_steps


def local_inputs(root):
    for name in [
        "data/raw/census/nyc_tracts_2021.gpkg",
        "data/raw/census/nyc_targets_2021.parquet",
        "data/features/nyc_geo_features.parquet",
    ]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (root / "data/features/nyc_geo_features.source.json").write_text(
        json.dumps({"tract_geometry": "data/raw/census/nyc_tracts_2021.gpkg"})
    )


def test_rebuilding_features_requires_raw_input_with_downloads_disabled(tmp_path):
    local_inputs(tmp_path)
    steps = preparation_steps("nyc", 2021, "ridge", skip_downloads=True, root=tmp_path)
    assert not any(step[0] == "compute_osm_features_from_pbf.py" for step in steps)
    with pytest.raises(ValueError, match="nyc.osm.pbf"):
        preparation_steps(
            "nyc", 2021, "ridge", skip_downloads=True, root=tmp_path, rebuild_features=True
        )
    raw = tmp_path / "data/raw/osm/nyc.osm.pbf"
    raw.parent.mkdir(parents=True)
    raw.touch()
    steps = preparation_steps(
        "nyc", 2021, "ridge", skip_downloads=True, root=tmp_path, rebuild_features=True
    )
    scripts = [step[0] for step in steps]
    assert scripts == [
        "compute_osm_features_from_pbf.py",
        "build_dataset_multi_city.py",
        "cache_predictions.py",
    ]
    assert "--force" in steps[0]


def test_prediction_metadata_rejects_replaced_artifacts(tmp_path):
    import json
    from hashlib import sha256
    from terralens.data.dashboard import load_prediction_metadata

    artifact = tmp_path / "result.parquet"
    artifact.write_bytes(b"first prediction version")
    sidecar = artifact.with_suffix(".metrics.json")
    metadata = {
        "evaluation": "random_holdout",
        "prediction_sha256": sha256(artifact.read_bytes()).hexdigest(),
    }
    sidecar.write_text(json.dumps(metadata))
    assert load_prediction_metadata(artifact) == metadata
    artifact.write_bytes(b"replacement predictions")
    assert load_prediction_metadata(artifact) == {}
    sidecar.write_text("[]")
    assert load_prediction_metadata(artifact) == {}
    sidecar.write_text("bad json")
    assert load_prediction_metadata(artifact) == {}


def test_dashboard_baseline_uses_training_average(tmp_path, monkeypatch):
    import json
    import pandas as pd
    from streamlit.testing.v1 import AppTest
    from terralens.data import dashboard

    pred_dir = tmp_path / "outputs/preds"
    pred_dir.mkdir(parents=True)
    path = pred_dir / "nyc__ridge__osm_only__median_income.parquet"
    pd.DataFrame(
        {
            "tract_id": ["a", "b"],
            "lat": [40.7, 40.8],
            "lon": [-74.0, -73.9],
            "y_true": [100.0, 200.0],
            "y_pred": [110.0, 180.0],
        }
    ).to_parquet(path)
    path.with_suffix(".metrics.json").write_text(
        json.dumps(
            {
                "evaluation": "random_holdout",
                "baseline": {"strategy": "training_mean", "value": 150.0},
                "model_selection": {"method": "cross_validation", "folds": 3},
            }
        )
    )
    original = dashboard.discover_predictions
    monkeypatch.setattr(dashboard, "discover_predictions", lambda: original(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    metric = next(metric for metric in app.metric if metric.label == "Baseline RMSE")
    assert metric.value == "50"
    assert any("90.0% lower squared error" in element.value for element in app.markdown)
    assert any("3-fold cross-validation" in element.value for element in app.caption)


def test_spatial_metadata_rejects_stale_datasets_and_modified_predictions(tmp_path):
    import json
    from hashlib import sha256
    from terralens.data.dashboard import load_spatial_metadata

    path = tmp_path / "outputs/preds/nyc__ridge__osm_only__median_income.parquet"
    identity = {
        "city": "nyc",
        "model": "ridge",
        "feature_set": "osm_only",
        "target": "median_income",
        "dataset_sha256": "dataset-1",
    }
    sidecar = path.parent.parent / "spatial_validation" / f"{path.stem}.spatial.json"
    sidecar.parent.mkdir(parents=True)
    spatial_artifact = sidecar.with_suffix(".parquet")
    spatial_artifact.write_bytes(b"spatial predictions")
    details = {
        **identity,
        "evaluation": "nested_spatial_block_cv",
        "prediction_sha256": sha256(spatial_artifact.read_bytes()).hexdigest(),
    }
    sidecar.write_text(json.dumps(details))
    assert load_spatial_metadata(path, identity) == details
    assert load_spatial_metadata(path, {**identity, "dataset_sha256": "dataset-2"}) == {}
    spatial_artifact.write_bytes(b"replacement predictions")
    assert load_spatial_metadata(path, identity) == {}
