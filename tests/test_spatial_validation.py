"""Local regression tests for leakage boundaries and geographic validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from scripts.validate_models import evaluate_model, geographic_blocks, tract_id_hash


def frame():
    points = [(x, y) for y in range(4) for x in range(4) for _ in range(4)]
    return pd.DataFrame(
        {
            "tract_id": [f"{i:011d}" for i in range(len(points))],
            "lat": [37.0 + y / 100 for x, y in points],
            "lon": [-122.0 + x / 100 for x, y in points],
            "signal": np.arange(len(points), dtype=float),
            "median_income": np.arange(len(points), dtype=float) * 10 + 1000,
        }
    )


def test_grid_uses_only_coordinates_and_keeps_edges_inside():
    d = frame()
    original, metadata = geographic_blocks(d, 4)
    d["median_income"] = d.median_income.sample(frac=1, random_state=4).to_numpy()
    changed, _ = geographic_blocks(d, 4)
    np.testing.assert_array_equal(original, changed)
    assert set(original) == set(range(16))
    assert metadata["occupied_blocks"] == 16
    assert metadata["uses_target_values"] is False
    assert metadata["buffer_distance"] is None


def test_nested_selection_receives_training_blocks_only(monkeypatch):
    import scripts.model_selection

    d = frame()
    blocks, _ = geographic_blocks(d, 4)
    calls = []

    def selector(model_name, X_train, y_train, target, *, seed, cv_folds, n_jobs, cv_splits):
        ids = X_train.signal.astype(int).to_numpy()
        local_groups = blocks[ids]
        coverage = np.zeros(len(ids), dtype=int)
        for train, validation in cv_splits:
            assert not (set(local_groups[train]) & set(local_groups[validation]))
            assert not (set(train) & set(validation))
            coverage[validation] += 1
        assert (coverage == 1).all()
        calls.append(set(ids))
        model = DummyRegressor(strategy="mean").fit(X_train, y_train)
        return model, {"scope": "outer_training_rows_only", "splitter": "provided_splits"}

    monkeypatch.setattr(scripts.model_selection, "select_and_fit_model", selector)
    predictions, details = evaluate_model(
        d,
        ["lat", "lon", "signal"],
        "median_income",
        "ridge",
        outer_folds=3,
        inner_folds=3,
        n_jobs=1,
    )
    assert len(calls) == 3
    assert set(predictions.tract_id) == set(d.tract_id)
    assert predictions.outer_fold.nunique() == 3
    for fold, seen in enumerate(calls):
        test = predictions.loc[predictions.outer_fold == fold].index
        assert not (set(test) & seen)
        expected = d.loc[list(seen), "median_income"].mean()
        np.testing.assert_allclose(predictions.loc[test, "training_mean_prediction"], expected)
        assert not (
            set(details["folds"][fold]["train_blocks"]) & set(details["folds"][fold]["test_blocks"])
        )
    np.testing.assert_allclose(predictions.y_pred, predictions.training_mean_prediction)
    assert details["model_metrics"] == details["training_mean_baseline_metrics"]
    assert details["mse_skill_vs_training_mean"] == 0


def test_target_columns_cannot_be_validation_features():
    with pytest.raises(ValueError, match="cannot be predictors"):
        evaluate_model(frame(), ["lat", "lon", "median_income"], "median_income", "ridge")


def test_spatial_grid_rejects_invalid_coordinates():
    d = frame()
    d.loc[0, "lon"] = np.nan
    with pytest.raises(ValueError, match="finite tract coordinates"):
        geographic_blocks(d)


def test_insufficient_spatial_groups_fails_instead_of_random_fallback():
    d = frame()
    d["lat"] = 37.0
    d["lon"] = -122.0
    with pytest.raises(ValueError, match="Too few occupied spatial blocks"):
        evaluate_model(d, ["lat", "lon", "signal"], "median_income", "ridge")


def test_hash_is_order_independent():
    assert tract_id_hash(["001", "002"]) == tract_id_hash(["002", "001"])
    assert tract_id_hash(["001", "002"]) != tract_id_hash(["001", "003"])


def test_cli_records_exact_dataset_and_prediction_hashes(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys
    from scripts import validate_models

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dataset_path = data_dir / "san_francisco_dataset.parquet"
    d = frame()
    d.to_parquet(dataset_path, index=False)
    expected_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    out_dir = tmp_path / "spatial"

    def evaluate(data, feature_names, target, model_name, **kwargs):
        result = data[["tract_id", "lat", "lon"]].copy()
        result["y_true"] = data.median_income
        result["y_pred"] = data.median_income + 10
        return result, {
            "model_metrics": {"r2": 0.9, "mae": 10.0, "rmse": 10.0},
            "training_mean_baseline_metrics": {"r2": 0.0, "mae": 100.0, "rmse": 110.0},
            "mse_skill_vs_training_mean": 0.99,
            "fold_r2_std": 0.01,
            "rows": len(data),
        }

    monkeypatch.setattr(validate_models, "evaluate_model", evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_models.py",
            "--city",
            "san_francisco",
            "--models",
            "ridge",
            "--targets",
            "median_income",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert validate_models.main() == 0
    artifact = out_dir / "san_francisco__ridge__osm_only__median_income.spatial.parquet"
    details = json.loads(artifact.with_suffix(".json").read_text())
    assert details["dataset_sha256"] == expected_hash
    assert details["prediction_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert details["dataset_path"] == str(dataset_path)
    assert {"numpy", "pandas", "scikit-learn"} <= details["package_versions"].keys()


def test_cli_refuses_to_certify_changed_dataset(tmp_path, monkeypatch):
    import sys
    from scripts import validate_models

    dataset_path = tmp_path / "san_francisco_dataset.parquet"
    d = frame()
    d.to_parquet(dataset_path, index=False)
    out_dir = tmp_path / "spatial"

    def evaluate(data, feature_names, target, model_name, **kwargs):
        changed = d.copy()
        changed.loc[0, "median_income"] += 1
        changed.to_parquet(dataset_path, index=False)
        return pd.DataFrame(), {}

    monkeypatch.setattr(validate_models, "evaluate_model", evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_models.py",
            "--city",
            "san_francisco",
            "--models",
            "ridge",
            "--targets",
            "median_income",
            "--data-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
        ],
    )
    with pytest.raises(RuntimeError, match="dataset changed during spatial validation"):
        validate_models.main()
    assert not list(out_dir.glob("*.spatial.json"))
    assert not list(out_dir.glob("*.spatial.parquet"))
