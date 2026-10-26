"""Dataset identity, geographic validity, and mutation guards for training."""

import hashlib
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from scripts import cache_predictions
from scripts.cache_predictions import load_dataset
from scripts.model_selection import OriginalScaleRegressor


def dataset(rows=12):
    return pd.DataFrame(
        {
            "tract_id": [f"{index:011d}" for index in range(rows)],
            "lat": np.linspace(40.6, 40.8, rows),
            "lon": np.linspace(-74.1, -73.9, rows),
            "poi_count": np.arange(rows),
            "median_income": np.linspace(20_000, 80_000, rows),
        }
    )


def write_dataset(tmp_path, df):
    path = tmp_path / "nyc_dataset.parquet"
    df.to_parquet(path, index=False)
    return path


def test_loader_preserves_leading_zero_ids_and_hashes_exact_loaded_bytes(tmp_path):
    path = write_dataset(tmp_path, dataset())
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_dataset("nyc", data_dir=tmp_path)
    assert loaded["tract_id"].tolist() == dataset()["tract_id"].tolist()
    assert all(isinstance(value, str) for value in loaded["tract_id"])
    assert loaded.attrs["dataset_sha256"] == expected_hash
    assert loaded.attrs["dataset_path"] == str(path)


@pytest.mark.parametrize("bad_id", [None, "", "  ", "00000000001"])
def test_loader_rejects_null_blank_or_duplicate_tract_ids(tmp_path, bad_id):
    df = dataset()
    df.loc[0, "tract_id"] = bad_id
    write_dataset(tmp_path, df)
    with pytest.raises(ValueError, match="tract IDs"):
        load_dataset("nyc", data_dir=tmp_path)


@pytest.mark.parametrize(
    "coordinate,value",
    [
        ("lat", np.nan),
        ("lat", np.inf),
        ("lat", 90.01),
        ("lat", -90.01),
        ("lon", 180.01),
        ("lon", -180.01),
        ("lon", -np.inf),
        ("lon", None),
    ],
)
def test_loader_rejects_invalid_coordinates_instead_of_imputing_them(tmp_path, coordinate, value):
    df = dataset()
    df.loc[0, coordinate] = value
    write_dataset(tmp_path, df)
    with pytest.raises(ValueError, match=f"invalid {coordinate}"):
        load_dataset("nyc", data_dir=tmp_path)


def test_loader_detects_source_mutation_while_parquet_is_being_parsed(tmp_path, monkeypatch):
    path = write_dataset(tmp_path, dataset())
    read_parquet = pd.read_parquet

    def mutate_after_parse(*args, **kwargs):
        loaded = read_parquet(*args, **kwargs)
        changed = dataset()
        changed["median_income"] += 1
        changed.to_parquet(path, index=False)
        return loaded

    monkeypatch.setattr(cache_predictions.pd, "read_parquet", mutate_after_parse)
    with pytest.raises(ValueError, match="Dataset changed"):
        load_dataset("nyc", data_dir=tmp_path)


def test_cache_does_not_save_predictions_after_dataset_changes_during_fit(tmp_path, monkeypatch):
    path = write_dataset(tmp_path, dataset())
    output = tmp_path / "predictions"

    def mutate_during_fit(model_name, X, y, target, **kwargs):
        fitted = OriginalScaleRegressor(DummyRegressor(), target).fit(X, y)
        changed = dataset()
        changed["median_income"] += 1
        changed.to_parquet(path, index=False)
        return fitted, {}

    monkeypatch.setattr(cache_predictions, "select_and_fit_model", mutate_during_fit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cache_predictions.py",
            "--city",
            "nyc",
            "--targets",
            "median_income",
            "--models",
            "ridge",
            "--data-dir",
            str(tmp_path),
            "--out-dir",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="Dataset changed"):
        cache_predictions.main()
    assert not list(output.glob("*.parquet"))
    assert not (output / "manifest.csv").exists()
