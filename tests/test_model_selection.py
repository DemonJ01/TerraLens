"""Checks for fold-local selection, original-unit scoring, and target validity."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold

from scripts import model_selection
from scripts.cache_predictions import select_features
from scripts.model_selection import OriginalScaleRegressor, TractFeatureBasis, select_and_fit_model


def features(rows=12):
    return pd.DataFrame(
        {
            "lat": np.linspace(40, 41, rows),
            "lon": np.linspace(-74, -73, rows),
            "poi_count": np.arange(rows, dtype=float),
        }
    )


def test_cv_uses_fold_training_mean_and_reports_pooled_original_unit_errors(monkeypatch):
    X = features()
    y = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0])
    splits = list(KFold(n_splits=3, shuffle=True, random_state=8).split(X))

    def candidates(*args):
        for transform in ["none", "log1p"]:
            yield OriginalScaleRegressor(DummyRegressor(), "median_income", transform), {
                "target_transform": transform
            }

    monkeypatch.setattr(model_selection, "_candidates", candidates)
    fitted, metadata = select_and_fit_model("ridge", X, y, "median_income", cv_splits=splits)
    errors = []
    for transform in ["none", "log1p"]:
        expected = np.zeros(len(y))
        for training, validation in splits:
            value = (
                y.iloc[training].mean()
                if transform == "none"
                else np.expm1(np.log1p(y.iloc[training]).mean())
            )
            expected[validation] = value
        errors.append(float(np.sqrt(np.mean((expected - y) ** 2))))
    assert [c["oof_metrics"]["rmse"] for c in metadata["candidates"]] == pytest.approx(errors)
    assert metadata["selected_candidate"] == int(np.argmin(errors))
    assert fitted.target_transform == "none"
    assert fitted.predict(X).tolist() == pytest.approx([y.mean()] * len(y))
    assert metadata["preprocessing_scope"] == "fit_independently_inside_each_training_fold"


def test_invalid_rate_labels_are_rejected_instead_of_clipped():
    with pytest.raises(ValueError, match="Rate labels"):
        select_and_fit_model("ridge", features(), pd.Series([2.0] * 12), "poverty_rate")


def test_overlapping_inner_training_validation_rows_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        select_and_fit_model(
            "ridge",
            features(),
            pd.Series(np.linspace(0, 1, 12)),
            "poverty_rate",
            cv_splits=[(np.arange(8), np.arange(4)), (np.arange(4), np.arange(4, 12))],
        )


def test_imputation_and_geographic_basis_fit_only_on_supplied_training_rows():
    X = features(8)
    X.loc[0, "poi_count"] = np.nan
    basis = TractFeatureBasis(kind="spatial").fit(X, np.arange(8))
    before = basis.imputer_.statistics_.copy()
    unseen = features(2)
    unseen["poi_count"] = [np.nan, 1.0e9]
    transformed = basis.transform(unseen)
    assert np.isfinite(transformed).all()
    assert basis.imputer_.statistics_ == pytest.approx(before)
    assert before[2] == pytest.approx(4.0)


def test_repeated_row_indices_cannot_bias_inner_validation():
    with pytest.raises(ValueError, match="repeat"):
        select_and_fit_model(
            "ridge",
            features(),
            pd.Series(np.linspace(0, 1, 12)),
            "poverty_rate",
            cv_splits=[
                (np.arange(4, 12), np.array([0, 1, 2, 2, 3])),
                (np.arange(4), np.arange(4, 12)),
            ],
        )


def test_osm_service_roads_and_pois_are_not_misclassified_as_complaints():
    X = features()
    X["road_service_length_m"] = 20
    X["road_service_density"] = 0.1
    X["poi_services_count"] = 10
    X["poi_services_density"] = 0.2
    X["service_request_count"] = 4
    X["complaint_count"] = 5
    X["nyc311_count"] = 6
    selected = select_features(X, "osm_only")
    assert {
        "road_service_length_m",
        "road_service_density",
        "poi_services_count",
        "poi_services_density",
    }.issubset(selected)
    assert not {"service_request_count", "complaint_count", "nyc311_count"}.intersection(selected)
