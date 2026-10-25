"""Train-only model selection with predictions and scoring in original target units."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from scripts.model_zoo import get_model

MONEY_TARGETS = {"median_income", "median_rent", "median_home_value"}
RATE_TARGETS = {"poverty_rate", "unemployment_rate", "bachelors_plus_rate"}


class TractFeatureBasis(TransformerMixin, BaseEstimator):
    """Ridge basis learned only from the fitting fold's features and labels.

    Ridge remains a linear model in this explicit basis. Smooth geographic RBF
    terms allow local variation; optional splines allow nonlinear feature effects.
    Every fitted operation, including feature selection, runs inside inner CV.
    """

    def __init__(self, kind="linear", gamma=1.0, seed=42):
        self.kind = kind
        self.gamma = gamma
        self.seed = seed

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        values = self.imputer_.fit_transform(X)
        self.coordinate_indices_ = [
            i for i, name in enumerate(self.feature_names_in_) if name in {"lat", "lon"}
        ]
        self.other_indices_ = [
            i for i in range(self.n_features_in_) if i not in self.coordinate_indices_
        ]
        expanded = self._expand(values)
        self.scaler_ = StandardScaler().fit(expanded)
        scaled = self.scaler_.transform(expanded)
        self.splines_ = None
        self.spatial_ = None
        if self.kind == "spline_spatial":
            self.selector_ = SelectKBest(f_regression, k=min(20, scaled.shape[1]))
            selected = self.selector_.fit_transform(scaled, y)
            self.splines_ = SplineTransformer(
                n_knots=4, degree=3, include_bias=False, extrapolation="linear"
            ).fit(selected)
        if self.kind != "linear" and len(self.coordinate_indices_) == 2:
            coordinates = values[:, self.coordinate_indices_]
            self.coordinate_scaler_ = StandardScaler().fit(coordinates)
            self.spatial_ = Nystroem(
                kernel="rbf",
                gamma=self.gamma,
                n_components=min(64, len(values)),
                random_state=self.seed,
            ).fit(self.coordinate_scaler_.transform(coordinates))
        return self

    def _expand(self, values):
        extra = values[:, self.other_indices_]
        return np.column_stack([values, np.sign(extra) * np.log1p(np.abs(extra))])

    def transform(self, X):
        values = self.imputer_.transform(X)
        scaled = self.scaler_.transform(self._expand(values))
        parts = [scaled]
        if self.splines_ is not None:
            parts.append(self.splines_.transform(self.selector_.transform(scaled)))
        if self.spatial_ is not None:
            coordinates = self.coordinate_scaler_.transform(values[:, self.coordinate_indices_])
            parts.append(self.spatial_.transform(coordinates))
        return np.column_stack(parts)


class OriginalScaleRegressor(RegressorMixin, BaseEstimator):
    """Apply optional log training and domain bounds to predictions, never labels."""

    def __init__(self, estimator, target, target_transform="none"):
        self.estimator = estimator
        self.target = target
        self.target_transform = target_transform

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.estimator_ = clone(self.estimator)
        transformed = np.log1p(y) if self.target_transform == "log1p" else y
        self.estimator_.fit(X, transformed)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        prediction = np.asarray(self.estimator_.predict(X), dtype=float)
        if self.target_transform == "log1p":
            # Numerical guard only: log predictions above 30 already mean >$10T.
            prediction = np.expm1(np.clip(prediction, -50.0, 30.0))
        if self.target in RATE_TARGETS:
            prediction = np.clip(prediction, 0.0, 1.0)
        elif self.target in MONEY_TARGETS:
            prediction = np.maximum(prediction, 0.0)
        return prediction


def _candidates(model_name, target, seed, n_jobs):
    """Bounded, prespecified search; no held-out results influence this grid."""
    if model_name == "ridge":
        configurations = [
            ("linear", 1.0, 10.0, "none"),
            ("spatial", 0.3, 10.0, "none"),
            ("spatial", 2.0, 10.0, "none"),
            ("spline_spatial", 1.0, 10.0, "none"),
            ("spline_spatial", 1.0, 100.0, "none"),
            ("spatial", 1.0, 100.0, "none"),
        ]
        if target in MONEY_TARGETS:
            configurations.extend(
                [("spatial", 1.0, 10.0, "log1p"), ("spline_spatial", 1.0, 10.0, "log1p")]
            )
        for kind, gamma, alpha, transform in configurations:
            estimator = Pipeline(
                [
                    ("basis", TractFeatureBasis(kind=kind, gamma=gamma, seed=seed)),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        get_model(model_name, seed=seed, n_jobs=n_jobs).set_params(alpha=alpha),
                    ),
                ]
            )
            yield OriginalScaleRegressor(estimator, target, transform), {
                "basis": kind,
                "gamma": gamma,
                "alpha": alpha,
                "target_transform": transform,
            }
        return

    grids = {
        "extra_trees": [
            {"n_estimators": 256, "min_samples_leaf": 1, "max_features": 1.0},
            {"n_estimators": 256, "min_samples_leaf": 3, "max_features": 1.0},
            {"n_estimators": 256, "min_samples_leaf": 8, "max_features": 0.8},
            {"n_estimators": 256, "min_samples_leaf": 3, "max_features": 0.6},
        ],
        "hist_gb": [
            {
                "max_iter": 250,
                "max_leaf_nodes": 7,
                "max_depth": None,
                "min_samples_leaf": 5,
                "l2_regularization": 1.0,
            },
            {
                "max_iter": 250,
                "max_leaf_nodes": 15,
                "max_depth": None,
                "min_samples_leaf": 10,
                "l2_regularization": 5.0,
            },
            {
                "max_iter": 250,
                "max_leaf_nodes": 15,
                "max_depth": None,
                "min_samples_leaf": 20,
                "l2_regularization": 10.0,
            },
            {
                "max_iter": 350,
                "max_leaf_nodes": 7,
                "max_depth": None,
                "min_samples_leaf": 15,
                "l2_regularization": 20.0,
            },
        ],
        "xgboost": [
            {
                "n_estimators": 300,
                "max_depth": 2,
                "learning_rate": 0.04,
                "min_child_weight": 3,
                "reg_lambda": 10.0,
            },
            {
                "n_estimators": 400,
                "max_depth": 3,
                "learning_rate": 0.04,
                "min_child_weight": 5,
                "reg_lambda": 10.0,
            },
            {
                "n_estimators": 400,
                "max_depth": 4,
                "learning_rate": 0.04,
                "min_child_weight": 10,
                "reg_lambda": 20.0,
            },
            {
                "n_estimators": 300,
                "max_depth": 2,
                "learning_rate": 0.06,
                "min_child_weight": 10,
                "reg_lambda": 30.0,
            },
        ],
    }
    transforms = ("none", "log1p") if target in MONEY_TARGETS else ("none",)
    for parameters in grids.get(model_name, [{}]):
        for transform in transforms:
            model = get_model(model_name, seed=seed, n_jobs=n_jobs).set_params(**parameters)
            steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
            if model_name in {"elasticnet", "mlp"}:
                steps.append(("scale", StandardScaler()))
            steps.append(("model", model))
            yield OriginalScaleRegressor(Pipeline(steps), target, transform), {
                **parameters,
                "target_transform": transform,
            }


def _scores(y, prediction):
    return {
        "r2": float(r2_score(y, prediction)) if len(y) >= 2 else None,
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
    }


def select_and_fit_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target: str,
    *,
    seed: int = 42,
    cv_folds: int = 3,
    n_jobs: int = 2,
    cv_splits=None,
):
    """Select on training-fold OOF MSE, then refit on all supplied training rows.

    Explicit inner splits may be supplied for nested spatial validation. Every
    row must be validated exactly once and no fit/validation rows may overlap.
    The caller alone holds the outer test set; this API never receives it.
    """
    if len(X_train) != len(y_train) or len(y_train) < 3:
        raise ValueError("Model selection needs at least three aligned training rows")
    if n_jobs < 1 or cv_folds < 2:
        raise ValueError("n_jobs must be positive and cv_folds must be at least two")
    y = np.asarray(y_train, dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Non-finite target values must be excluded before selection")
    if target in MONEY_TARGETS and np.any(y < 0):
        raise ValueError("Monetary labels must be nonnegative")
    if target in RATE_TARGETS and np.any((y < 0) | (y > 1)):
        raise ValueError("Rate labels must lie in [0, 1]")
    splits = (
        list(cv_splits)
        if cv_splits is not None
        else list(
            KFold(n_splits=min(cv_folds, len(y)), shuffle=True, random_state=seed).split(X_train)
        )
    )
    if len(splits) < 2:
        raise ValueError("At least two inner validation folds are required")
    coverage = np.zeros(len(y), dtype=int)
    validated_splits = []
    for train_index, validation_index in splits:
        train_index, validation_index = np.asarray(train_index), np.asarray(validation_index)
        if any(
            index.ndim != 1 or index.dtype.kind not in "iu"
            for index in (train_index, validation_index)
        ):
            raise ValueError("Inner fold indices must be one-dimensional integer arrays")
        if not len(train_index) or not len(validation_index):
            raise ValueError("Inner folds must have training and validation rows")
        if any(len(np.unique(index)) != len(index) for index in (train_index, validation_index)):
            raise ValueError("Inner folds must not repeat row indices")
        if np.intersect1d(train_index, validation_index).size:
            raise ValueError("Inner training and validation rows overlap")
        if min(train_index.min(), validation_index.min()) < 0 or max(
            train_index.max(), validation_index.max()
        ) >= len(y):
            raise ValueError("Inner fold indices are out of bounds")
        if len(train_index) + len(validation_index) != len(y):
            raise ValueError("Each inner fold must partition all supplied training rows")
        coverage[validation_index] += 1
        validated_splits.append((train_index, validation_index))
    if not np.all(coverage == 1):
        raise ValueError("Inner validation folds must cover every training row exactly once")
    splits = validated_splits

    candidates = []
    best_estimator, best_rmse, best_index = None, np.inf, None
    with threadpool_limits(limits=n_jobs):
        for index, (estimator, configuration) in enumerate(
            _candidates(model_name, target, seed, n_jobs)
        ):
            prediction = np.full(len(y), np.nan)
            fold_scores = []
            for train_index, validation_index in splits:
                fitted = clone(estimator).fit(X_train.iloc[train_index], y[train_index])
                validation_prediction = fitted.predict(X_train.iloc[validation_index])
                if not np.isfinite(validation_prediction).all():
                    raise ValueError(f"Non-finite validation predictions from {model_name}")
                prediction[validation_index] = validation_prediction
                fold_scores.append(_scores(y[validation_index], validation_prediction))
            scores = _scores(y, prediction)
            candidates.append(
                {"parameters": configuration, "oof_metrics": scores, "fold_metrics": fold_scores}
            )
            if scores["rmse"] < best_rmse:
                best_estimator, best_rmse, best_index = estimator, scores["rmse"], index
        best_estimator.fit(X_train, y)

    selected = candidates[best_index]
    metadata = {
        "method": "cross_validation",
        "scope": "outer_training_rows_only",
        "folds": len(splits),
        "splitter": "provided_splits" if cv_splits is not None else "shuffled_kfold",
        "scoring": "neg_mean_squared_error",
        "aggregation": "pooled_out_of_fold_predictions_in_original_target_units",
        "candidate_count": len(candidates),
        "selected_candidate": best_index,
        "selected_parameters": selected["parameters"],
        "cv_metrics": selected["oof_metrics"],
        "candidates": candidates,
        "feature_names": list(X_train.columns),
        "preprocessing_scope": "fit_independently_inside_each_training_fold",
        "seed": seed,
    }
    return best_estimator, metadata
