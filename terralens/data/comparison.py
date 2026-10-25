"""Compare independently saved predictions on the same observed census tracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from terralens.data.dashboard import prediction_metrics

REQUIRED_COLUMNS = ("tract_id", "lat", "lon", "y_true", "y_pred")


@dataclass(frozen=True)
class ModelComparison:
    """Copied predictions and full-precision summaries on one common population.

    ``frames`` use sorted, exact string tract IDs as their index. ``scores`` and
    ``counts`` retain the mapping's model names; display labels belong in the UI.
    ``pairwise`` measures actual prediction differences, not rounded scores.
    """

    frames: dict[str, pd.DataFrame]
    scores: pd.DataFrame
    counts: pd.DataFrame
    pairwise: pd.DataFrame

    @property
    def common_n(self) -> int:
        return len(next(iter(self.frames.values()))) if self.frames else 0


def _validated_frame(model: str, frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{model}: missing prediction columns: {', '.join(sorted(missing))}")
    ids = frame["tract_id"]
    if ids.isna().any() or not ids.map(lambda value: isinstance(value, str)).all():
        raise ValueError(f"{model}: tract IDs must be non-null strings")
    if ids.map(lambda value: not value or value != value.strip()).any():
        raise ValueError(f"{model}: tract IDs must be nonempty and have no surrounding whitespace")
    if ids.duplicated().any():
        raise ValueError(f"{model}: duplicate tract IDs cannot be compared")
    result = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    valid = np.isfinite(result.loc[:, list(REQUIRED_COLUMNS[1:])].to_numpy(dtype=float)).all(axis=1)
    valid &= result.lat.between(-90, 90) & result.lon.between(-180, 180)
    result = result.loc[valid].set_index("tract_id").sort_index()
    # Error values in a source artifact may be stale. Always derive them anew.
    result["error"] = result.y_pred - result.y_true
    return result


def align_model_predictions(
    frames: Mapping[str, pd.DataFrame],
    *,
    truth_rtol: float = 1e-10,
    truth_atol: float = 1e-8,
    coordinate_atol: float = 1e-7,
) -> ModelComparison:
    """Align exact tract IDs and recompute metrics on finite common observations.

    Counts distinguish invalid observations from otherwise valid observations
    absent in another model. Matching tracts must agree on published truth and
    coordinates; contradictory artifacts fail explicitly instead of silently
    selecting one model's labels. Inputs are never modified.
    """
    if not frames:
        raise ValueError("At least one model's predictions are required")
    if any(not isinstance(model, str) or not model for model in frames):
        raise ValueError("Model names must be nonempty strings")
    if min(truth_rtol, truth_atol, coordinate_atol) < 0:
        raise ValueError("Comparison tolerances must be nonnegative")

    validated = {model: _validated_frame(model, frame) for model, frame in frames.items()}
    for model_a, model_b in combinations(validated, 2):
        a, b = validated[model_a], validated[model_b]
        shared = a.index.intersection(b.index).sort_values()
        if shared.empty:
            continue
        if not np.allclose(
            a.loc[shared, "y_true"],
            b.loc[shared, "y_true"],
            rtol=truth_rtol,
            atol=truth_atol,
        ):
            raise ValueError(f"{model_a} and {model_b}: observed values differ on matching tracts")
        if not np.allclose(
            a.loc[shared, ["lat", "lon"]],
            b.loc[shared, ["lat", "lon"]],
            rtol=0,
            atol=coordinate_atol,
        ):
            raise ValueError(f"{model_a} and {model_b}: coordinates differ on matching tracts")

    common = next(iter(validated.values())).index
    for frame in validated.values():
        common = common.intersection(frame.index)
    common = common.sort_values()
    if common.empty:
        raise ValueError("Models have no common tracts with finite observed and predicted values")

    aligned = {model: frame.loc[common].copy() for model, frame in validated.items()}
    scores = pd.DataFrame(
        [{"model": model, **prediction_metrics(frame)} for model, frame in aligned.items()]
    )
    counts = pd.DataFrame(
        [
            {
                "model": model,
                "original_rows": len(frames[model]),
                "valid_rows": len(validated[model]),
                "compared_rows": len(common),
                "discarded_rows": len(frames[model]) - len(common),
                "invalid_rows": len(frames[model]) - len(validated[model]),
                "unmatched_valid_rows": len(validated[model]) - len(common),
            }
            for model in aligned
        ]
    )
    pairwise = []
    for model_a, model_b in combinations(aligned, 2):
        a, b = aligned[model_a].y_pred.to_numpy(), aligned[model_b].y_pred.to_numpy()
        difference = np.abs(a - b)
        pairwise.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "identical_predictions": bool(np.array_equal(a, b)),
                "max_absolute_difference": float(difference.max()),
                "mean_absolute_difference": float(difference.mean()),
            }
        )
    return ModelComparison(
        frames=aligned,
        scores=scores,
        counts=counts,
        pairwise=pd.DataFrame(
            pairwise,
            columns=[
                "model_a",
                "model_b",
                "identical_predictions",
                "max_absolute_difference",
                "mean_absolute_difference",
            ],
        ),
    )


def paired_predictions(comparison: ModelComparison, model_a: str, model_b: str) -> pd.DataFrame:
    """Return per-tract predictions and signed differences for two distinct models.

    A positive prediction difference means A estimates a higher value. A positive
    absolute-error difference means A is less accurate on that tract. Values are
    kept in the target's original units and are not rounded or amplified.
    """
    if model_a == model_b:
        raise ValueError("Choose two different models for a paired comparison")
    if model_a not in comparison.frames or model_b not in comparison.frames:
        raise ValueError("Both models must exist in the aligned comparison")
    a, b = comparison.frames[model_a], comparison.frames[model_b]
    result = a[["lat", "lon"]].copy()
    result["observed"] = a.y_true
    result["prediction_a"] = a.y_pred
    result["prediction_b"] = b.y_pred
    result["error_a"] = a.y_pred - a.y_true
    result["error_b"] = b.y_pred - b.y_true
    result["prediction_difference"] = result.prediction_a - result.prediction_b
    result["absolute_error_difference"] = result.error_a.abs() - result.error_b.abs()
    return result.reset_index()
