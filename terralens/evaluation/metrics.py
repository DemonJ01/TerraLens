from __future__ import annotations

import numpy as np
import pandas as pd


def _finite_pairs(y_true, y_pred):
    """Validate vectors and keep only observations with two finite values."""
    try:
        truth = np.asarray(y_true, dtype=float)
        prediction = np.asarray(y_pred, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Targets and predictions must contain numeric values.") from exc
    if truth.ndim != 1 or prediction.ndim != 1:
        raise ValueError("Targets and predictions must be one-dimensional arrays.")
    if truth.shape != prediction.shape:
        raise ValueError("Targets and predictions must have the same length.")
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if not valid.any():
        raise ValueError("No finite target/prediction pairs are available for evaluation.")
    return truth[valid], prediction[valid]


def regression_metrics(y_true, y_pred):
    y_true, y_pred = _finite_pairs(y_true, y_pred)
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))

    # Match the usual finite R² convention for constant targets; a single
    # observation cannot define R² at all.
    if len(y_true) < 2:
        r2 = float("nan")
    else:
        residual = float(np.sum(np.square(error)))
        total = float(np.sum(np.square(y_true - y_true.mean())))
        r2 = float(1.0 - residual / total) if total > 0 else float(residual == 0)

    # Correlating ranks computes Spearman directly, including tied ranks,
    # without an optional SciPy dependency or undefined-correlation warnings.
    true_ranks = pd.Series(y_true).rank().to_numpy()
    pred_ranks = pd.Series(y_pred).rank().to_numpy()
    if len(y_true) > 1 and np.ptp(true_ranks) > 0 and np.ptp(pred_ranks) > 0:
        rho = float(np.corrcoef(true_ranks, pred_ranks)[0, 1])
    else:
        rho = None

    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": rho}
