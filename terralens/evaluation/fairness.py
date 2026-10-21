from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import _finite_pairs


def errors_by_income_quartile(y_true, y_pred):
    y_true, y_pred = _finite_pairs(y_true, y_pred)

    df = pd.DataFrame(
        {
            "true": y_true,
            "pred": y_pred,
            "err": y_pred - y_true,
            "abs_err": np.abs(y_pred - y_true),
        }
    )

    # Ties can collapse quantile boundaries. Generate labels after identifying
    # the actual bins rather than forcing four labels onto fewer intervals.
    if df["true"].nunique() == 1:
        df["quartile"] = "All values"
    else:
        bins, edges = pd.qcut(
            df["true"], q=min(4, len(df)), labels=False, retbins=True, duplicates="drop"
        )
        count = len(edges) - 1
        labels = [f"Q{i + 1}" for i in range(count)]
        labels[0] += " (Low)"
        labels[-1] += " (High)"
        df["quartile"] = pd.Categorical.from_codes(bins.astype(int), labels, ordered=True)

    summary = (
        df.groupby("quartile", observed=True)
        .agg(
            mae=("abs_err", "mean"),
            rmse=("err", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            bias=("err", "mean"),
            income_min=("true", "min"),
            income_max=("true", "max"),
            income_mean=("true", "mean"),
            n=("true", "count"),
        )
        .reset_index()
    )

    return summary


def save_fairness_report(y_true, y_pred, out_json: str | Path, metadata=None):
    y_true, y_pred = _finite_pairs(y_true, y_pred)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    summary = errors_by_income_quartile(y_true, y_pred)

    report = {
        "metadata": metadata or {},
        "overall": {
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "bias": float(np.mean(y_pred - y_true)),
        },
        "by_quartile": summary.to_dict(orient="records"),
    }

    with out_json.open("w") as f:
        json.dump(report, f, indent=2)

    out_csv = out_json.with_suffix(".csv")
    summary.to_csv(out_csv, index=False)

    print(f"✅ Saved fairness report: {out_json}")
    print(f"✅ Saved fairness table:  {out_csv}")
    return report
