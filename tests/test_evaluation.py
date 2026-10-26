import json
import math

import numpy as np
import pytest

from terralens.evaluation.fairness import (
    errors_by_income_quartile,
    save_fairness_report,
)
from terralens.evaluation.metrics import regression_metrics


def test_metrics_evaluate_only_aligned_finite_pairs():
    result = regression_metrics([1, 2, np.nan, 4, np.inf], [1, 3, 6, np.nan, 8])
    assert result["mae"] == pytest.approx(0.5)
    assert result["rmse"] == pytest.approx(math.sqrt(0.5))
    assert result["r2"] == pytest.approx(-1)
    assert result["spearman"] == pytest.approx(1)


def test_metrics_small_and_constant_samples():
    one = regression_metrics([5], [7])
    assert math.isnan(one["r2"])
    assert one["spearman"] is None
    assert regression_metrics([5, 5], [5, 5])["r2"] == 1
    assert regression_metrics([5, 5], [4, 5])["r2"] == 0


@pytest.mark.parametrize(
    "truth,prediction,message",
    [
        ([], [], "No finite"),
        ([np.nan], [1], "No finite"),
        ([1, 2], [1], "same length"),
        ([[1]], [[1]], "one-dimensional"),
    ],
)
def test_evaluation_rejects_invalid_vectors(truth, prediction, message):
    for evaluate in (regression_metrics, errors_by_income_quartile):
        with pytest.raises(ValueError, match=message):
            evaluate(truth, prediction)


@pytest.mark.parametrize("truth", [[5], [5] * 12, [1] * 9 + [2] * 3, [1, 2]])
def test_quartiles_handle_small_or_tied_samples(truth):
    summary = errors_by_income_quartile(truth, np.asarray(truth) + 2)
    assert summary["n"].sum() == len(truth)
    assert summary["mae"].to_numpy() == pytest.approx(2)
    assert summary["bias"].to_numpy() == pytest.approx(2)
    assert summary["quartile"].notna().all()


def test_fairness_report_overall_and_groups_use_same_finite_rows(tmp_path):
    destination = tmp_path / "reports" / "fairness.json"
    report = save_fairness_report([1, 2, np.nan, 4], [2, 4, 8, np.inf], destination)
    assert report["overall"]["mae"] == pytest.approx(1.5)
    assert sum(row["n"] for row in report["by_quartile"]) == 2
    assert json.loads(destination.read_text()) == report
    assert destination.with_suffix(".csv").exists()
