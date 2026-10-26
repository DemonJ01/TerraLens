import numpy as np
import pandas as pd
import pytest

from terralens.data.comparison import align_model_predictions, paired_predictions


def frame(ids=("001", "002", "003"), predictions=(11.0, 18.0, 33.0)):
    positions = {"001": 0, "002": 1, "003": 2, "004": 3}
    return pd.DataFrame(
        {
            "tract_id": list(ids),
            "lat": [40 + positions[i] * 0.1 for i in ids],
            "lon": [-74 + positions[i] * 0.1 for i in ids],
            "y_true": [10 + positions[i] * 10 for i in ids],
            "y_pred": list(predictions),
            "error": 999,
        }
    )


def test_comparison_aligns_tract_ids_and_keeps_inputs_unchanged():
    a = frame()
    b = frame(("003", "001", "002"), (29, 12, 24))
    original = b.copy(deep=True)
    comparison = align_model_predictions({"a": a, "b": b})
    assert comparison.common_n == 3
    assert comparison.frames["b"].index.tolist() == ["001", "002", "003"]
    assert comparison.frames["b"].y_pred.tolist() == [12, 24, 29]
    pd.testing.assert_frame_equal(b, original)
    scores = comparison.scores.set_index("model")
    assert scores.loc["a", "mae"] == 2
    assert scores.loc["b", "mae"] == pytest.approx(7 / 3)
    paired = paired_predictions(comparison, "a", "b")
    assert paired.tract_id.tolist() == ["001", "002", "003"]
    assert paired.prediction_difference.tolist() == [-1, -6, 4]
    assert paired.absolute_error_difference.tolist() == [-1, -2, 2]
    assert paired.error_a.tolist() == [1, -2, 3]
    assert paired.error_b.tolist() == [2, 4, -1]


def test_partial_overlap_and_missing_values_use_identical_observations_for_all_models():
    a = frame(predictions=(11, np.nan, 33))
    b = frame(("002", "003", "004"), (22, 29, 35))
    comparison = align_model_predictions({"a": a, "b": b})
    assert comparison.common_n == 1
    assert comparison.frames["a"].index.tolist() == ["003"]
    counts = comparison.counts.set_index("model")
    assert counts.loc["a", "invalid_rows"] == 1
    assert counts.loc["a", "unmatched_valid_rows"] == 1
    assert counts.loc["b", "unmatched_valid_rows"] == 2
    assert counts.discarded_rows.tolist() == [2, 2]
    assert comparison.scores.n.tolist() == [1, 1]
    assert comparison.scores.r2.isna().all()


@pytest.mark.parametrize("column,value", [("y_true", 999), ("lat", 41.0), ("lon", -72.0)])
def test_conflicting_observations_or_coordinates_fail_explicitly(column, value):
    a, b = frame(), frame()
    b.loc[0, column] = value
    with pytest.raises(ValueError, match="differ on matching tracts"):
        align_model_predictions({"a": a, "b": b})


@pytest.mark.parametrize(
    "ids",
    [("001", "001", "003"), ("001", None, "003"), ("001", "002 ", "003"), (1, 2, 3)],
)
def test_duplicate_null_whitespace_and_nonstring_ids_are_rejected(ids):
    bad = frame()
    bad["tract_id"] = list(ids)
    with pytest.raises(ValueError, match="tract IDs"):
        align_model_predictions({"a": bad, "b": frame()})


def test_exact_copied_predictions_are_reported_even_after_reordering():
    a = frame()
    b = a.iloc[::-1].copy()
    comparison = align_model_predictions({"a": a, "b": b})
    row = comparison.pairwise.iloc[0]
    assert row.identical_predictions
    assert row.mean_absolute_difference == row.max_absolute_difference == 0
    b.loc[b.tract_id == "001", "y_pred"] += 0.000001
    comparison = align_model_predictions({"a": a, "b": b})
    row = comparison.pairwise.iloc[0]
    assert not row.identical_predictions
    assert row.max_absolute_difference == pytest.approx(0.000001)


def test_no_common_observations_and_empty_input_fail_cleanly():
    with pytest.raises(ValueError, match="At least one"):
        align_model_predictions({})
    with pytest.raises(ValueError, match="no common tracts"):
        align_model_predictions({"a": frame(("001",), (11,)), "b": frame(("004",), (40,))})
    unavailable = frame(predictions=(np.nan, np.inf, np.nan))
    with pytest.raises(ValueError, match="no common tracts"):
        align_model_predictions({"a": unavailable, "b": frame()})


def test_near_equal_source_values_allow_only_explicit_numeric_tolerance():
    a, b = frame(), frame()
    b["y_true"] += 1e-10
    b["lat"] += 1e-9
    align_model_predictions({"a": a, "b": b})
    with pytest.raises(ValueError, match="observed values"):
        align_model_predictions({"a": a, "b": b}, truth_atol=0, truth_rtol=0)


def test_invalid_coordinates_are_counted_and_pair_choices_validated():
    a = frame()
    a.loc[0, "lat"] = 91
    comparison = align_model_predictions({"a": a, "b": frame()})
    assert comparison.common_n == 2
    assert comparison.counts.iloc[0].invalid_rows == 1
    with pytest.raises(ValueError, match="different models"):
        paired_predictions(comparison, "a", "a")
    with pytest.raises(ValueError, match="exist"):
        paired_predictions(comparison, "a", "missing")
