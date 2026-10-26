from pathlib import Path
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from terralens.data.comparison import align_model_predictions, paired_predictions
from terralens.ui.comparison import score_chart, difference_chart


def chart_rows(spec):
    return [row for data in spec.get("datasets", {}).values() for row in data]


def test_score_chart_displays_precise_measurements_and_zero_reference():
    scores = pd.DataFrame(
        {
            "model": ["ridge", "extra_trees"],
            "r2": [0.533834296258, 0.535667937762],
            "mae": [0.06123456, 0.05987543],
            "rmse": [0.08, 0.079],
            "n": [264, 264],
        }
    )
    spec = score_chart(scores, "R²", {}, True).to_dict()
    points = [row for row in chart_rows(spec) if "Value" in row]
    assert [row["Label"] for row in points] == ["0.535668", "0.533834"]
    assert np.allclose([row["Value"] for row in points], [0.535667937762, 0.533834296258])
    assert spec["layer"][1]["encoding"]["x"]["scale"]["domain"][0] == 0
    mae = score_chart(scores, "MAE", {}, True).to_dict()
    points = [row for row in chart_rows(mae) if "Value" in row]
    assert np.isclose(points[0]["Value"], 5.987543)
    assert mae["layer"][1]["encoding"]["x"]["title"] == "MAE (percentage points)"


def test_difference_chart_uses_actual_signed_gaps_in_percentage_points():
    data = pd.DataFrame(
        {
            "tract_id": ["001", "002"],
            "lat": [40.0, 41.0],
            "lon": [-74.0, -73.0],
            "y_true": [0.1, 0.2],
            "y_pred": [0.12, 0.16],
        }
    )
    other = data.assign(y_pred=[0.1, 0.22]).iloc[::-1]
    c = align_model_predictions({"ridge": data, "extra_trees": other})
    pair = paired_predictions(c, "ridge", "extra_trees")
    spec = difference_chart(pair, rate=True, model_a="Ridge", model_b="Extra Trees").to_dict()
    points = [row for row in chart_rows(spec) if "Gap" in row]
    assert points[0]["tract_id"] == "002"
    assert np.isclose(points[0]["Gap"], -6.0)
    assert np.isclose(points[1]["Gap"], 2.0)
    assert np.allclose(pair.prediction_difference, [0.02, -0.06])


def test_identity_conflicts_do_not_produce_misleading_leaderboard(tmp_path, monkeypatch):
    from terralens.data import dashboard

    folder = tmp_path / "outputs/preds"
    folder.mkdir(parents=True)
    for model, observed in [("ridge", [100.0, 200.0]), ("extra_trees", [100.0, 220.0])]:
        pd.DataFrame(
            {
                "tract_id": ["a", "b"],
                "lat": [40.7, 40.8],
                "lon": [-74.0, -73.9],
                "y_true": observed,
                "y_pred": [110.0, 180.0],
            }
        ).to_parquet(folder / f"nyc__{model}__osm_only__median_income.parquet")
    original = dashboard.discover_predictions
    monkeypatch.setattr(dashboard, "discover_predictions", lambda: original(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    assert not app.dataframe
    assert any("observed values differ" in w.value for w in app.warning)


def test_score_labels_use_visible_theme_color_and_face_into_plot():
    scores = pd.DataFrame(
        {
            "model": ["ridge", "extra_trees"],
            "r2": [-0.65, 0.79],
            "mae": [0.06, 0.05],
            "rmse": [0.12, 0.08],
            "n": [47, 47],
        }
    )
    for color in ["#e6edf3", "#172033"]:
        spec = score_chart(scores, "R²", {}, True, text_color=color).to_dict()
        labels = [layer for layer in spec["layer"] if layer["mark"]["type"] == "text"]
        assert len(labels) == 2
        assert all(layer["mark"]["color"] == color for layer in labels)
        assert [(layer["mark"]["align"], layer["mark"]["dx"]) for layer in labels] == [
            ("right", -12),
            ("left", 12),
        ]
