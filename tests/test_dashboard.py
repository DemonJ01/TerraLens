from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from terralens.data import dashboard
from terralens.utils.maps import build_prediction_map


def write_predictions(root, name="nyc__ridge__osm_only__median_income", **columns):
    path = root / "outputs" / "preds" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "lat": [40.7, 40.8],
            "lon": [-74.0, -73.9],
            "y_true": [100.0, 200.0],
            "y_pred": [110.0, 180.0],
            **columns,
        }
    ).to_parquet(path, index=False)
    return path


def test_missing_stale_and_corrupt_manifests_recover_real_predictions(tmp_path):
    path = write_predictions(tmp_path)
    manifest = path.parent / "manifest.csv"
    for content in [
        "",
        "bad,columns\n1,2\n",
        "city,model,feature_set,target,path\nnyc,ridge,osm_only,median_income,missing.parquet\n",
    ]:
        manifest.write_text(content)
        found = dashboard.discover_predictions(tmp_path)
        assert found.city.tolist() == ["nyc"]
        assert found.path.tolist() == [str(path)]


def test_new_real_results_replace_sample_without_editing_manifest(tmp_path):
    samples = dashboard.discover_predictions(tmp_path)
    assert samples.city.str.endswith("_sample").all()
    write_predictions(tmp_path)
    found = dashboard.discover_predictions(tmp_path)
    assert found.city.tolist() == ["nyc"]


def test_bad_files_are_skipped_and_empty_install_has_sample(tmp_path):
    folder = tmp_path / "outputs" / "preds"
    folder.mkdir(parents=True)
    (folder / "nyc__ridge__osm_only__median_income.parquet").write_bytes(b"broken parquet")
    found = dashboard.discover_predictions(tmp_path)
    assert len(found) == 3
    assert all(not dashboard.load_predictions(p).empty for p in found.path)


def test_invalid_coordinates_and_stale_errors_are_cleaned(tmp_path):
    path = write_predictions(tmp_path, lat=[float("inf"), 40.8], error=[999, 999])
    loaded = dashboard.load_predictions(path)
    assert len(loaded) == 1
    assert loaded.error.iloc[0] == -20
    assert loaded.tract_id.iloc[0] == "area-1"


def test_metrics_use_identical_valid_pairs():
    result = dashboard.prediction_metrics(
        pd.DataFrame(
            {
                "y_true": [10, 20, 1000, np.nan],
                "y_pred": [12, 18, np.inf, 1],
            }
        )
    )
    assert result == {"n": 2, "mae": 2, "rmse": 2, "r2": 0.84}
    assert dashboard.prediction_metrics(pd.DataFrame({"y_true": [np.nan], "y_pred": [1]}))["n"] == 0


@pytest.mark.parametrize("mode", ["Cluster", "Points", "Heatmap"])
def test_maps_handle_constant_values_without_label_overlay(tmp_path, mode):
    data = dashboard.load_predictions(write_predictions(tmp_path, y_true=[1, 1], y_pred=[1, 1]))
    rendered = build_prediction_map(data, "error", mode).get_root().render()
    assert "divIcon" not in rendered
    assert "NaN" not in rendered
    assert "openstreetmap.org" in rendered


def test_error_heatmap_counts_large_negative_and_positive_errors(tmp_path):
    data = dashboard.load_predictions(
        write_predictions(tmp_path, y_true=[100, 100], y_pred=[0, 200])
    )
    map_view = build_prediction_map(data, "error", "Heatmap")
    heat = next(
        child for child in map_view._children.values() if child.__class__.__name__ == "HeatMap"
    )
    assert heat.data[0][2] == heat.data[1][2] == 1.0


def test_heatmap_outlier_toggle_and_zero_errors():
    data = pd.DataFrame(
        {
            "lat": np.linspace(40.6, 40.8, 60),
            "lon": -74.0,
            "error": [0] + [1] * 58 + [100000],
        }
    )

    def weights(clip):
        view = build_prediction_map(data, "error", "Heatmap", clip=clip)
        heat = next(
            child for child in view._children.values() if child.__class__.__name__ == "HeatMap"
        )
        return [point[2] for point in heat.data]

    clipped, full = weights(True), weights(False)
    # Leaflet itself floors zero weights to a visible opacity, so omit them.
    assert len(clipped) == len(full) == 59
    assert clipped[0] == 1
    assert full[0] == pytest.approx(0.00001)


def test_dashboard_all_controls_and_tabs(tmp_path, monkeypatch):
    original_discover = dashboard.discover_predictions
    monkeypatch.setattr(dashboard, "discover_predictions", lambda: original_discover(tmp_path))
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    assert len(app.tabs) == 5
    assert len(app.dataframe) == 2  # leaderboard and comparison both populated
    assert len(app.get("download_button")) == 2
    assert "ResizeObserver" in app.get("iframe")[0].proto.srcdoc
    assert any("synthetic" in notice.value for notice in app.info)
    for model in ["extra_trees", "hist_gb", "xgboost"]:
        for mode in ["Cluster", "Points", "Heatmap"]:
            for layer in ["Prediction", "Error", "Ground truth"]:
                app.selectbox[3].select(model)
                app.radio[0].set_value(mode)
                app.radio[1].set_value(layer)
                app.toggle[0].set_value(layer != "Error")
                app.run(timeout=20)
                assert not app.exception, [exc.message for exc in app.exception]
                assert not app.error
                assert len(app.dataframe[0].value) == 3
    app.button[0].click().run(timeout=20)
    assert not app.exception
    # External training results should appear after reload, replacing demo choices.
    write_predictions(tmp_path)
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert app.selectbox[0].value == "nyc"
    assert app.selectbox[3].value == "ridge"
    assert app.dataframe[0].value["MAE"].iloc[0] == 15
    # Replacing the same file must invalidate cached data, too.
    write_predictions(tmp_path, y_pred=[100.0, 200.0])
    app.button[0].click().run(timeout=20)
    assert app.dataframe[0].value["MAE"].iloc[0] == 0
    write_predictions(tmp_path, y_true=[np.nan, np.nan])
    app.button[0].click().run(timeout=20)
    for layer in ["Prediction", "Error", "Ground truth"]:
        app.radio[1].set_value(layer).run(timeout=20)
        assert not app.exception
        assert not app.error
    assert any("Observed values" in message.value for message in app.info)
    assert not any("**R²: nan" in message.value for message in app.markdown)
