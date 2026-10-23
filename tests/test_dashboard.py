import re
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


@pytest.mark.parametrize("mode", ["Cluster", "Points", "Heatmap"])
def test_maps_allow_mouse_wheel_zoom(tmp_path, mode):
    data = dashboard.load_predictions(write_predictions(tmp_path))
    view = build_prediction_map(data, "y_pred", mode)
    assert view.options["scroll_wheel_zoom"] is True
    # Leaflet's default resize handler can run while the iframe is hidden.
    assert view.options["track_resize"] is False
    assert re.search(r'"trackResize"\s*:\s*false', view.get_root().render())


@pytest.mark.parametrize("layer", ["y_pred", "y_true", "error"])
def test_heatmap_attaches_only_after_the_container_has_visible_dimensions(tmp_path, layer):
    data = dashboard.load_predictions(write_predictions(tmp_path))
    view = build_prediction_map(data, layer, "Heatmap")
    heat = next(child for child in view._children.values() if child.__class__.__name__ == "HeatMap")
    sizing = next(
        child for child in view._children.values() if child.__class__.__name__ == "_MapSizing"
    )
    assert heat.show is False
    assert sizing.heat_layer is heat
    rendered = view.get_root().render()
    assert f"const heatLayer = {heat.get_name()};" in rendered
    # Eager .addTo invokes Canvas getImageData on a zero-width hidden iframe.
    assert not re.search(rf"{heat.get_name()}\s*\.addTo\s*\(\s*{view.get_name()}\s*\)", rendered)
    resize_handler = rendered.split("function resizeMap()", 1)[1].split("const ResetView", 1)[0]
    assert resize_handler.index("size.width > 0 && size.height > 0") < resize_handler.index(
        "heatLayer.addTo(map)"
    )
    assert "else if (heatLayer !== null && map.hasLayer(heatLayer))" in resize_handler
    assert "map.removeLayer(heatLayer)" in resize_handler


@pytest.mark.parametrize("mode", ["Cluster", "Points"])
def test_map_popups_explain_missing_observed_values(tmp_path, mode):
    data = dashboard.load_predictions(write_predictions(tmp_path, y_true=[np.nan, np.nan]))
    rendered = build_prediction_map(data, "y_pred", mode).get_root().render()
    assert rendered.count("<b>Data value:</b> Not available") == 2
    assert rendered.count("<b>Error:</b> Not available") == 2
    assert "<b>Estimate:</b> 110.00" in rendered
    assert "<b>Estimate:</b> 180.00" in rendered
    assert ": nan" not in rendered.lower()


def test_reload_preserves_map_document_until_data_or_settings_change(tmp_path, monkeypatch):
    write_predictions(tmp_path)
    original_discover = dashboard.discover_predictions
    monkeypatch.setattr(dashboard, "discover_predictions", lambda: original_discover(tmp_path))
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    def map_document():
        assert not app.exception, [exc.message for exc in app.exception]
        return app.get("iframe")[0].proto.srcdoc

    original = map_document()
    app.button[0].click().run(timeout=20)
    assert map_document() == original

    app.radio[0].set_value("Points").run(timeout=20)
    points = map_document()
    assert points != original
    app.button[0].click().run(timeout=20)
    assert map_document() == points

    app.radio[1].set_value("Ground truth").run(timeout=20)
    truth = map_document()
    assert truth != points
    app.toggle[0].set_value(False).run(timeout=20)
    unclipped = map_document()
    assert unclipped != truth

    # Return to predictions so a changed estimate is visible in the map itself.
    app.radio[1].set_value("Prediction").run(timeout=20)
    before_replacement = map_document()
    write_predictions(tmp_path, y_pred=[120.0, 190.0])
    app.button[0].click().run(timeout=20)
    replacement = map_document()
    assert replacement != before_replacement
    app.button[0].click().run(timeout=20)
    assert map_document() == replacement


@pytest.mark.parametrize("observed", [[100.0], [100.0, 100.0]])
def test_compare_explains_undefined_r_squared_without_empty_chart(tmp_path, monkeypatch, observed):
    count = len(observed)
    write_predictions(
        tmp_path,
        lat=[40.7 + i * 0.01 for i in range(count)],
        lon=[-74.0] * count,
        y_true=observed,
        y_pred=[110.0] * count,
    )
    original_discover = dashboard.discover_predictions
    monkeypatch.setattr(dashboard, "discover_predictions", lambda: original_discover(tmp_path))
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    assert not app.exception, [exc.message for exc in app.exception]
    compare = app.tabs[3]
    assert not compare.get("arrow_vega_lite_chart")
    notices = [notice.value.lower() for notice in compare.info]
    notices += [caption.value.lower() for caption in compare.caption]
    assert any("different observed values" in notice for notice in notices)


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
