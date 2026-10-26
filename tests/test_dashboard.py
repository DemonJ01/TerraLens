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


@pytest.mark.parametrize("embedded_identity", [False, True])
@pytest.mark.parametrize(
    "field,wrong_value",
    [
        ("city", "chicago"),
        ("model", "xgboost"),
        ("feature_set", "osm_311"),
        ("target", "median_rent"),
    ],
)
def test_stale_manifest_cannot_relabel_canonical_predictions(
    tmp_path, embedded_identity, field, wrong_value
):
    identity = {
        "city": "nyc",
        "model": "ridge",
        "feature_set": "osm_only",
        "target": "median_income",
    }
    predictions = write_predictions(tmp_path, **(identity if embedded_identity else {}))
    pd.DataFrame([{**identity, field: wrong_value, "path": str(predictions)}]).to_csv(
        predictions.parent / "manifest.csv", index=False
    )
    found = dashboard.discover_predictions(tmp_path)
    assert found[dashboard.MANIFEST_COLUMNS[:4]].to_dict("records") == [identity]


def test_custom_manifest_paths_use_embedded_identity_and_read_each_file_once(tmp_path, monkeypatch):
    identity = {
        "city": "nyc",
        "model": "ridge",
        "feature_set": "osm_only",
        "target": "median_income",
    }
    predictions = write_predictions(tmp_path, name="custom-results", **identity)
    pd.DataFrame(
        [
            {**identity, "city": "chicago", "path": str(predictions)},
            {**identity, "path": str(predictions)},
        ]
    ).to_csv(predictions.parent / "manifest.csv", index=False)
    original_load = dashboard.load_predictions
    reads = []

    def counting_load(path):
        reads.append(path)
        return original_load(path)

    monkeypatch.setattr(dashboard, "load_predictions", counting_load)
    found = dashboard.discover_predictions(tmp_path)
    assert found[dashboard.MANIFEST_COLUMNS[:4]].to_dict("records") == [identity]
    assert reads == [str(predictions)]


@pytest.mark.parametrize("field", ["city", "model", "feature_set", "target"])
def test_numeric_manifest_labels_are_skipped(tmp_path, field):
    predictions = write_predictions(tmp_path, name="custom-results")
    identity = {
        "city": "nyc",
        "model": "ridge",
        "feature_set": "osm_only",
        "target": "median_income",
        field: 42,
    }
    pd.DataFrame([{**identity, "path": str(predictions)}]).to_csv(
        predictions.parent / "manifest.csv", index=False
    )
    found = dashboard.discover_predictions(tmp_path)
    assert found.city.str.endswith("_sample").all()


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


@pytest.mark.parametrize("mode", ["Cluster", "Points"])
@pytest.mark.parametrize(
    "target", ["poverty_rate", "unemployment_rate", "bachelors_plus_rate"]
)
def test_rate_popups_preserve_small_errors_and_explain_the_scale(tmp_path, mode, target):
    data = dashboard.load_predictions(
        write_predictions(
            tmp_path,
            target=target,
            y_true=[0.05, np.nan],
            y_pred=[0.050988830267, 0.146],
        )
    )
    rendered = build_prediction_map(data, "y_pred", mode).get_root().render()
    assert rendered.count("<b>Units:</b> rate (0–1 scale)") == 2
    assert "<b>Data value:</b> 0.050000" in rendered
    assert "<b>Estimate:</b> 0.050989" in rendered
    assert "<b>Error:</b> 0.000989" in rendered
    assert "<b>Data value:</b> Not available" in rendered
    assert "<b>Error:</b> Not available" in rendered


@pytest.mark.parametrize("observed,scored", [([np.nan, np.nan], 0), ([100.0, np.nan], 1)])
def test_prediction_count_does_not_depend_on_observed_values(
    tmp_path, monkeypatch, observed, scored
):
    write_predictions(tmp_path, y_true=observed)
    original_discover = dashboard.discover_predictions
    monkeypatch.setattr(dashboard, "discover_predictions", lambda: original_discover(tmp_path))
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    assert not app.exception, [exc.message for exc in app.exception]
    assert not app.error
    metrics = next(item.value for item in app.markdown if 'class="metrics"' in item.value)
    assert '<div class="metric-label">Tracts</div><div class="metric-value">2</div>' in metrics
    assert any(
        caption.value
        == f"2 tracts have predictions; {scored} have observed values for accuracy scores."
        for caption in app.caption
    )
    map_html = app.get("iframe")[0].proto.srcdoc
    assert map_html.count("<b>Area:</b>") == 2
    if scored:
        assert app.dataframe[0].value["N"].tolist() == [scored]
        assert app.metric[0].value == "10"
    else:
        assert not app.dataframe
        assert any("Observed values are needed" in message.value for message in app.info)


def test_ordinary_reruns_preserve_map_and_explicit_reload_rebuilds_it(tmp_path, monkeypatch):
    write_predictions(tmp_path)
    original_discover = dashboard.discover_predictions
    monkeypatch.setattr(dashboard, "discover_predictions", lambda: original_discover(tmp_path))
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    def map_document():
        assert not app.exception, [exc.message for exc in app.exception]
        return app.get("iframe")[0].proto.srcdoc

    original = map_document()
    app.run(timeout=20)
    assert map_document() == original
    app.button[0].click().run(timeout=20)
    assert map_document() != original

    app.radio[0].set_value("Points").run(timeout=20)
    points = map_document()
    assert points != original
    app.run(timeout=20)
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
    app.run(timeout=20)
    assert map_document() == replacement


def test_reload_rereads_files_and_renderer_changes_invalidate_map_cache(tmp_path, monkeypatch):
    from branca.element import Element

    from terralens.utils import maps

    predictions = write_predictions(tmp_path)
    original_discover = dashboard.discover_predictions
    monkeypatch.setattr(dashboard, "discover_predictions", lambda: original_discover(tmp_path))
    renderer_source = tmp_path / "maps.py"
    renderer_source.write_text("renderer version one")
    monkeypatch.setattr(maps, "__file__", str(renderer_source))
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    assert not app.exception
    original_html = app.get("iframe")[0].proto.srcdoc
    original_renderer = maps.build_prediction_map

    def changed_renderer(*args, **kwargs):
        view = original_renderer(*args, **kwargs)
        view.get_root().html.add_child(Element("<!-- refreshed-renderer -->"))
        return view

    monkeypatch.setattr(maps, "build_prediction_map", changed_renderer)
    app.run(timeout=20)
    assert app.get("iframe")[0].proto.srcdoc == original_html

    app.button[0].click().run(timeout=20)
    refreshed = app.get("iframe")[0].proto.srcdoc
    assert "<!-- refreshed-renderer -->" in refreshed
    app.run(timeout=20)
    assert app.get("iframe")[0].proto.srcdoc == refreshed

    # A renderer source edit must invalidate HTML even without clicking Reload.
    renderer_source.write_text("renderer version two")
    app.run(timeout=20)
    assert app.get("iframe")[0].proto.srcdoc != refreshed

    # Explicit reload also rereads content if a file's cache metadata is unchanged.
    original_stat = predictions.stat()
    original_loader = dashboard.load_predictions
    reads = []

    def changed_loader(path):
        reads.append(path)
        data = original_loader(path)
        data["y_pred"] = data["y_true"]
        data["error"] = 0.0
        return data

    monkeypatch.setattr(dashboard, "load_predictions", changed_loader)
    app.run(timeout=20)
    assert app.dataframe[0].value["MAE"].iloc[0] == 15
    discovery_reads = len(reads)
    app.button[0].click().run(timeout=20)
    assert len(reads) > discovery_reads
    assert not app.exception
    assert app.dataframe[0].value["MAE"].iloc[0] == 0
    current_stat = predictions.stat()
    assert (current_stat.st_mtime_ns, current_stat.st_size) == (
        original_stat.st_mtime_ns,
        original_stat.st_size,
    )


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
    assert len(app.dataframe) == 3  # scores, comparison, and paired predictions
    assert len(app.get("download_button")) == 3
    assert "ResizeObserver" in app.get("iframe")[0].proto.srcdoc
    assert any("synthetic" in notice.value for notice in app.info)
    for model in ["extra_trees", "hist_gb", "xgboost"]:
        for mode in ["Cluster", "Points", "Heatmap"]:
            for layer in ["Prediction", "Error", "Ground truth"]:
                app.sidebar.selectbox[3].select(model)
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
    assert app.sidebar.selectbox[0].value == "nyc"
    assert app.sidebar.selectbox[3].value == "ridge"
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
