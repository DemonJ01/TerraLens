"""Export every cached map variant without replacing saved research map data."""

import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.make_interactive_maps import LAYERS, MODELS, TARGETS, export_map

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MARKERS = {
    "nyc_xgboost_osm_only_income_map.html": (
        "3da29a4bf541865d9b6ae2a7d532ff7af456922e78f7fe6161e2932627964930"
    ),
    "nyc_xgboost_osm_311_income_map.html": (
        "a87b02432509d5909b62b17c6159fa152b2fa65d48a1221cfc3a3b294b0ebf31"
    ),
}


def write_cache(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "tract_id": ["fixture-a", "fixture-b", "bad-row"],
            "lat": [40.70, 40.72, np.inf],
            "lon": [-74.00, -73.98, -74.0],
            "y_true": [0.2, 0.8, 1],
            "y_pred": [0.3, 0.7, 2],
            "error": [999, 999, 999],
        }
    ).to_parquet(path, index=False)


@pytest.mark.parametrize("mode", ["Cluster", "Points", "Heatmap"])
@pytest.mark.parametrize("layer", LAYERS)
def test_exports_every_map_mode_and_layer(tmp_path, mode, layer):
    cached = tmp_path / "predictions.parquet"
    write_cache(cached)
    output = export_map(cached, tmp_path / "maps/view.html", "TerraLens <test>", layer, mode)
    html = output.read_text()
    assert "<title>TerraLens &lt;test&gt;</title>" in html
    assert "L.map(" in html
    assert "L.divIcon(" not in html
    assert "NaN" not in html and "Infinity" not in html
    assert "tile.openstreetmap.org" in html
    if mode == "Heatmap":
        assert "L.heatLayer(" in html
    else:
        assert html.count("L.circleMarker(") == 2
        assert "fixture-a" in html and "fixture-b" in html
        assert "bad-row" not in html


@pytest.mark.parametrize("model", MODELS)
def test_cli_accepts_every_cached_model_and_isolated_paths(tmp_path, model):
    target = TARGETS[MODELS.index(model) % len(TARGETS)]
    cached = tmp_path / "cache" / f"chicago__{model}__osm_only__{target}.parquet"
    write_cache(cached)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/make_interactive_maps.py"),
            "--city",
            "chicago",
            "--model",
            model,
            "--target",
            target,
            "--cache-dir",
            str(cached.parent),
            "--output-dir",
            str(tmp_path / "maps"),
            "--layer",
            "error",
            "--mode",
            "Heatmap",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Wrote interactive TerraLens map:" in result.stdout
    maps = list((tmp_path / "maps").glob("*.html"))
    assert len(maps) == 1 and "L.heatLayer(" in maps[0].read_text()


@pytest.mark.parametrize("name", ORIGINAL_MARKERS)
def test_repaired_saved_maps_preserve_all_original_marker_data(name, tmp_path):
    html = (ROOT / "docs/maps" / name).read_text()
    assert html.count("L.circleMarker(") == 2196
    marker_block = re.search(
        r"            var circle_marker_.*?(?=            marker_cluster_.*?\.addTo\(map_)",
        html,
        re.S,
    ).group(0)
    assert hashlib.sha256(marker_block.encode()).hexdigest() == ORIGINAL_MARKERS[name]
    assert "tile.openstreetmap.org/{z}/{x}/{y}.png" in html
    assert "CARTO" not in html and "cartocdn" not in html
    assert "L.divIcon(" not in html and ".setIcon(div_icon_" not in html
    assert "<title>TerraLens" in html
    # Validate the entire generated JavaScript without requiring browser globals.
    if node := shutil.which("node"):
        script = tmp_path / "map.js"
        script.write_text("\n".join(re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", html, re.S)))
        subprocess.run([node, "--check", str(script)], check=True, capture_output=True, text=True)
