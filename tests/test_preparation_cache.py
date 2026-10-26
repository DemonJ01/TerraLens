"""Preparation must not join features measured against another year's tracts."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.prepare_city import preparation_steps


def cached_inputs(root, *, requested_year=2022, feature_year=2021, with_osm=True):
    for relative in [
        f"data/raw/census/nyc_tracts_{requested_year}.gpkg",
        f"data/raw/census/nyc_targets_{requested_year}.parquet",
        "data/features/nyc_geo_features.parquet",
        *(["data/raw/osm/nyc.osm.pbf"] if with_osm else []),
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    source = root / "data/features/nyc_geo_features.source.json"
    source.write_text(
        json.dumps({"tract_geometry": f"data/raw/census/nyc_tracts_{feature_year}.gpkg"})
    )
    return source


def test_different_tract_year_forces_feature_recomputation(tmp_path):
    cached_inputs(tmp_path)
    steps = preparation_steps("nyc", 2022, "ridge", root=tmp_path, skip_downloads=True)
    assert [step[0] for step in steps] == [
        "compute_osm_features_from_pbf.py",
        "build_dataset_multi_city.py",
        "cache_predictions.py",
    ]
    assert "--force" in steps[0]
    assert steps[0][steps[0].index("--tracts") + 1].endswith("nyc_tracts_2022.gpkg")


def test_different_tract_year_requires_osm_input_when_downloads_disabled(tmp_path):
    cached_inputs(tmp_path, with_osm=False)
    with pytest.raises(ValueError, match="nyc.osm.pbf"):
        preparation_steps("nyc", 2022, "ridge", root=tmp_path, skip_downloads=True)
    steps = preparation_steps("nyc", 2022, "ridge", root=tmp_path)
    assert steps[0] == ["download_osm_extract.py", "--city", "nyc"]
    assert "--force" in steps[1]


@pytest.mark.parametrize("source_text", [None, "bad json", "[]", "{}", '{"tract_geometry": 42}'])
def test_unverifiable_cached_features_are_recomputed(tmp_path, source_text):
    source = cached_inputs(tmp_path)
    if source_text is None:
        source.unlink()
    else:
        source.write_text(source_text)
    steps = preparation_steps("nyc", 2022, "ridge", root=tmp_path, skip_downloads=True)
    assert steps[0][0] == "compute_osm_features_from_pbf.py"
    assert "--force" in steps[0]


def test_matching_absolute_tract_path_reuses_feature_cache(tmp_path):
    source = cached_inputs(tmp_path, requested_year=2021, with_osm=False)
    source.write_text(
        json.dumps({"tract_geometry": str(tmp_path / "data/raw/census/nyc_tracts_2021.gpkg")})
    )
    steps = preparation_steps("nyc", 2021, "ridge", root=tmp_path, skip_downloads=True)
    assert [step[0] for step in steps] == ["build_dataset_multi_city.py", "cache_predictions.py"]


@pytest.mark.parametrize("skip_downloads", [False, True])
def test_full_workflow_uses_shared_city_preparation_with_original_model_defaults(
    tmp_path, skip_downloads
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    project_scripts = Path(__file__).resolve().parents[1] / "scripts"
    for name in ["run_everything.sh", "refresh_results.sh"]:
        shutil.copyfile(project_scripts / name, scripts / name)
    interpreter = tmp_path / "record-python.sh"
    interpreter.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$TERRALENS_COMMAND_LOG"\n')
    interpreter.chmod(0o755)
    log = tmp_path / "commands.log"
    subprocess.run(
        ["bash", str(scripts / "run_everything.sh"), "2022", *(
            ["--skip-downloads"] if skip_downloads else []
        )],
        env={**os.environ, "TERRALENS_PYTHON": str(interpreter), "TERRALENS_COMMAND_LOG": str(log)},
        check=True,
        capture_output=True,
        text=True,
    )
    calls = [line.split() for line in log.read_text().splitlines()]
    prepared = [call for call in calls if call[0] == "scripts/prepare_city.py"]
    assert len(prepared) == 3
    for call, city in zip(prepared, ["nyc", "chicago", "san_francisco"]):
        assert call[:7] == [
            "scripts/prepare_city.py", "--city", city, "--year", "2022",
            "--models", "xgboost,extra_trees,hist_gb,ridge",
        ]
        assert ("--skip-downloads" in call) is skip_downloads
    assert not any(call[0] == "scripts/compute_osm_features_from_pbf.py" for call in calls)
