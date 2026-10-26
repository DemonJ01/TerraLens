import subprocess
import sys
from pathlib import Path

import pytest
import requests

from scripts.download_acs_targets import ACS_VARS, fetch_county_tracts


class CensusResponse:
    def __init__(self, payload=None, status_code=200, json_error=False):
        self.payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self.json_error = json_error

    def json(self):
        if self.json_error:
            raise ValueError("<html>secret-test-key</html>")
        return self.payload


def test_missing_key_fails_before_request(monkeypatch):
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)

    def unexpected_request(*args, **kwargs):
        pytest.fail("Census request must not run without an API key")

    monkeypatch.setattr(requests, "get", unexpected_request)
    with pytest.raises(RuntimeError, match="Set CENSUS_API_KEY"):
        fetch_county_tracts(2021, "36", "061")


def test_environment_key_and_geography_are_sent_correctly(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", "secret-test-key")
    headers = ["NAME", *ACS_VARS.values(), "state", "county", "tract"]
    row = ["Tract", *(["10"] * len(ACS_VARS)), "6", "75", "10100"]

    def fake_get(url, *, params, timeout):
        assert url == "https://api.census.gov/data/2021/acs/acs5"
        assert params["key"] == "secret-test-key"
        assert params["in"] == "state:06 county:075"
        assert not {"state", "county", "tract"}.intersection(params["get"].split(","))
        assert timeout == 60
        return CensusResponse([headers, row])

    monkeypatch.setattr(requests, "get", fake_get)
    data = fetch_county_tracts(2021, "6", "75")
    assert data.tract_id.tolist() == ["06075010100"]
    assert data[ACS_VARS["median_income"]].iloc[0] == 10


@pytest.mark.parametrize("status_code", [400, 401, 403, 500])
def test_http_errors_do_not_echo_key_or_server_body(monkeypatch, status_code):
    monkeypatch.setenv("CENSUS_API_KEY", "secret-test-key")
    monkeypatch.setattr(
        requests, "get", lambda *args, **kwargs: CensusResponse(status_code=status_code)
    )
    with pytest.raises(RuntimeError) as error:
        fetch_county_tracts(2021, "36", "061")
    assert "secret-test-key" not in str(error.value)
    assert "Census" in str(error.value)


def test_non_json_auth_page_is_actionable_and_redacted(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", "secret-test-key")
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: CensusResponse(json_error=True))
    with pytest.raises(RuntimeError, match="CENSUS_API_KEY") as error:
        fetch_county_tracts(2021, "36", "061")
    assert "secret-test-key" not in str(error.value)
    assert error.value.__suppress_context__


def test_request_exceptions_do_not_echo_credentials(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", "secret-test-key")

    def failed_request(*args, **kwargs):
        raise requests.ConnectionError("https://api.census.gov/?key=secret-test-key")

    monkeypatch.setattr(requests, "get", failed_request)
    with pytest.raises(RuntimeError, match="Could not reach") as error:
        fetch_county_tracts(2021, "36", "061")
    assert "secret-test-key" not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize("payload", [{"error": "secret-test-key"}, [["NAME"]], [["NAME", "NAME"]]])
def test_unexpected_data_is_rejected_without_echoing_it(monkeypatch, payload):
    monkeypatch.setenv("CENSUS_API_KEY", "secret-test-key")
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: CensusResponse(payload))
    with pytest.raises(RuntimeError, match="invalid data table") as error:
        fetch_county_tracts(2021, "36", "061")
    assert "secret-test-key" not in str(error.value)


def test_cli_missing_key_has_no_traceback_or_partial_output(monkeypatch, tmp_path):
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)
    script = Path(__file__).resolve().parents[1] / "scripts/download_acs_targets.py"
    result = subprocess.run(
        [sys.executable, str(script), "--city", "nyc", "--source", "census"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "CENSUS_API_KEY" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "data").exists()


def test_bulk_reader_matches_api_variables_and_filters_city(monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd
    import pyogrio

    from scripts import download_acs_targets as acs

    layers = ["X19_INCOME", "X25_HOUSING", "X17_POVERTY", "X23_EMPLOYMENT", "X15_EDUCATION"]
    mapping = {}
    for variable in ACS_VARS.values():
        table, suffix = variable.split("_")
        mapping[f"{table}e{int(suffix[:-1])}"] = variable
    monkeypatch.setattr(acs, "download_archive", lambda url, path: path)
    monkeypatch.setattr(pyogrio, "list_layers", lambda path: np.array([[n, None] for n in layers]))
    monkeypatch.setattr(pyogrio, "read_info", lambda *args, **kw: {"fields": ["GEOID", *mapping]})

    def fake_read(path, *, layer, columns, read_geometry):
        assert not read_geometry
        return pd.DataFrame(
            {
                "GEOID": ["1400000US36061000100", "1400000US36001000100"],
                **{col: [100, 999] for col in columns if col != "GEOID"},
            }
        )

    monkeypatch.setattr(pyogrio, "read_dataframe", fake_read)
    raw, urls = acs.fetch_bulk_tracts("nyc", 2021, tmp_path)
    assert raw.tract_id.tolist() == ["36061000100"]
    assert set(ACS_VARS.values()).issubset(raw)
    assert len(urls) == 1 and "2021ACS" in urls[0]
    assert raw[ACS_VARS["median_income"]].iloc[0] == 100


def test_clean_targets_does_not_convert_negative_sentinels_into_valid_rates():
    import numpy as np
    import pandas as pd

    from scripts.download_acs_targets import clean_targets

    row = {variable: [100] for variable in ACS_VARS.values()}
    row.update({"tract_id": ["36061000100"]})
    row[ACS_VARS["poverty_num"]] = [-666666666]
    row[ACS_VARS["poverty_den"]] = [-666666666]
    cleaned = clean_targets(pd.DataFrame(row))
    assert np.isnan(cleaned.poverty_rate.iloc[0])
