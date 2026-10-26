"""TIGER archive downloads remain recoverable after interrupted requests."""

from io import BytesIO
from zipfile import ZipFile, is_zipfile

import pytest
import requests

from scripts import fetch_census_tracts_multi as tracts


def archive_bytes():
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("tracts.shp", "fixture")
    return buffer.getvalue()


class ArchiveResponse:
    def __init__(self, content, fail=False):
        self.content = content
        self.fail = fail

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.content
        if self.fail:
            raise requests.ConnectionError("Download interrupted")


def test_interrupted_archive_is_not_cached_and_retry_succeeds(tmp_path, monkeypatch):
    responses = iter([ArchiveResponse(b"partial", fail=True), ArchiveResponse(archive_bytes())])
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: next(responses))
    output = tmp_path / "tl_2021_36_tract.zip"
    with pytest.raises(requests.ConnectionError):
        tracts.fetch_state_tracts(2021, "36", tmp_path)
    assert not output.exists()
    assert not output.with_suffix(".zip.part").exists()
    assert tracts.fetch_state_tracts(2021, "36", tmp_path) == output
    assert is_zipfile(output)


def test_invalid_cached_archive_is_replaced(tmp_path, monkeypatch):
    output = tmp_path / "tl_2021_36_tract.zip"
    output.write_bytes(b"incomplete old download")
    monkeypatch.setattr(
        requests, "get", lambda *args, **kwargs: ArchiveResponse(archive_bytes())
    )
    tracts.fetch_state_tracts(2021, "36", tmp_path)
    assert is_zipfile(output)


def test_valid_cached_archive_does_not_make_network_request(tmp_path, monkeypatch):
    output = tmp_path / "tl_2021_36_tract.zip"
    output.write_bytes(archive_bytes())

    def unexpected_request(*args, **kwargs):
        pytest.fail("Valid cached TIGER archives must be reused")

    monkeypatch.setattr(requests, "get", unexpected_request)
    assert tracts.fetch_state_tracts(2021, "36", tmp_path) == output


def test_invalid_http_success_does_not_replace_existing_archive(tmp_path, monkeypatch):
    output = tmp_path / "tracts.zip"
    original = archive_bytes()
    output.write_bytes(original)
    monkeypatch.setattr(
        requests, "get", lambda *args, **kwargs: ArchiveResponse(b"<html>maintenance</html>")
    )
    with pytest.raises(RuntimeError, match="invalid TIGER archive"):
        tracts.download("https://example.test/tracts.zip", output)
    assert output.read_bytes() == original
    assert not output.with_suffix(".zip.part").exists()
