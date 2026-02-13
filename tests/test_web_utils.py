"""Tests for web_utils module."""

from pathlib import Path

import requests
import responses

from trunx.web_utils import download_from_url


def test_download_from_url_success(mock_download_url: tuple[str, bytes], tmp_path: Path) -> None:
    url, expected_content = mock_download_url
    output_file = tmp_path / "downloaded.txt"

    download_from_url(url, str(output_file))

    assert output_file.exists()
    assert output_file.read_bytes() == expected_content


def test_download_from_url_http_error(
    mock_responses: responses.RequestsMock, tmp_path: Path, caplog
) -> None:
    url = "https://example.com/not-found.txt"
    mock_responses.add(responses.GET, url, status=404)
    output_file = tmp_path / "should_not_exist.txt"

    download_from_url(url, str(output_file))

    assert not output_file.exists()
    assert "Error while connecting" in caplog.text


def test_download_from_url_connection_error(
    mock_responses: responses.RequestsMock, tmp_path: Path, caplog
) -> None:
    url = "https://example.com/unreachable.txt"
    mock_responses.add(
        responses.GET, url, body=requests.exceptions.ConnectionError("Connection refused")
    )
    output_file = tmp_path / "should_not_exist.txt"

    download_from_url(url, str(output_file))

    assert not output_file.exists()
    assert "Error while connecting" in caplog.text
