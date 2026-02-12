"""Pytest fixtures for web utilities testing."""

import pytest
import responses


@pytest.fixture(scope="function")
def mock_responses():
    """Activate responses mock for HTTP requests."""
    with responses.RequestsMock() as rsps:
        yield rsps


@pytest.fixture
def sample_content() -> bytes:
    """Return sample binary content for download tests."""
    return b"test file content"


@pytest.fixture
def mock_download_url(
    mock_responses: responses.RequestsMock, sample_content: bytes
) -> tuple[str, bytes]:
    """Set up a mock URL ready for download.

    Parameters
    ----------
    mock_responses : responses.RequestsMock
        The mocked responses context.
    sample_content : bytes
        The content to serve from the mock URL.

    Returns
    -------
    tuple[str, bytes]
        The mock URL and expected content.
    """
    url = "https://example.com/data/file.txt"
    mock_responses.add(responses.GET, url, body=sample_content, status=200)
    return url, sample_content
