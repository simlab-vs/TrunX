"""ICOS dataset related utilities."""

import logging
from pathlib import Path

from icoscp_core.dataclient import DataClient
from icoscp_core.icos import bootstrap

logger = logging.getLogger(__name__)


def get_ICOS_client(token: str) -> DataClient:
    """Initialize a client for the ICOS data portal.

    Parameters
    ----------
    token: str
        Authentication token for the ICOS portal.
        Can be fetched from https://cpauth.icos-cp.eu. Token are only valid for 27 hours.
    """
    try:
        _, data_client = bootstrap.fromCookieToken(token)
        return data_client

    # icoscp_core raises bare exceptions, we thus have to catch-all and re-raise
    except Exception as e:
        logging.error(
            f"Potential error in ICOS client initialization.\n Used credential: token: {token}"
        )
        logging.error(f"Original exception: {e}")
        raise


def download_ICOS_object(dobj_uri: str, folder: Path, data: DataClient) -> None:
    """Download a given ICOS object.

    Parameters
    ----------
    dobj_uri: str
        URI to the object to download.
    folder: Path
        Path to the folder where the file will be saved.
    data_client: DataClient
        Initialized data client for the ICOS portal. Use `get_ICOS_client`.
    """
    folder.mkdir(parents=True, exist_ok=True)
    filename = data.save_to_folder(dobj_uri, str(folder))
    logger.info(f"Successfuly downloaded {filename}")
