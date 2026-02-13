"""Example of how to download data objects from the ICOS portal.

The environment variable ICOS_API_TOKEN should be initialized with an ICOS
token from https://cpauth.icos-cp.eu
"""

import logging
import os

from dotenv import load_dotenv

from trunx.config import icos_raw_data_folder
from trunx.datasets.icos import download_ICOS_object, get_ICOS_client

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Get token from .env file and initialize client.
    load_dotenv()
    try:
        token = os.environ["ICOS_API_TOKEN"]
        data_client = get_ICOS_client(token)
    except KeyError:
        raise RuntimeError("Environment variable ICOS_API_TOKEN must be defined.") from None

    # Dummy object.
    dobj_uri = "https://meta.icos-cp.eu/objects/E2MVHezJQReXShfzPtBVlVwS"
    download_ICOS_object(dobj_uri, icos_raw_data_folder, data_client)
