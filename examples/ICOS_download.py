"""Example of how to download Level 2 products from the ICOS portal.

The environment variable ICOS_API_TOKEN should be initialized with an ICOS
token from https://cpauth.icos-cp.eu
"""

import logging
import os

from dotenv import load_dotenv

from trunx.config import icos_raw_data_folder
from trunx.datasets.icos import download_ICOS_L2, get_ICOS_clients, list_L2_station_ids

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Get token from .env file and initialize clients.
    load_dotenv()
    try:
        token = os.environ["ICOS_API_TOKEN"]
    except KeyError:
        raise RuntimeError("Environment variable ICOS_API_TOKEN must be defined.") from None
    meta_client, data_client = get_ICOS_clients(token)

    # List of station IDS
    station_ids = list_L2_station_ids(meta_client)
    print(f"Station IDS: {station_ids}")

    # All Level 2 products for Davos.
    download_ICOS_L2(
        "CH-Dav",
        products=None,  # specify the list of products if you want to download only specific ones
        folder=icos_raw_data_folder,
        meta=meta_client,
        data=data_client,
    )
