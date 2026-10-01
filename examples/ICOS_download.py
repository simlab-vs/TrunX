"""Example of how to download Level 2 products from the ICOS portal.

The environment variables ICOS_USERNAME and ICOS_PASSWORD should be initialized
with the credentials of an ICOS account (https://cpauth.icos-cp.eu).
"""

import logging
import os

from dotenv import load_dotenv

from trunx.config import icos_raw_data_folder
from trunx.datasets.icos import download_ICOS_L2, get_ICOS_clients, list_L2_station_ids

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Get credentials from .env file and initialize clients.
    load_dotenv()
    try:
        username = os.environ["ICOS_USERNAME"]
        password = os.environ["ICOS_PASSWORD"]

    except KeyError as e:
        raise RuntimeError(f"Environment variable {e} must be defined.") from None
    meta_client, data_client = get_ICOS_clients(username, password)

    # List of station IDS
    station_ids = list_L2_station_ids(meta_client)

    # All Level 2 products for Davos.
    download_ICOS_L2(
        "CH-Dav",
        products=None,  # specify the list of products if you want to download only specific ones
        folder=icos_raw_data_folder,
        meta=meta_client,
        data=data_client,
    )
