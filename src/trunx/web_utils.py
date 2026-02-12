"""Web-related utilities.

Functionalities include:

- downloading file (streaming mode) from URL

"""

import logging

import requests

logger = logging.getLogger(__name__)


def download_from_url(url: str, output_file: str):
    """
    Download data from the given URL and save it to the specified output file.

    Parameters
    ----------
    url : str
        The URL to download the file from.
    output_file : str
        The local path where the downloaded file will be saved.

    Returns
    -------
    None
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Successfuly downloaded {output_file}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error while connecting to {url}: {e}")
