"""Download and unzip the PROFOUND database.

Mirrors what the `ProfoundData` R package's `downloadDatabase()` does (see
`data/ProfoundData/R/registerDatabaseFunctions.R`), without needing R or its
dependencies installed: fetch the zip from PIK Potsdam and extract it to get
`ProfoundData.sqlite`.
"""

import os
import zipfile

import requests

from trunx.config import raw_data_folder

PROFOUND_URL = "http://www.pik-potsdam.de/data/doi/10.5880/PIK.2019.008/ProfoundData.zip"


def download_profound_database(dest_dir: str) -> str:
    """Download and unzip the PROFOUND database into `dest_dir`.

    Skips the download if the zip is already present.

    Parameters
    ----------
    dest_dir : str
        Directory to download the zip into and unzip it in.

    Returns
    -------
    str
        Path to the extracted `ProfoundData.sqlite` file.
    """
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "ProfoundData.zip")

    if os.path.exists(zip_path):
        print(f"Already downloaded: {zip_path}")
    else:
        print(f"Downloading {PROFOUND_URL} -> {zip_path}")
        response = requests.get(PROFOUND_URL, timeout=300)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(response.content)

    print(f"Unzipping {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    sqlite_path = os.path.join(dest_dir, "ProfoundData.sqlite")
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"Expected {sqlite_path} after unzipping, but it's missing")
    return sqlite_path


if __name__ == "__main__":
    dest_dir = os.path.join(raw_data_folder, "PROFOUND")
    sqlite_path = download_profound_database(dest_dir)
    print(f"PROFOUND database ready at: {sqlite_path}")
