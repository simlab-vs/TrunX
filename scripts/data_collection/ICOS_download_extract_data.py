import json
import os
import zipfile
import logging
from icoscp_core.icos import bootstrap

# Constants
TOKEN_FILE_PATH = "./tokens/cpauthToken_auth_conf.json"
DATA_FOLDER_PATH = "./data/raw/ICOS/"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_icoscp_credentials(token_file_path: str) -> dict:
    """
    Load the authentication credentials from a JSON file.

    Parameters
    ----------
    token_file_path (str): Path to the JSON file containing the credentials.

    Returns
    -------
    dict: A dictionary containing the credentials (username, password).
    """
    try:
        with open(token_file_path, "r") as f:
            credentials = json.load(f)
        return credentials
    except FileNotFoundError:
        logging.error(f"Token file not found: {token_file_path}")
        raise
    except json.JSONDecodeError:
        logging.error("Error decoding the JSON token file.")
        raise
    except Exception as e:
        logging.error(f"An error occurred while reading the token file: {e}")
        raise


def download_and_extract_data(dobj_uri: str, folder_path: str, data) -> None:
    """
    Download the file using the provided object URI and extract its contents.

    Parameters
    ----------
    dobj_uri (str): The URI to the object to download.
    folder_path (str): The path to the folder where the file will be saved.
    data: The data object used to interact with the icoscp service.

    Returns
    -------
        None
    """
    try:
        # Save the ZIP file to the folder
        filename = data.save_to_folder(dobj_uri, folder_path)
        zip_file_path = os.path.join(folder_path, filename)

        logging.info(f"Downloaded ZIP file: {filename}")
        logging.info(f"ZIP file path: {zip_file_path}")

        # Extract the ZIP file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(folder_path)
            logging.info(f"Extracted files to {folder_path}")

            # List extracted files excluding .zip and .DS_Store
            extracted_files = [
                f for f in os.listdir(folder_path) if not f.endswith(".zip") and f != ".DS_Store"
            ]
            logging.info("Extracted files:")
            for ext_file in extracted_files:
                logging.info(f"- {ext_file}")

    except Exception as e:
        logging.error(f"Error downloading or extracting file: {e}")
        raise


if __name__ == "__main__":
    #  Get credentials from the token file
    credentials = get_icoscp_credentials(TOKEN_FILE_PATH)

    # Bootstrap using the credentials
    try:
        meta, data = bootstrap.fromCredentials(credentials["username"], credentials["password"])
    except Exception as e:
        logging.error(f"Error bootstrapping the data object: {e}")
        exit(1)

    # Download and extract the data
    dobj_uri = "https://meta.icos-cp.eu/objects/E2MVHezJQReXShfzPtBVlVwS"

    download_and_extract_data(dobj_uri, DATA_FOLDER_PATH, data)
