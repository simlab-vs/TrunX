"""ERA5 data download and processing script."""

import os

import boto3
import pandas as pd
from dotenv import load_dotenv

from trunx.config import era5_data_folder

load_dotenv()

s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

bucket_name = "jaxifer"


def download_era5_data(s3_metric_key: str):
    """Download ERA5 data from S3, concatenate yearly CSVs, and save as Parquet."""
    precip = []
    for year in range(1992, 2026):
        s3_key = f"{s3_metric_key}_{year}.csv"
        local_file_path = os.path.join(era5_data_folder, s3_key[10:])
        try:
            s3_client.download_file(bucket_name, s3_key, local_file_path)
            print(f"Successfully downloaded {s3_key} to {local_file_path}")
        except Exception as e:
            print(f"Error downloading file: {e}")
        try:
            df = pd.read_csv(local_file_path)
            precip.append(df)
        except Exception as e:
            print(f"Error reading CSV file {local_file_path}: {e}")

    precip = pd.concat(precip, ignore_index=True)
    precip.to_parquet(os.path.join(era5_data_folder, f"{s3_metric_key[10:]}.parquet"), index=False)
    print(
        f"Successfully saved the concatenated DataFrame \
            to 'data/era5/{s3_metric_key[10:]}.parquet'"
    )


if __name__ == "__main__":
    # Precipitation
    s3_metric_key = "data/era5/era5_total_precipitation"
    download_era5_data(s3_metric_key)

    # Temperature
    s3_metric_key = "data/era5/era5_2m_temperature"
    download_era5_data(s3_metric_key)

    # Solar radiation
    s3_metric_key = "data/era5/era5_surface_solar_radiation_downwards"
    download_era5_data(s3_metric_key)
