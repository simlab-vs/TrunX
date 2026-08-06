"""Download GOSIF GPP for ICP forest locations."""

import datetime as dt
import gzip
import os
from io import BytesIO

import numpy as np
import polars as pl
import rasterio
import requests

from trunx.config import clean_data_folder, raw_data_folder
from trunx.gp3.create_data_inputs import dms_to_decimal

base_url = "https://data.globalecology.unh.edu/data/GOSIF-GPP_v2/Monthly/Mean/"


def get_icp_data():
    """Load ICP data, filter for relevant species, and convert coordinates."""
    icp_df = pl.read_parquet(os.path.join(clean_data_folder, "icp_tree_data.parquet"))
    icp_df = icp_df.filter(
        pl.col("specie").is_in(
            list(
                [
                    "Picea abies",
                    "Pinus sylvestris",
                    "Fagus sylvatica",
                    "Quercus robur",
                    "Quercus petraea",
                ]
            )
        )
    )

    icp_df = icp_df.with_columns(
        pl.col("plot_latitude").map_elements(dms_to_decimal, return_dtype=pl.Float64).alias("Lat"),
        pl.col("plot_longitude")
        .map_elements(dms_to_decimal, return_dtype=pl.Float64)
        .alias("Lon"),
    )

    return icp_df


def get_GOSIF_GPP_for_icp(icp_df):
    """Download GOSIF GPP for ICP forest locations and save to CSV."""
    SCALE_FACTOR = 0.01  # scale factor from GOSIF documentation
    # GOSIF documentation states these are fill values for missing data
    FILL_VALUES = [32766, 32767]
    icp_loc = icp_df.select(["Lat", "Lon", "plot_id"])

    all_frames = []
    # convert ICP points once (important for speed)
    coords = list(zip(icp_loc["Lon"].to_list(), icp_loc["Lat"].to_list(), strict=True))
    plot_ids = icp_loc["plot_id"].to_list()
    for year in range(2000, dt.datetime.now().year):
        for month in range(1, 13):
            file_name = f"GOSIF_GPP_{year}.M{month:02d}_Mean.tif.gz"
            url = base_url + file_name

            print(f"Processing {year}-{month:02d}")

            r = requests.get(url)

            if r.status_code != 200 or r.content[:2] != b"\x1f\x8b":
                print(f"Skipping {file_name}")
                continue

            file_path = os.path.join(raw_data_folder, "GOSIF", file_name)

            with open(file_path, "wb") as f:
                f.write(r.content)

            # Read from the saved file instead of memory
            with open(file_path, "rb") as f:
                content = f.read()

            # Decompress the saved file
            with gzip.GzipFile(fileobj=BytesIO(content)) as gz:
                tif_bytes = gz.read()

            # open raster in memory
            with rasterio.MemoryFile(tif_bytes) as memfile, memfile.open() as src:
                # sample ICP points
                values = [v[0] for v in src.sample(coords)]

            values = np.array(values, dtype=np.float32)

            # Remove fill values
            values[np.isin(values, FILL_VALUES)] = np.nan

            df = pl.DataFrame({"year": year, "month": month, "plot_id": plot_ids, "GPP": values})

            all_frames.append(df)

    final_df = pl.concat(all_frames).unique()
    final_df = final_df.with_columns(
        pl.col("GPP") * SCALE_FACTOR  # GPP in gC/m2/month
    )
    final_df = final_df.with_columns(
        pl.col("GPP") / 12.011  # GPP in gC/m2/month -> mol C/m2/month
    )

    final_df.write_csv(os.path.join(clean_data_folder, "GOSIF_GPP_icp.csv"))


if __name__ == "__main__":
    icp_df = get_icp_data()

    # This will take a while, so we can run it once and save the output
    # Download GOSIF GPP for ICP points and save to CSV
    get_GOSIF_GPP_for_icp(icp_df)
