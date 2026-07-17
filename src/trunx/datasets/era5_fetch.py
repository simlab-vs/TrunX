"""Script to query ERA5 data from the CDS API, extract values at specific locations, and save."""

import os
import tempfile
import time

import boto3
import cdsapi
import polars as pl
import xarray as xr
from dotenv import load_dotenv

load_dotenv()

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
)

client = cdsapi.Client()


def aggregate_to_daily(df: pl.DataFrame, var_col: str) -> pl.DataFrame:
    """Reduce hourly rows to one row per (date, location), keeping only available data."""
    group_cols = ["date", "latitude", "longitude"]

    daily = df.with_columns(pl.col("valid_time").dt.date().alias("date"))

    if var_col == "t2m":
        daily = daily.group_by(group_cols).agg(
            pl.col(var_col).min().alias(f"{var_col}_min"),
            pl.col(var_col).max().alias(f"{var_col}_max"),
            pl.col(var_col).mean().alias(f"{var_col}_mean"),
        )
        value_cols = [f"{var_col}_min", f"{var_col}_max", f"{var_col}_mean"]
    elif var_col == "tp":
        daily = daily.group_by(group_cols).agg(pl.col(var_col).sum())
        value_cols = [var_col]
    else:
        daily = daily.group_by(group_cols).agg(pl.col(var_col).mean())
        value_cols = [var_col]

    return daily.filter(pl.any_horizontal(pl.col(c).is_not_null() for c in value_cols))


def fetch_era5_data(year) -> None:
    """Fetch ERA5-Land data, one year at a time, across all variables."""
    os.makedirs(output_dir, exist_ok=True)

    for var in variables:
        hours = [f"{h:02d}:00" for h in range(24)] if var == "2m_temperature" else ["00:00"]
        monthly_dfs = []
        # s3_key = f"data/era5/era5_{var}_{year}.csv"
        months = [f"{m:02d}" for m in range(1, 13)]
        days = [f"{d:02d}" for d in range(1, 32)]
        for month in months:
            print(f"Querying CDS API for {var} in {year}-{month}")

            request = {
                "variable": [var],
                "year": [year],
                "month": [month],
                "day": days,
                "time": hours,
                "data_format": "grib",
                "download_format": "unarchived",
                "area": [69.59, -6.2, 38.5, 30.8],
            }
            # Use a temporary file to store the downloaded GRIB data

            with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as tmp_grib:
                grib_path = tmp_grib.name
            try:
                client.retrieve("reanalysis-era5-land", request).download(grib_path)
                # Extract data at the target locations using xarray, convert to Polars
                with xr.open_dataset(grib_path, engine="cfgrib") as ds:
                    ds_points = ds.sel(
                        latitude=target_lats, longitude=target_lons, method="nearest"
                    )
                    # Convert to Polars DataFrame via Pandas to handle multi-index
                    # and datetime properly
                    df_pandas = ds_points.to_dataframe().reset_index()
                    df_polars = pl.from_pandas(df_pandas)
                    # If 'step' column exists, convert it to total hours for easier analysis
                    if "step" in df_polars.columns:
                        df_polars = df_polars.with_columns(pl.col("step").dt.total_hours())

                    metadata_cols = {
                        "time",
                        "step",
                        "point",
                        "number",
                        "latitude",
                        "longitude",
                        "valid_time",
                        "surface",
                    }
                    data_cols = [col for col in df_polars.columns if col not in metadata_cols]

                    if data_cols:
                        target_var_col = data_cols[0]
                        daily_df = aggregate_to_daily(df_polars, target_var_col)
                        monthly_dfs.append(daily_df)

                        print(
                            f"  Collected {daily_df.shape[0]} daily rows for {var} {year}-{month}"
                        )
                    else:
                        print(f"  Warning: no data column found for {var} {year}-{month}")

            except Exception as e:
                print(f"Error processing {var} for {year}-{month}: {e}")

            finally:
                if os.path.exists(grib_path):
                    os.remove(grib_path)

            time.sleep(2)

        # Concatenate all 12 months and write one CSV per variable/year
        if monthly_dfs:
            year_df = pl.concat(monthly_dfs)
            output_path = os.path.join(output_dir, f"era5_{var}_{year}.csv")
            year_df.write_csv(output_path)
            # with open(output_path, "rb") as f:
            #     s3_client.upload_fileobj(f, "jaxifer", s3_key)
            # print(f"Successfully uploaded multi-point data to S3: {s3_key}")
            # print(f"Saved {output_path} — shape: {year_df.shape}")
        else:
            print(f"No data collected for {var} {year}.")


if __name__ == "__main__":
    icp_locations = pl.read_csv("data/clean/full_icp_plot_locations.csv")
    # Regular 0.1-degree ERA5-Land grid points clipped to Switzerland's actual
    # border (not just its bounding box) — see data/clean/switzerland_boundary.gpkg.
    era5_switzerland_points = pl.read_csv("data/clean/era5_switzerland_points.csv")
    all_locations = pl.concat(
        [
            icp_locations.select(pl.col("plot_id").cast(pl.Utf8), "Lat", "Lon"),
            era5_switzerland_points,
        ]
    )

    target_lats = xr.DataArray(all_locations["Lat"].to_list(), dims="point")
    target_lons = xr.DataArray(all_locations["Lon"].to_list(), dims="point")

    variables = [
        "2m_temperature",
        "total_precipitation",
        "surface_solar_radiation_downwards",
        # "leaf_area_index_high_vegetation",
        # "leaf_area_index_low_vegetation",
    ]

    output_dir = "data/era5"

    # change here the years and variables as needed
    years = [str(y) for y in range(2000, 2001)]

    for year in years:
        fetch_era5_data(year)
