"""
Module provides functions to read ICOS NRT atmospheric trace gas files.

The `read_icos_nrt_file` function reads a specified ICOS NRT file, extracts the relevant data,
and convert them into Polars DataFrames for further analysis.
"""

import os
from pathlib import Path

import polars as pl


def read_icos_nrt_file(
    file_path: str | Path,
    header_line: int = 38,
    data_start_line: int = 39,
) -> pl.DataFrame:
    """
    Read an ICOS NRT atmospheric trace gas file into a Polars DataFrame.

    Parameters
    ----------
    file_path : str | Path
        Path to the ICOS NRT file (.N2O, .CO2, etc.).
    header_line : int, default=38
        Zero-based index of the line containing column names.
    data_start_line : int, default=39
        Zero-based index where data rows start.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with parsed columns and a `datetime` column.
    """
    file_path = Path(file_path)

    # --- Read file as text ---
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    # --- Extract column names ---
    column_names = lines[header_line].strip().split(";")

    # --- Extract data rows ---
    data = [line.strip().split(";") for line in lines[data_start_line:]]

    # --- Create Polars DataFrame ---
    df = pl.DataFrame(data, schema=column_names)

    # --- Cast numeric columns ---
    int_cols = ["Year", "Month", "Day", "Hour", "Minute", "NbPoints", "InstrumentId", "QualityId"]
    float_cols = ["DecimalDate", "n2o", "Stdev"]

    df = df.with_columns(
        [pl.col(c).cast(pl.Int32, strict=False) for c in int_cols if c in df.columns]
        + [pl.col(c).cast(pl.Float64, strict=False) for c in float_cols if c in df.columns]
    )

    # --- Create datetime column ---
    df = df.with_columns(
        pl.datetime(
            pl.col("Year"),
            pl.col("Month"),
            pl.col("Day"),
            pl.col("Hour"),
            pl.col("Minute"),
        ).alias("datetime")
    )

    # --- Reorder columns (datetime first) ---
    df = df.select(["datetime"] + [c for c in df.columns if c != "datetime"])

    return df


if __name__ == "__main__":
    folder_path = "./data/raw/ICOS/"
    files = [f for f in os.listdir(folder_path) if not f.endswith(".zip") and f != ".DS_Store"]

    for file in files:
        df = read_icos_nrt_file(os.path.join(folder_path, file))

        print(df.head())
