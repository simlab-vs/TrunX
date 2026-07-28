"""Check for stand management information in the ICP Level I data (y1_st1)."""

import os

import polars as pl

from trunx.config import clean_data_folder, icp_raw_data_folder
from trunx.datasets.icp_level2_data import _find_csv, _make_plot_id

MANAGEMENT_COLS: list[str] = [
    "code_manage_type",
    "code_manage_intensity_plot",
    "code_manage_intensity_buffer",
    "code_silvicult_system",
    "cutting_year",
    "code_canopy_gaps",
    "code_notimb_util_plot",
    "code_notimb_util_buffer",
]


def load_stand_data_with_locations() -> pl.DataFrame:
    """Load the stand management table joined to the curated plot locations.

    Returns
    -------
    pl.DataFrame
        One row per plot x survey year, restricted to plots present in
        `full_icp_plot_locations.csv`, with `Lat`/`Lon` columns added.
    """
    path = _find_csv(os.path.join(icp_raw_data_folder, "595_y1_*/y1_st1.csv"))
    stand_df = pl.read_csv(path, separator=";", ignore_errors=True).pipe(_make_plot_id)

    icp_loc_path = os.path.join(clean_data_folder, "full_icp_plot_locations.csv")
    locations = pl.read_csv(icp_loc_path, schema_overrides={"plot_id": pl.Utf8})

    return stand_df.join(locations, on="plot_id", how="inner")


if __name__ == "__main__":
    print()
    stand_df = load_stand_data_with_locations()
    print(
        f"Loaded {stand_df.height} rows for {stand_df['plot_id'].n_unique()} plots "
        "(restricted to icp level2 plots)"
    )
    print()
    for col in MANAGEMENT_COLS:
        n_rows = stand_df[col].drop_nulls().len()
        n_plots = stand_df.filter(pl.col(col).is_not_null())["plot_id"].n_unique()
        print(f"{col}: {n_rows} rows ({n_plots} plots) with a value")

    has_management = stand_df.filter(
        pl.any_horizontal(pl.col(col).is_not_null() for col in MANAGEMENT_COLS)
    )
    print(
        f"\n{has_management.height} rows ({has_management['plot_id'].n_unique()} plots) "
        "have at least one management field populated"
    )
    print(has_management.select("plot_id", "Lat", "Lon", "survey_year", *MANAGEMENT_COLS).head(20))
