"""Prepare site data for 3PG model."""

import jax.numpy as jnp
import numpy as np
import polars as pl

from trunx.gp3.model_inputs import SiteData


def prepare_site(site: pl.DataFrame) -> SiteData:
    """Check the site data for consistency."""
    # Ensure Polars DataFrame
    if not isinstance(site, pl.DataFrame):
        site = pl.DataFrame(site)

    # Must contain exactly one row
    if site.height != 1:
        raise ValueError("Site table shall contain exactly one row")

    required_cols = [
        "latitude",
        "altitude",
        "soil_class",
        "asw_i",
        "asw_min",
        "asw_max",
        "from",
        "to",
    ]

    # Check column names and order (R uses identical())
    if site.columns != required_cols:
        raise ValueError(
            "Columns names of the site table must correspond to: "
            "latitude, altitude, soil_class, asw_i, asw_min, asw_max, from, to"
        )

    # Check for NA / null values
    if site.select(pl.any_horizontal(pl.all().is_null())).item():
        raise ValueError("Site table should not contain NAs")

    # Parse dates
    try:
        from_date = pl.Series(site["from"]).str.strptime(pl.Date, "%Y-%m").item()
        to_date = pl.Series(site["to"]).str.strptime(pl.Date, "%Y-%m").item()
    except Exception as err:
        raise ValueError("The simulation dates (from/to) are in the wrong format") from err

    if from_date >= to_date:
        raise ValueError("The start date is later than the end date")

    # Extract scalar values (single row)
    latitude = site["latitude"].item()
    altitude = site["altitude"].item()
    soil_class = site["soil_class"].item()
    asw_i = site["asw_i"].item()
    asw_min = site["asw_min"].item()
    asw_max = site["asw_max"].item()

    # Value checks
    if latitude < -90 or latitude > 90:
        raise ValueError("Latitude shall be within a range of [-90:90]")

    if altitude < 0 or altitude > 4000:
        raise ValueError("Altitude shall be within a range of [0:4000]")

    if soil_class not in range(-1, 5):
        raise ValueError("Soil class shall be within a range of [-1:4]")

    if asw_i < 0:
        raise ValueError("ASW initial shall be greater than 0")

    if asw_min < 0:
        raise ValueError("ASW minimum shall be greater than 0")

    if asw_max < 0:
        raise ValueError("ASW maximum shall be greater than 0")

    # Return final table (same columns, unchanged)
    site = site.select(required_cols)

    row = site.row(0, named=True)
    year_i, month_i = map(int, row["from"].split("-"))

    site_data = SiteData(
        latitude=jnp.asarray([row["latitude"]]),
        altitude=jnp.asarray([row["altitude"]]),
        soil_class=jnp.asarray([row["soil_class"]]),
        ASW=jnp.asarray([row["asw_i"]]),
        ASW_max=jnp.asarray([row["asw_max"]]),
        ASW_min=jnp.asarray([row["asw_min"]]),
        year_i=jnp.asarray([year_i]),
        month_i=jnp.asarray([month_i]),
        site_start=np.datetime64(row["from"]),
        site_end=np.datetime64(row["to"]),
    )
    return site_data


if __name__ == "__main__":
    file_path = "./data/data.input.xlsx"
    sheet_name = "site"

    df = pl.read_excel(file_path, sheet_name=sheet_name)

    site_data = prepare_site(df)

    print(site_data)
