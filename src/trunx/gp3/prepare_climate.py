"""Prepare climate data for 3PG model."""

from datetime import date

import jax.numpy as jnp
import numpy as np
import polars as pl

from trunx.gp3.model_inputs import ClimateData

def get_vpd(tmin, tmax):
    """Calculate daytime vapor pressure deficit (VPD)."""
    vpd_min = 6.10780 * (17.2690 * tmin / (237.30 + tmin)).exp()
    vpd_max = 6.10780 * (17.2690 * tmax / (237.30 + tmax)).exp()

    return (vpd_max - vpd_min) / 2


def clim_range(climate):
    """Check whether climate data are within plausible ranges."""
    # Temperature
    if (
        climate.select(pl.max_horizontal(["tmp_min", "tmp_max", "tmp_ave"])).max().item() > 50
        or climate.select(pl.min_horizontal(["tmp_min", "tmp_max", "tmp_ave"])).min().item() < -50
    ):
        print("Warning: Temperature is outside of limits (-50 to 50 °C)!")

    if climate.filter(pl.col("tmp_max") < pl.col("tmp_ave")).height > 0:
        raise ValueError("Average temperature is greater than maximum temperature!")

    if climate.filter(pl.col("tmp_ave") < pl.col("tmp_min")).height > 0:
        raise ValueError("Minimum temperature is greater than average temperature!")

    # Precipitation
    if climate.filter(pl.col("prcp") < 0).height > 0:
        raise ValueError("Precipitation has negative values.")

    if climate.filter(pl.col("prcp") > 10000).height > 0:
        print("Warning: Precipitation outside plausible range (0–10000).")

    # Solar radiation
    if climate.filter(pl.col("srad") < 0).height > 0:
        raise ValueError("Solar radiation has negative values.")

    if climate.filter(pl.col("srad") > 100).height > 0:
        print("Warning: Solar radiation outside plausible range (0–100).")

    # Frost days
    if climate.filter(pl.col("frost_days") < 0).height > 0:
        raise ValueError("Frost days have negative values.")

    if climate.filter(pl.col("frost_days") > 31).height > 0:
        print("Warning: Frost days outside plausible range (0–31).")

    # VPD
    if climate.filter(pl.col("vpd_day") < 0).height > 0:
        raise ValueError("VPD has negative values.")

    if climate.filter(pl.col("vpd_day") > 40).height > 0:
        print("Warning: VPD outside plausible range (0–40).")


def prepare_climate(climate, from_="2001-01", to="2010-11"):
    """Prepare climate table for 3-PG simulation."""
    required = ["tmp_min", "tmp_max", "prcp", "srad", "frost_days"]
    missing = [c for c in required if c not in climate.columns]
    if missing:
        raise ValueError(
            "Climate table must include the following columns: "
            "tmp_min, tmp_max, prcp, srad, frost_days"
        )

    if climate.select(pl.col(required).null_count()).to_series().sum() > 0:
        raise ValueError("Climate table should not contain NAs")

    from_date = date.fromisoformat(from_ + "-01")
    to_date = date.fromisoformat(to + "-01")

    if from_date >= to_date:
        raise ValueError("The start date is later than the end date")

    if climate.height == 12:
        # Replicate average climate
        n_years = to_date.year - from_date.year + 1
        month_i = from_date.month
        month_e = to_date.month

        climate = pl.concat([climate] * n_years)

        climate = climate.with_columns(
            [
                pl.Series("year", np.repeat(np.arange(from_date.year, to_date.year + 1), 12)),
                pl.Series("month", np.tile(np.arange(1, 13), n_years)),
            ]
        )

        if month_i > 1:
            climate = climate.slice(month_i - 1)

        if month_e < 12:
            climate = climate.slice(0, climate.height - (12 - month_e))

    else:
        # Subset long climate series
        if not {"year", "month"}.issubset(climate.columns):
            raise ValueError("Climate table must include year and month for subsetting.")

        climate = climate.with_columns(
            pl.date(pl.col("year"), pl.col("month"), pl.lit(1)).alias("date")
        )
        
        if from_date < climate["date"].min() or to_date > climate["date"].max():
            raise ValueError(
                "Requested time period is outside of provided dates in climate table."
            )

        climate = climate.filter((pl.col("date") >= from_date) & (pl.col("date") <= to_date))

    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

    climate = climate.with_columns(
        pl.when(pl.col("frost_days").is_not_null())
        .then(
            pl.struct(["frost_days", "month"]).map_elements(
                lambda x: min(x["frost_days"], days_in_month[x["month"] - 1])
            )
        )
        .otherwise(pl.col("frost_days"))
        .alias("frost_days")
    )

    if "tmp_ave" not in climate.columns:
        climate = climate.with_columns(
            ((pl.col("tmp_min") + pl.col("tmp_max")) / 2).alias("tmp_ave")
        )

    if "vpd_day" not in climate.columns:
        climate = climate.with_columns(
            get_vpd(pl.col("tmp_min"), pl.col("tmp_max")).alias("vpd_day")
        )

    if "co2" not in climate.columns:
        climate = climate.with_columns(pl.lit(350.0).alias("co2"))

    if "d13catm" not in climate.columns:
        climate = climate.with_columns(pl.lit(-7.1).alias("d13catm"))

    climate = climate.select(
        [
            "year",
            "month",
            "tmp_min",
            "tmp_max",
            "tmp_ave",
            "prcp",
            "srad",
            "frost_days",
            "vpd_day",
            "co2",
            "d13catm",
        ]
    )

    clim_range(climate)

    n_years = len(climate.select(pl.col("year")).unique())
    start_month = np.datetime64(
        str(climate.select(pl.col("year")).min().item())
        + "-"
        + str(climate.select(pl.col("month")).min().item()).zfill(2),
        "M",
    )

    climate_data = ClimateData(
        T_avg=jnp.array(climate["tmp_ave"].to_numpy()),
        precip=jnp.array(climate["prcp"].to_numpy()),
        solar_rad=jnp.array(climate["srad"].to_numpy()),
        frost_days=jnp.array(climate["frost_days"].to_numpy()),
        n_days=jnp.tile(jnp.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]), n_years),
        VPD=jnp.array(climate["vpd_day"].to_numpy()),
        start_month=start_month,
    )

    return climate_data


if __name__ == "__main__":
    file_path = "./data/data.input.xlsx"
    sheet_name = "climate"

    climate = pl.read_excel(file_path, sheet_name=sheet_name)

    climate = prepare_climate(climate)

    print(climate)
