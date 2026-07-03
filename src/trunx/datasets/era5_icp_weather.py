"""Prepare monthly ERA5 weather time-series for ICP plot locations.

Unit conversions applied
------------------------
- Temperature  : ERA5 `t2m` (K) -> °C  by subtracting 273.15
- Precipitation: ERA5 `tp` (m)  ->  mm  by multiplying by 1 000
- Solar rad.   : ERA5 `ssrd` (J m⁻²) -> MJ m⁻² day⁻¹ by dividing by 1 000 000
"""

import logging
import os

import polars as pl
from haversine import haversine

from trunx.config import clean_data_folder, era5_data_folder

logger = logging.getLogger(__name__)

_KELVIN_OFFSET = 273.15
_ERA5_PARQUETS = {
    "era5_total_precipitation": "tp",
    "era5_2m_temperature": "t2m",
    "era5_surface_solar_radiation_downwards": "ssrd",
}


def _normalize_plot_id(x: float) -> str:
    """Convert a float plot identifier to zero-padded string 'CC.PPPP'."""
    country = int(x)
    plot = round((x - country) * 10000)
    return f"{country:02d}.{plot:04d}"


def build_icp_era5_mapping(
    icp_locations: pl.DataFrame,
    era5_locations: pl.DataFrame,
) -> pl.DataFrame:
    """Find the nearest ERA5 grid point for each ICP plot."""
    distances = [
        {
            "plot_id": icp_locations["plot_id"][i],
            "icp_lat": icp_locations["Lat"][i],
            "icp_lon": icp_locations["Lon"][i],
            "era5_lat": era5_locations["latitude"][j],
            "era5_lon": era5_locations["longitude"][j],
            "distance_km": haversine(
                (icp_locations["Lat"][i], icp_locations["Lon"][i]),
                (era5_locations["latitude"][j], era5_locations["longitude"][j]),
            ),
        }
        for i in range(len(icp_locations))
        for j in range(len(era5_locations))
    ]
    df = pl.DataFrame(distances)
    return df.join(
        df.group_by("plot_id").agg(pl.col("distance_km").min()),
        on=["plot_id", "distance_km"],
        how="inner",
    ).unique("plot_id")


def _load_era5() -> pl.DataFrame:
    """Load ERA5 parquets and return one row per (date, location) with daily stats.

    Temperature is aggregated across all available time steps per day to produce
    ``t2m_min``, ``t2m_max``, and ``t2m_mean`` (all in K).  Precipitation and
    solar radiation are summed / averaged per day as appropriate.

    Returns
    -------
    pl.DataFrame
        Columns ``date``, ``latitude``, ``longitude``,
        ``t2m_min``, ``t2m_max``, ``t2m_mean`` (K),
        ``tp`` (m), ``ssrd`` (J m⁻²).
    """

    def _parse_date(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.drop(["time", "step", "point", "number", "surface"])
            .with_columns(
                pl.col("valid_time").str.slice(0, 10).str.to_datetime("%Y-%m-%d").alias("date")
            )
            .drop("valid_time")
        )

    t2m_daily = (
        _parse_date(pl.read_parquet(os.path.join(era5_data_folder, "era5_2m_temperature.parquet")))
        .group_by(["date", "latitude", "longitude"])
        .agg(
            pl.col("t2m").min().alias("t2m_min"),
            pl.col("t2m").max().alias("t2m_max"),
            pl.col("t2m").mean().alias("t2m_mean"),
        )
    )

    tp = (
        _parse_date(
            pl.read_parquet(os.path.join(era5_data_folder, "era5_total_precipitation.parquet"))
        )
        .group_by(["date", "latitude", "longitude"])
        .agg(pl.col("tp").sum())
    )

    ssrd = (
        _parse_date(
            pl.read_parquet(
                os.path.join(era5_data_folder, "era5_surface_solar_radiation_downwards.parquet")
            )
        )
        .group_by(["date", "latitude", "longitude"])
        .agg(pl.col("ssrd").mean())
    )

    return t2m_daily.join(tp, on=["date", "latitude", "longitude"]).join(
        ssrd, on=["date", "latitude", "longitude"]
    )


def _aggregate_monthly(
    era5_daily: pl.DataFrame,
    mapping: pl.DataFrame,
) -> pl.DataFrame:
    """Attach plot IDs and aggregate ERA5 daily data to monthly weather."""
    era5_plots = era5_daily.join(
        mapping.select(["plot_id", "era5_lat", "era5_lon"]),
        left_on=["latitude", "longitude"],
        right_on=["era5_lat", "era5_lon"],
        how="inner",
    ).with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    )

    return (
        era5_plots.group_by(["plot_id", "year", "month"])
        .agg(
            (pl.col("t2m_mean").mean() - _KELVIN_OFFSET).alias("tmp_ave"),
            (pl.col("t2m_min").mean() - _KELVIN_OFFSET).alias("tmp_min"),
            (pl.col("t2m_max").mean() - _KELVIN_OFFSET).alias("tmp_max"),
            ((pl.col("t2m_min") - _KELVIN_OFFSET) < 0.0).sum().cast(pl.Int32).alias("frost_days"),
            (pl.col("tp") * 1000.0).sum().alias("prcp"),
            (pl.col("ssrd") / 1_000_000.0).mean().alias("srad"),
        )
        .with_columns(
            pl.when(pl.col("tmp_min") > pl.col("tmp_ave"))
            .then(pl.col("tmp_ave"))
            .otherwise(pl.col("tmp_min"))
            .alias("tmp_min"),
            pl.when(pl.col("tmp_max") < pl.col("tmp_ave"))
            .then(pl.col("tmp_ave"))
            .otherwise(pl.col("tmp_max"))
            .alias("tmp_max"),
        )
        .sort(["plot_id", "year", "month"])
    )


def prepare_era5_weather(output_path: str | None = None) -> pl.DataFrame:
    """Build and save monthly ERA5 weather for all ICP plot locations."""
    if output_path is None:
        output_path = str(os.path.join(clean_data_folder, "era5_weather_icp_plots.parquet"))

    icp_locations = pl.read_csv(
        os.path.join(clean_data_folder, "icp_plot_locations.csv")
    ).with_columns(
        pl.col("plot_id").map_elements(_normalize_plot_id, return_dtype=pl.Utf8).alias("plot_id")
    )
    logger.info("Loaded %d ICP plot locations", icp_locations.height)

    era5_daily = _load_era5()
    era5_locations = era5_daily.select(["latitude", "longitude"]).unique()
    logger.info("ERA5: %d unique grid points", era5_locations.height)

    mapping = build_icp_era5_mapping(icp_locations, era5_locations)
    logger.info(
        "Mapped %d ICP plots; median distance = %.1f km",
        mapping.height,
        mapping["distance_km"].median(),
    )

    weather = _aggregate_monthly(era5_daily, mapping)
    logger.info(
        "Aggregated %d plot-months across %d plots",
        weather.height,
        weather["plot_id"].n_unique(),
    )

    weather.write_parquet(output_path)
    logger.info("Saved to %s", output_path)
    return weather


def get_plot_weather(
    plot_id: str,
    weather_df: pl.DataFrame,
) -> tuple[list, pl.DataFrame]:
    """Extract ERA5 monthly weather for one ICP plot."""
    plot_weather = (
        weather_df.filter(pl.col("plot_id") == plot_id)
        .select(["year", "month", "tmp_ave", "tmp_min", "tmp_max", "frost_days", "prcp", "srad"])
        .sort(["year", "month"])
        .drop_nulls()
    )

    if plot_weather.is_empty():
        return [], plot_weather

    dated = plot_weather.with_columns(pl.date(pl.col("year"), pl.col("month"), 1).alias("date"))
    min_date = dated.select(pl.col("date").min()).item()
    max_date = dated.select(pl.col("date").max()).item()
    all_months = pl.date_range(start=min_date, end=max_date, interval="1mo", eager=True)
    existing = set(dated["date"].to_list())
    miss_months = sorted(set(all_months) - existing)

    return miss_months, plot_weather


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = prepare_era5_weather()
    print(df.head())
