"""Build monthly weather for the 3PG model at one or many lat/lon locations.

Prefers real observations: if an ICP plot is closer to a location than the
nearest available ERA5 grid point, its observed weather is used (gap-filled
from ERA5, same as the ICP-plot pipeline in `weather_processing.py`).
Otherwise, falls back to the nearest ERA5 grid point directly, generalizing
the ICP-plot lookup in `era5_icp_weather.py` to any location.

`get_monthly_weather` looks up a single location. `build_tree_plot_weather`
applies the same rule to every plot in `trunx_tree_level_data.parquet` in one
pass, which is far cheaper than calling `get_monthly_weather` per plot since
the ERA5 data and nearest-point search are shared across all plots.
"""

import logging
import os

import numpy as np
import polars as pl
from haversine import Unit, haversine_vector

from trunx.config import clean_data_folder
from trunx.datasets.era5_icp_weather import (
    _KELVIN_OFFSET,
    _load_era5,
    _normalize_plot_id,
    get_plot_weather,
)
from trunx.gp3.weather_processing import aggregate_icp_monthly, fill_weather_with_era5

logger = logging.getLogger(__name__)

WEATHER_COLUMNS = [
    "plot_id",
    "source",
    "year",
    "month",
    "tmp_ave",
    "tmp_min",
    "tmp_max",
    "frost_days",
    "prcp",
    "srad",
    "weather_source",
]

_ICP_VALUE_COLS = ["tmp_ave", "tmp_min", "tmp_max", "frost_days", "prcp", "srad"]


def _nearest_indices(
    query_points: np.ndarray,
    ref_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest ref_point row index and distance (km) for each query point."""
    distances = haversine_vector(query_points, ref_points, unit=Unit.KILOMETERS, comb=True)
    nearest_idx = distances.argmin(axis=0)
    nearest_dist = distances[nearest_idx, np.arange(distances.shape[1])]
    return nearest_idx, nearest_dist


def _nearest_icp_plot(lat: float, lon: float, icp_locations: pl.DataFrame) -> tuple[str, float]:
    """Find the nearest ICP plot to (lat, lon)."""
    ref_points = icp_locations.select(["Lat", "Lon"]).to_numpy()
    idx, dist = _nearest_indices(np.array([[lat, lon]]), ref_points)
    nearest = icp_locations.row(int(idx[0]), named=True)
    return _normalize_plot_id(nearest["plot_id"]), float(dist[0])


def _nearest_era5_point(
    lat: float,
    lon: float,
    era5_locations: pl.DataFrame,
) -> tuple[float, float, float]:
    """Find the nearest ERA5 grid point to (lat, lon) among available locations."""
    ref_points = era5_locations.select(["latitude", "longitude"]).to_numpy()
    idx, dist = _nearest_indices(np.array([[lat, lon]]), ref_points)
    nearest = era5_locations.row(int(idx[0]), named=True)
    return nearest["latitude"], nearest["longitude"], float(dist[0])


def _monthly_agg_exprs() -> list[pl.Expr]:
    """Polars expressions aggregating ERA5 daily data to monthly weather."""
    return [
        (pl.col("t2m_mean").mean() - _KELVIN_OFFSET).alias("tmp_ave"),
        (pl.col("t2m_min").mean() - _KELVIN_OFFSET).alias("tmp_min"),
        (pl.col("t2m_max").mean() - _KELVIN_OFFSET).alias("tmp_max"),
        ((pl.col("t2m_min") - _KELVIN_OFFSET) < 0.0).sum().cast(pl.Int32).alias("frost_days"),
        (pl.col("tp") * 1000.0).sum().alias("prcp"),
        (pl.col("ssrd") / 1_000_000.0).mean().alias("srad"),
    ]


def _clip_temp_bounds_exprs() -> list[pl.Expr]:
    """Clamp tmp_min/tmp_max so they never cross tmp_ave."""
    return [
        pl.when(pl.col("tmp_min") > pl.col("tmp_ave"))
        .then(pl.col("tmp_ave"))
        .otherwise(pl.col("tmp_min"))
        .alias("tmp_min"),
        pl.when(pl.col("tmp_max") < pl.col("tmp_ave"))
        .then(pl.col("tmp_ave"))
        .otherwise(pl.col("tmp_max"))
        .alias("tmp_max"),
    ]


_WEATHER_VALUE_DTYPES: dict[str, type[pl.DataType]] = {
    "year": pl.Int32,
    "month": pl.Int8,
    "tmp_ave": pl.Float64,
    "tmp_min": pl.Float64,
    "tmp_max": pl.Float64,
    "frost_days": pl.Int32,
    "prcp": pl.Float64,
    "srad": pl.Float64,
}


def _cast_weather_values(df: pl.DataFrame) -> pl.DataFrame:
    """Cast weather value columns to a common dtype so sources concat cleanly."""
    return df.with_columns([pl.col(c).cast(dtype) for c, dtype in _WEATHER_VALUE_DTYPES.items()])


def _icp_weather(
    plot_id: str,
    icp_raw: pl.DataFrame | None = None,
    era5_weather_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build monthly weather for an ICP plot, gap-filled from ERA5.

    Adds a `weather_source` column, per month: "ICP" if every value is a real
    ICP observation, "ERA5" if every value had to be filled from ERA5, and
    "ICP/ERA5" if some values are real ICP observations and others were
    filled (e.g. a station with real temperature but no precipitation
    sensor).
    """
    if icp_raw is None:
        icp_raw = pl.read_parquet(os.path.join(clean_data_folder, "ICP_weather_data.parquet"))
    raw_weather_df = aggregate_icp_monthly(icp_raw, plot_id)

    if raw_weather_df.height == 0:
        logger.info("No ICP weather records for plot %s — using ERA5 only", plot_id)
        if era5_weather_df is None:
            era5_weather_df = pl.read_parquet(
                os.path.join(clean_data_folder, "era5_weather_icp_plots.parquet")
            )
        _, weather_df = get_plot_weather(plot_id, era5_weather_df)
        return _cast_weather_values(weather_df).with_columns(
            pl.lit("ERA5").alias("weather_source")
        )

    icp_status = raw_weather_df.select(
        "year",
        "month",
        pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in _ICP_VALUE_COLS]).alias(
            "_icp_present_count"
        ),
    )

    start_year = int(raw_weather_df.select(pl.col("year").min()).item())
    miss_months, weather_df = fill_weather_with_era5(raw_weather_df, plot_id, start_year)
    if miss_months:
        logger.warning(
            "%d month(s) still missing for ICP plot %s after filling from ERA5",
            len(miss_months),
            plot_id,
        )

    weather_df = (
        weather_df.join(icp_status, on=["year", "month"], how="left")
        .with_columns(pl.col("_icp_present_count").fill_null(0))
        .with_columns(
            pl.when(pl.col("_icp_present_count") == len(_ICP_VALUE_COLS))
            .then(pl.lit("ICP"))
            .when(pl.col("_icp_present_count") == 0)
            .then(pl.lit("ERA5"))
            .otherwise(pl.lit("ICP/ERA5"))
            .alias("weather_source")
        )
        .drop("_icp_present_count")
    )

    return _cast_weather_values(weather_df)


def _era5_weather(era5_daily: pl.DataFrame, lat: float, lon: float) -> pl.DataFrame:
    """Build monthly weather from the ERA5 grid point at (lat, lon)."""
    point_daily = era5_daily.filter(
        (pl.col("latitude") == lat) & (pl.col("longitude") == lon)
    ).with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    )

    return (
        point_daily.group_by(["year", "month"])
        .agg(_monthly_agg_exprs())
        .with_columns(_clip_temp_bounds_exprs())
        .with_columns(pl.lit("ERA5").alias("weather_source"))
        .sort(["year", "month"])
    )


def _era5_monthly_all(era5_daily: pl.DataFrame) -> pl.DataFrame:
    """Aggregate ERA5 daily data to monthly weather for every grid point at once."""
    return (
        era5_daily.with_columns(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        )
        .group_by(["latitude", "longitude", "year", "month"])
        .agg(_monthly_agg_exprs())
        .with_columns(_clip_temp_bounds_exprs())
        .with_columns(pl.lit("ERA5").alias("weather_source"))
    )


def get_monthly_weather(lat: float, lon: float) -> pl.DataFrame:
    """Build monthly weather for the 3PG model at a given location.

    Uses the nearest ICP plot's observed weather (gap-filled from ERA5) when
    it is closer to (lat, lon) than the nearest available ERA5 grid point;
    otherwise uses that ERA5 grid point directly.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees.
    lon : float
        Longitude in decimal degrees.

    Returns
    -------
    pl.DataFrame
        Monthly weather with columns `year`, `month`, `tmp_ave`, `tmp_min`,
        `tmp_max`, `frost_days`, `prcp`, `srad`, `weather_source` ("ICP",
        "ERA5", or "ICP/ERA5" for a month mixing both, per month).
    """
    icp_locations = pl.read_csv(os.path.join(clean_data_folder, "full_icp_plot_locations.csv"))
    icp_plot_id, icp_distance_km = _nearest_icp_plot(lat, lon, icp_locations)

    era5_daily = _load_era5()
    era5_locations = era5_daily.select(["latitude", "longitude"]).unique()
    era5_lat, era5_lon, era5_distance_km = _nearest_era5_point(lat, lon, era5_locations)

    if icp_distance_km <= era5_distance_km:
        logger.info(
            "Nearest ICP plot %s (%.1f km) is closer than the nearest ERA5 grid point "
            "(%.1f km) to (%.4f, %.4f) — using ICP weather, gap-filled from ERA5.",
            icp_plot_id,
            icp_distance_km,
            era5_distance_km,
            lat,
            lon,
        )
        return _icp_weather(icp_plot_id)

    logger.info(
        "Nearest ERA5 grid point (%.1f km) is closer than the nearest ICP plot %s "
        "(%.1f km) to (%.4f, %.4f) — using ERA5 weather.",
        era5_distance_km,
        icp_plot_id,
        icp_distance_km,
        lat,
        lon,
    )
    if era5_distance_km > 50:
        logger.warning(
            "Nearest ERA5 grid point is %.1f km from (%.4f, %.4f) — weather may be a poor "
            "match for this location.",
            era5_distance_km,
            lat,
            lon,
        )
    return _era5_weather(era5_daily, era5_lat, era5_lon)


def build_tree_plot_weather(
    tree_data_path: str | None = None,
    output_path: str | None = None,
) -> pl.DataFrame:
    """Build monthly weather for every plot in `trunx_tree_level_data.parquet`.

    Applies the same rule as `get_monthly_weather` — nearest ICP plot if
    closer than the nearest ERA5 grid point, otherwise ERA5 — to every
    (plot_id, source) pair at once: the ERA5 data, the ERA5/ICP nearest-point
    search, and the ERA5 monthly aggregation are each computed once and
    shared across all plots, rather than reloaded per plot.

    Parameters
    ----------
    tree_data_path : str | None
        Parquet path to read plot locations from. Defaults to
        `clean_data_folder/trunx_tree_level_data.parquet`.
    output_path : str | None
        Parquet path to write the result. Defaults to
        `clean_data_folder/trunx_plot_weather.parquet`.

    Returns
    -------
    pl.DataFrame
        Long-format weather with columns `plot_id`, `source`, `year`,
        `month`, `tmp_ave`, `tmp_min`, `tmp_max`, `frost_days`, `prcp`,
        `srad`, `weather_source` ("ICP", "ERA5", or "ICP/ERA5" for a month
        mixing both, per month) — one row per plot per month.
    """
    if tree_data_path is None:
        tree_data_path = str(os.path.join(clean_data_folder, "trunx_tree_level_data.parquet"))
    if output_path is None:
        output_path = str(os.path.join(clean_data_folder, "trunx_plot_weather.parquet"))

    plot_locations = (
        pl.read_parquet(tree_data_path)
        .group_by(["plot_id", "source"])
        .agg(pl.col("lat").mean(), pl.col("lon").mean())
        .sort(["source", "plot_id"])
    )
    logger.info("Found %d plots in %s", plot_locations.height, tree_data_path)

    icp_locations = pl.read_csv(
        os.path.join(clean_data_folder, "full_icp_plot_locations.csv")
    ).with_columns(
        pl.col("plot_id")
        .map_elements(_normalize_plot_id, return_dtype=pl.Utf8)
        .alias("icp_plot_id")
    )

    era5_daily = _load_era5()
    era5_locations = era5_daily.select(["latitude", "longitude"]).unique()

    plot_coords = plot_locations.select(["lat", "lon"]).to_numpy()
    icp_coords = icp_locations.select(["Lat", "Lon"]).to_numpy()
    era5_coords = era5_locations.select(["latitude", "longitude"]).to_numpy()

    icp_idx, icp_dist_km = _nearest_indices(plot_coords, icp_coords)
    era5_idx, era5_dist_km = _nearest_indices(plot_coords, era5_coords)

    icp_plot_id_arr = icp_locations["icp_plot_id"].to_numpy()
    plot_locations = plot_locations.with_columns(
        pl.Series("icp_plot_id", icp_plot_id_arr[icp_idx]),
        pl.Series("icp_distance_km", icp_dist_km),
        pl.Series("era5_lat", era5_coords[era5_idx, 0]),
        pl.Series("era5_lon", era5_coords[era5_idx, 1]),
        pl.Series("era5_distance_km", era5_dist_km),
    ).with_columns((pl.col("icp_distance_km") <= pl.col("era5_distance_km")).alias("use_icp"))

    n_icp = int(plot_locations["use_icp"].sum())
    logger.info(
        "%d plot(s) matched to a closer ICP plot; %d use ERA5 directly",
        n_icp,
        plot_locations.height - n_icp,
    )

    era5_monthly = _era5_monthly_all(era5_daily)
    era5_weather = (
        plot_locations.filter(~pl.col("use_icp"))
        .select(
            "plot_id",
            "source",
            pl.col("era5_lat").alias("latitude"),
            pl.col("era5_lon").alias("longitude"),
        )
        .join(era5_monthly, on=["latitude", "longitude"], how="left")
        .drop(["latitude", "longitude"])
    )

    icp_plots_needed = plot_locations.filter(pl.col("use_icp"))["icp_plot_id"].unique().to_list()
    if icp_plots_needed:
        icp_raw = pl.read_parquet(os.path.join(clean_data_folder, "ICP_weather_data.parquet"))
        era5_weather_icp_plots = pl.read_parquet(
            os.path.join(clean_data_folder, "era5_weather_icp_plots.parquet")
        )
        icp_weather_by_plot = pl.concat(
            [
                _icp_weather(icp_plot_id, icp_raw, era5_weather_icp_plots).with_columns(
                    pl.lit(icp_plot_id).alias("icp_plot_id")
                )
                for icp_plot_id in icp_plots_needed
            ],
            how="vertical",
        )
        icp_weather = (
            plot_locations.filter(pl.col("use_icp"))
            .select("plot_id", "source", "icp_plot_id")
            .join(icp_weather_by_plot, on="icp_plot_id", how="left")
            .drop("icp_plot_id")
        )
    else:
        icp_weather = era5_weather.clear()

    weather = pl.concat(
        [_standardize_weather(icp_weather), _standardize_weather(era5_weather)],
        how="vertical",
    ).sort(["source", "plot_id", "year", "month"])

    weather.write_parquet(output_path)
    logger.info(
        "Saved %d rows for %d plots to %s", weather.height, plot_locations.height, output_path
    )
    return weather


def _standardize_weather(df: pl.DataFrame) -> pl.DataFrame:
    """Cast a plot weather table to `WEATHER_COLUMNS` with matching dtypes."""
    return _cast_weather_values(
        df.with_columns(
            pl.col("plot_id").cast(pl.Utf8),
            pl.col("source").cast(pl.Utf8),
            pl.col("weather_source").cast(pl.Utf8),
        )
    ).select(WEATHER_COLUMNS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Build the combined weather data.
    # weather_df = build_tree_plot_weather()
    weather_df = pl.read_parquet(os.path.join(clean_data_folder, "trunx_plot_weather.parquet"))
    print(f"\n=== Plot weather ({weather_df.height} rows) ===")
    print(weather_df.head())
    print(weather_df.shape)
