"""Build and gap-fill monthly ICP weather data for the 3PG model."""

import datetime
import logging
import os

import pandas as pd
import polars as pl

from trunx.config import clean_data_folder
from trunx.datasets.era5_icp_weather import get_plot_weather

logger = logging.getLogger(__name__)


def aggregate_icp_monthly(df: pl.DataFrame, plot_id: str) -> pl.DataFrame:
    """Aggregate raw ICP weather records to monthly values for one plot.

    Performs no ERA5 gap-filling — a metric is null for months where ICP has
    no observations of it.

    Parameters
    ----------
    df : pl.DataFrame
        Raw ICP weather data with columns ``plot_id``, ``code_variable``,
        ``year``, ``month``, ``daily_mean``, ``daily_min``, ``daily_max``.
    plot_id : str
        Plot identifier to aggregate.

    Returns
    -------
    pl.DataFrame
        Monthly ICP weather with columns ``year``, ``month``, ``tmp_ave``,
        ``tmp_min``, ``tmp_max``, ``frost_days``, ``prcp``, ``srad``.
    """
    plot_df = df.filter(pl.col("plot_id") == plot_id)

    temp_df = plot_df.filter(pl.col("code_variable") == "AT")
    prcp_df = plot_df.filter(pl.col("code_variable") == "PR")
    # Convert from W/m² to MJ/m² (1 W/m² = 0.0864 MJ/m²)
    srad_df = plot_df.filter(pl.col("code_variable") == "SR").with_columns(
        pl.col("daily_mean", "daily_min", "daily_max") * 0.0864
    )

    mtemp_df = (
        temp_df.group_by(["year", "month"])
        .agg(
            pl.col("daily_mean").mean().alias("tmp_ave"),
            pl.col("daily_min").mean().alias("tmp_min"),
            pl.col("daily_max").mean().alias("tmp_max"),
            (pl.col("daily_min") < 0).sum().alias("frost_days"),
        )
        .with_columns(
            pl.when(pl.col("tmp_ave") < pl.col("tmp_min"))
            .then(pl.col("tmp_ave"))
            .otherwise(pl.col("tmp_min"))
            .alias("tmp_min"),
            pl.when(pl.col("tmp_ave") > pl.col("tmp_max"))
            .then(pl.col("tmp_ave"))
            .otherwise(pl.col("tmp_max"))
            .alias("tmp_max"),
        )
    )
    mprcp_df = prcp_df.group_by(["year", "month"]).agg(pl.col("daily_mean").sum().alias("prcp"))
    msrad_df = srad_df.group_by(["year", "month"]).agg(pl.col("daily_mean").mean().alias("srad"))

    return (
        mtemp_df.join(mprcp_df, on=["year", "month"], how="full", coalesce=True)
        .join(msrad_df, on=["year", "month"], how="full", coalesce=True)
        .with_columns(pl.col("frost_days").cast(pl.Int32))
        .select("year", "month", "tmp_ave", "tmp_min", "tmp_max", "frost_days", "prcp", "srad")
        .sort(["year", "month"])
    )


def create_weather_input(df: pl.DataFrame, plot_id: str) -> tuple[list, pl.DataFrame]:
    """Create weather data input for 3PG model.

    Any of temperature, precipitation, or solar radiation entirely missing
    from the ICP data is filled from ERA5 reanalysis for the same plot.
    """
    weather_df = aggregate_icp_monthly(df, plot_id)

    value_groups = {
        "temperature": ["tmp_ave", "tmp_min", "tmp_max", "frost_days"],
        "precipitation": ["prcp"],
        "solar radiation": ["srad"],
    }
    empty_vars = [
        name for name, cols in value_groups.items() if weather_df[cols[0]].drop_nulls().is_empty()
    ]

    if empty_vars:
        era5_path = os.path.join(clean_data_folder, "era5_weather_icp_plots.parquet")
        era5_df = pl.read_parquet(era5_path)
        _, era5_weather = get_plot_weather(plot_id, era5_df)

        if era5_weather.is_empty():
            raise ValueError(
                f"No ICP or ERA5 data for plot_id {plot_id}: {', '.join(empty_vars)}. "
                "Please check the input data."
            )

        logger.info(
            "No ICP %s data for plot_id %s — filling from ERA5",
            ", ".join(empty_vars),
            plot_id,
        )
        fill_cols = [c for name in empty_vars for c in value_groups[name]]
        weather_df = (
            weather_df.join(
                era5_weather.select("year", "month", *fill_cols),
                on=["year", "month"],
                how="full",
                coalesce=True,
                suffix="_era5",
            )
            .with_columns(
                [pl.coalesce(pl.col(c), pl.col(f"{c}_era5")).alias(c) for c in fill_cols]
            )
            .select("year", "month", "tmp_ave", "tmp_min", "tmp_max", "frost_days", "prcp", "srad")
        )

    weather_df = weather_df.sort(["year", "month"]).drop_nulls()

    # Check for gaps
    weather_pl = weather_df.with_columns(pl.date(pl.col("year"), pl.col("month"), 1).alias("date"))
    min_date = weather_pl.select(pl.col("date").min()).item()
    max_date = weather_pl.select(pl.col("date").max()).item()
    all_months = pl.date_range(start=min_date, end=max_date, interval="1mo", eager=True)
    existing_months = set(weather_pl.select("date").to_series().to_list())
    miss_months = sorted(set(all_months) - existing_months)

    return miss_months, weather_df


def weather_summary(weather_df):
    """Summarize weather data."""
    summary_stats = {
        "Metric": [
            "Total records (months)",
            "Start date",
            "End date",
            "Expected months (based on date range)",
            "Actual months present",
            "Missing months",
            "Completeness (%)",
            "Has gaps",
            "Number of gaps",
            "Largest gap (months)",
            "Average gap size (months)",
            "Unique years",
            "Unique months",
        ],
        "Value": [],
    }

    # Calculate basic metrics
    total_records = weather_df.height
    start_date = (
        weather_df.select(pl.col("year").min()).item(),
        weather_df.select(pl.col("month").min()).item(),
    )
    end_date = (
        weather_df.select(pl.col("year").max()).item(),
        weather_df.select(pl.col("month").max()).item(),
    )

    # Calculate expected months if continuous
    start_year, start_month = start_date
    end_year, end_month = end_date
    expected_months = (end_year - start_year) * 12 + (end_month - start_month) + 1

    # Check for gaps
    weather_pl = weather_df.with_columns(pl.date(pl.col("year"), pl.col("month"), 1).alias("date"))
    min_date = weather_pl.select(pl.col("date").min()).item()
    max_date = weather_pl.select(pl.col("date").max()).item()
    all_months = pl.date_range(start=min_date, end=max_date, interval="1mo", eager=True)
    existing_months = set(weather_pl.select("date").to_series().to_list())
    miss_months = sorted(set(all_months) - existing_months)

    # Calculate gap statistics
    gaps = []
    if miss_months:
        # Find consecutive missing months (gaps)
        missing_series = pl.Series(miss_months)
        if len(missing_series) > 0:
            diff = missing_series.diff().dt.total_days()
            gap_starts = missing_series.filter((diff.is_null()) | (diff > 31)).to_list()
            gap_lengths = []

            if len(gap_starts) > 0:
                for i, gap_start in enumerate(gap_starts):
                    if i < len(gap_starts) - 1:
                        gap_end = gap_starts[i + 1]
                        gap_months = len(
                            missing_series.filter(
                                (missing_series >= gap_start) & (missing_series < gap_end)
                            )
                        )
                    else:
                        gap_months = len(missing_series.filter(missing_series >= gap_start))
                    gap_lengths.append(gap_months)
            else:
                gap_lengths = [len(missing_series)]

            gaps = gap_lengths
        else:
            gaps = [len(miss_months)]

    # Count years with complete data
    weather_df = weather_df.with_columns(
        (pl.col("year").cast(pl.Utf8) + "-" + pl.col("month").cast(pl.Utf8).str.zfill(2)).alias(
            "year_month"
        )
    )
    # Populate summary values
    summary_stats["Value"].extend(
        [
            total_records,
            f"{start_date[0]}-{start_date[1]:02d}",
            f"{end_date[0]}-{end_date[1]:02d}",
            expected_months,
            total_records,
            len(miss_months),
            f"{(total_records / expected_months * 100):.1f}%",
            "Yes" if miss_months else "No",
            len(gaps),
            max(gaps) if gaps else 0,
            f"{sum(gaps) / len(gaps):.1f}" if gaps else "N/A",
            weather_df.select(pl.col("year").n_unique()).item(),
            weather_df.select(pl.col("month").n_unique()).item(),
        ]
    )

    if len(miss_months) > 0:
        print("The data is not complete. \n")

    summary_df = pd.DataFrame(summary_stats)

    # Detailed missing months analysis

    print("MISSING MONTHS ANALYSIS")

    if miss_months:
        # Group missing months by year
        missing_df = pl.DataFrame({"missing_date": miss_months})
        missing_df = missing_df.with_columns(
            [
                pl.col("missing_date").dt.year().alias("year"),
                pl.col("missing_date").dt.month().alias("month"),
            ]
        )
        missing_by_year = missing_df.group_by("year").agg(pl.len()).sort("year")

        print("\nMissing months by year:")
        print(missing_by_year)

        print("\nFirst 10 missing months:")
        for month in miss_months[:10]:
            print(f"  - {month.strftime('%B %Y')}")

        if len(miss_months) > 10:
            print(f"  ... and {len(miss_months) - 10} more")

        # Check if missing months are at the beginning or end
        if miss_months[0] == all_months[0]:
            print("\n Missing data at the beginning of the time series")
        if miss_months[-1] == all_months[-1]:
            print("\n Missing data at the end of the time series")
    else:
        print("\n No missing months detected. The dataset is perfectly continuous!")

    return summary_df


def fill_weather_with_era5(
    weather_df: pl.DataFrame,
    plot_id: str,
    start_year: int,
) -> tuple[list, pl.DataFrame]:
    """Fill missing monthly weather with ERA5 reanalysis data.

    Ensures the weather series starts at ``start_year`` and has no gaps,
    filling any months missing from ICP data — whether at the start of the
    series or within it — using ERA5 weather already processed for the
    plot location.

    Parameters
    ----------
    weather_df : pl.DataFrame
        Monthly ICP weather data with columns ``year``, ``month``,
        ``tmp_ave``, ``tmp_min``, ``tmp_max``, ``frost_days``, ``prcp``,
        ``srad``.
    plot_id : str
        Plot identifier used to look up the matching ERA5 grid point.
    start_year : int
        First year the weather series must cover.

    Returns
    -------
    tuple[list, pl.DataFrame]
        Months still missing after filling with ERA5 (empty if none), and
        the resulting weather data sorted by year and month.
    """
    weather_df = weather_df.filter(pl.col("year") >= start_year)

    weather_pl = weather_df.with_columns(pl.date(pl.col("year"), pl.col("month"), 1).alias("date"))
    end_date = weather_pl.select(pl.col("date").max()).item()
    full_months = pl.DataFrame(
        {
            "date": pl.date_range(
                datetime.date(start_year, 1, 1), end_date, interval="1mo", eager=True
            )
        }
    ).with_columns(
        pl.col("date").dt.year().alias("year"), pl.col("date").dt.month().alias("month")
    )

    era5_df = pl.read_parquet(os.path.join(clean_data_folder, "era5_weather_icp_plots.parquet"))
    _, era5_weather = get_plot_weather(plot_id, era5_df)

    missing_before = full_months.join(weather_df, on=["year", "month"], how="anti")
    filled_from_era5 = missing_before.join(era5_weather, on=["year", "month"], how="inner")
    if filled_from_era5.height:
        logger.info(
            "Filled %d month(s) with ERA5 data for plot_id: %s",
            filled_from_era5.height,
            plot_id,
        )

    value_cols = ["tmp_ave", "tmp_min", "tmp_max", "frost_days", "prcp", "srad"]
    filled_df = (
        full_months.join(weather_df, on=["year", "month"], how="left")
        .join(era5_weather, on=["year", "month"], how="left", suffix="_era5")
        .with_columns([pl.coalesce(pl.col(c), pl.col(f"{c}_era5")).alias(c) for c in value_cols])
        .with_columns(pl.col("frost_days").cast(pl.Int32))
        .select(["date", "year", "month", *value_cols])
        .sort(["year", "month"])
    )

    still_missing_df = filled_df.filter(pl.col("tmp_ave").is_null())
    if still_missing_df.height:
        logger.warning(
            "%d month(s) still missing for plot_id %s even after ERA5 fill",
            still_missing_df.height,
            plot_id,
        )

    miss_months = sorted(still_missing_df["date"].to_list())
    weather_df = filled_df.drop_nulls(subset=value_cols).select(["year", "month", *value_cols])

    return miss_months, weather_df
