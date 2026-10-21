"""Process ICP Level II deposition data into monthly plot-level series.

This script loads cleaned plot-level deposition sampling periods, splits each
period across the months it overlaps (pro rata to days), fills missing months
per plot, and writes a parquet output.

Null handling strategy
----------------------
1. Sentinel, invalid and implausible values are converted to null during
    cleaning; months where a variable has no valid value stay null.
2. Missing plot-month rows are created on a complete monthly grid.
3. Deposition variables are first imputed per plot using a centered rolling
    mean (default window=5).
4. Remaining nulls are then imputed from month-of-year climatology using the
    same month in adjacent years (Y-2, Y-1, Y+1, Y+2) for the same plot.
5. ``num_deposition_obs`` and ``monthly_precip`` are filled with 0 when null.

Total N (``dep_n_tot``) is not filled: it keeps the measured values only and stays
null in the other months.

Some nulls can still remain when no neighboring or adjacent-year values exist
for a given plot-variable-month.

The ``avg_<var>`` and ``sum_<var>`` columns hold the mean and the sum of each
`ROLLING_COLUMNS` variable over the last 12 months (the month and the 11 before it)
of the plot.
"""

import logging
import os

import polars as pl
import polars.selectors as cs

from trunx.config import clean_data_folder
from trunx.datasets.icp_level2_data import _load_deposition_periods, _sum_or_null

logger = logging.getLogger(__name__)

# Deposition with 12-month mean and sum columns
ROLLING_COLUMNS = ["dep_n_tot", "dep_s_so4"]
# Smallest share of a month's days its sampling periods must cover for its sums to be
# scaled up to the whole month; less covered months are left missing
MIN_MONTH_COVERAGE = 0.5
# Longest gap (days) to the previous collection a point-dated sample is taken to cover
MAX_POINT_SAMPLE_DAYS = 62


def aggregate_monthly_deposition(
    df_dep: pl.DataFrame,
    flux_cols: list[str],
    non_conc: list[str],
) -> pl.DataFrame:
    """Aggregate deposition sampling periods to monthly plot-level summaries.

    Each period is split across the months it overlaps, pro rata to the days
    covered. Each monthly sum is then scaled to the whole month by the share of its days
    covered by periods with a value; months covered less than `MIN_MONTH_COVERAGE` are
    left missing. Overlapping periods cover more than the month, which the scaling
    corrects. Point-dated samples (start = end, mostly before 2002) cover the time since
    the plot's previous collection, up to `MAX_POINT_SAMPLE_DAYS`.

    Parameters
    ----------
    df_dep : pl.DataFrame
        Plot-level deposition records, one per sampling period.
    flux_cols : list[str]
        Deposition columns aggregated by monthly sum.
    non_conc : list[str]
        Deposition columns aggregated by monthly mean.

    Returns
    -------
    pl.DataFrame
        Monthly deposition records with one row per plot and month.
    """
    # Sampling periods are [date_start, date_end); zero-length ones count as one day.
    prev_end = pl.col("date_end").shift(1).over("plot_id")
    gap = (pl.col("date_end") - prev_end).dt.total_days()
    periods = (
        df_dep.sort("plot_id", "date_end", "date_start")
        .with_columns(
            date_start=pl.when(
                (pl.col("date_start") == pl.col("date_end"))
                & gap.is_between(1, MAX_POINT_SAMPLE_DAYS)
            )
            .then(prev_end)
            .otherwise(pl.col("date_start"))
        )
        .with_columns(
            n_days=pl.max_horizontal(
                (pl.col("date_end") - pl.col("date_start")).dt.total_days(), pl.lit(1)
            )
        )
        .with_columns(period_end=pl.col("date_start") + pl.duration(days=pl.col("n_days")))
        .with_columns(
            month_start=pl.date_ranges(
                pl.col("date_start").dt.month_start(),
                pl.col("period_end").dt.offset_by("-1d").dt.month_start(),
                interval="1mo",
            )
        )
        .explode("month_start")
        .with_columns(
            weight=(
                pl.min_horizontal("period_end", pl.col("month_start").dt.offset_by("1mo"))
                - pl.max_horizontal("date_start", "month_start")
            ).dt.total_days()
            / pl.col("n_days")
        )
    )

    summed = {**{c: c for c in flux_cols}, "monthly_precip": "quantity"}
    covered_days = pl.col("weight") * pl.col("n_days")
    monthly_agg: list[pl.Expr] = [
        pl.len().alias("num_deposition_obs"),
        pl.col("month_start").first().dt.month_end().dt.day().alias("days_in_month"),
    ]
    for name, col in summed.items():
        monthly_agg += [
            _sum_or_null(pl.col(col) * pl.col("weight")).alias(name),
            covered_days.filter(pl.col(col).is_not_null()).sum().alias(f"coverage_{name}"),
        ]
    if non_conc:
        monthly_agg.append(cs.by_name(*non_conc).mean())

    coverage = {name: pl.col(f"coverage_{name}") / pl.col("days_in_month") for name in summed}
    df_monthly = (
        periods.with_columns(
            pl.col("month_start").dt.year().alias("year"),
            pl.col("month_start").dt.month().alias("month"),
        )
        .group_by("plot_id", "year", "month")
        .agg(monthly_agg)
        .with_columns(
            pl.when(coverage[name] >= MIN_MONTH_COVERAGE)
            .then(pl.col(name) / coverage[name])
            .alias(name)
            for name in summed
        )
        .drop("days_in_month", *(f"coverage_{name}" for name in summed))
        .with_columns(pl.date(pl.col("year"), pl.col("month"), 1).dt.month_end().alias("date"))
        .sort("plot_id", "year", "month")
    )
    return df_monthly


def fill_missing_months(
    df_monthly: pl.DataFrame,
    flux_cols: list[str],
    non_conc: list[str],
    rolling_window: int = 5,
) -> pl.DataFrame:
    """Fill missing plot-months and impute deposition variables by rolling means.

    Parameters
    ----------
    df_monthly : pl.DataFrame
        Monthly deposition records.
    flux_cols : list[str]
        Flux-like deposition columns.
    non_conc : list[str]
        Non-flux deposition columns.
    rolling_window : int, default 5
        Window size used for centered per-plot rolling mean imputation.

    Returns
    -------
    pl.DataFrame
        Monthly table with complete per-plot monthly ranges.
    """
    plot_ranges = df_monthly.group_by("plot_id").agg(
        pl.col("date").min().alias("start"),
        pl.col("date").max().alias("end"),
    )

    complete_grid = (
        plot_ranges.with_columns(
            pl.date_ranges(
                start=pl.col("start"),
                end=pl.col("end"),
                interval="1mo",
                eager=False,
            ).alias("date")
        )
        .explode("date")
        .with_columns(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        )
        .select("plot_id", "year", "month", "date")
    )

    df_complete = complete_grid.join(
        df_monthly,
        on=["plot_id", "year", "month"],
        how="left",
        suffix="_orig",
    )

    if "date_orig" in df_complete.columns:
        df_complete = df_complete.drop("date_orig")

    dep_cols = [*flux_cols, *non_conc]
    for col in dep_cols:
        if col in df_complete.columns:
            df_complete = df_complete.with_columns(
                pl.when(pl.col(col).is_null())
                .then(
                    pl.col(col)
                    .rolling_mean(
                        window_size=rolling_window,
                        min_samples=1,
                        center=True,
                    )
                    .over("plot_id")
                )
                .otherwise(pl.col(col))
                .alias(col)
            )

    # Fallback: impute remaining nulls from month-of-year climatology using
    # the same month in the four adjacent years (Y-2, Y-1, Y+1, Y+2) per plot.
    year_offsets = [-2, -1, 1, 2]
    for col in dep_cols:
        if col in df_complete.columns:
            adjacent_year_cols: list[str] = []
            source = df_complete.select("plot_id", "year", "month", pl.col(col))

            for offset in year_offsets:
                candidate_col = f"{col}_adj_year_{offset:+d}"
                adjacent_year_cols.append(candidate_col)
                shifted = source.select(
                    pl.col("plot_id"),
                    (pl.col("year") - offset).alias("year"),
                    pl.col("month"),
                    pl.col(col).alias(candidate_col),
                )
                df_complete = df_complete.join(
                    shifted,
                    on=["plot_id", "year", "month"],
                    how="left",
                )

            climatology = pl.mean_horizontal([pl.col(name) for name in adjacent_year_cols])
            df_complete = df_complete.with_columns(
                pl.when(pl.col(col).is_null()).then(climatology).otherwise(pl.col(col)).alias(col)
            ).drop(adjacent_year_cols)

    for col in ["num_deposition_obs", "monthly_precip"]:
        if col in df_complete.columns:
            df_complete = df_complete.with_columns(pl.col(col).fill_null(0).alias(col))

    # Rolling-mean/climatology imputation of near-zero flux values can leave a
    # tiny (~1e-16) negative float64 rounding residual; fluxes can't be negative.
    df_complete = df_complete.with_columns(
        [pl.col(col).clip(lower_bound=0.0) for col in flux_cols if col in df_complete.columns]
    )

    return df_complete.sort("plot_id", "year", "month")


def add_12_month_aggregates(df_monthly: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Add the mean and the sum of each of `cols` over the last 12 months of its plot.

    Both cover the month and the 11 before it. The mean ignores missing values, so the
    first months of a plot's record average the months available so far; the sum is
    only given when all 12 months have a value, so it is a full annual deposition.

    Parameters
    ----------
    df_monthly : pl.DataFrame
        Monthly deposition records with `plot_id`, `date` and `cols` columns.
    cols : list[str]
        Deposition columns to aggregate.

    Returns
    -------
    pl.DataFrame
        `df_monthly` with `avg_<col>` and `sum_<col>` columns per item of `cols`.
    """
    # Windows over month ends are 13 months long after a leap-year February (29 Feb
    # - 12 mo = 28 Feb of the year before), so they are taken over month starts.
    month = pl.col("date").dt.month_start()
    return df_monthly.sort("plot_id", "date").with_columns(
        *(
            pl.col(col)
            .rolling_mean_by(month, window_size="12mo")
            .over("plot_id")
            .alias(f"avg_{col}")
            for col in cols
        ),
        *(
            pl.col(col)
            .rolling_sum_by(month, window_size="12mo", min_samples=12)
            .over("plot_id")
            .alias(f"sum_{col}")
            for col in cols
        ),
    )


def process_monthly_depositions(
    output_path: str | None = None,
    fill_missing: bool = True,
) -> pl.DataFrame:
    """Build monthly deposition table and write it to parquet.

    Parameters
    ----------
    output_path : str | None, default None
        Output parquet path. Defaults to
        ``data/clean/icp_monthly_deposition.parquet``.
    fill_missing : bool, default True
        Whether to fill missing months and impute missing deposition values, except
        total N.

    Returns
    -------
    pl.DataFrame
        Monthly deposition table.
    """
    if output_path is None:
        output_path = str(os.path.join(clean_data_folder, "icp_monthly_deposition.parquet"))

    df_dep, flux_cols, non_conc = _load_deposition_periods()
    logger.info("Loaded %d cleaned deposition periods", df_dep.height)

    df_monthly = aggregate_monthly_deposition(df_dep, flux_cols=flux_cols, non_conc=non_conc)
    logger.info("Built %d monthly deposition rows", df_monthly.height)

    if fill_missing:
        # Total N is kept as measured, missing where it is not
        fill_cols = [c for c in flux_cols if c != "dep_n_tot"]
        df_monthly = fill_missing_months(df_monthly, flux_cols=fill_cols, non_conc=non_conc)
        logger.info("After month filling: %d rows", df_monthly.height)

    df_monthly = add_12_month_aggregates(df_monthly, ROLLING_COLUMNS)
    df_monthly.write_parquet(output_path)
    logger.info("Saved monthly deposition table to %s", output_path)
    return df_monthly


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = process_monthly_depositions()
    print(out.head())
