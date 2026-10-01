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

Some nulls can still remain when no neighboring or adjacent-year values exist
for a given plot-variable-month.
"""

import logging
import os

import polars as pl
import polars.selectors as cs

from trunx.config import clean_data_folder
from trunx.datasets.icp_level2_data import _load_deposition_periods, _sum_or_null

logger = logging.getLogger(__name__)


def aggregate_monthly_deposition(
    df_dep: pl.DataFrame,
    flux_cols: list[str],
    non_conc: list[str],
) -> pl.DataFrame:
    """Aggregate deposition sampling periods to monthly plot-level summaries.

    Each period is split across the months it overlaps, pro rata to the days
    covered.

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
    periods = (
        df_dep.with_columns(
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

    monthly_agg: list[pl.Expr] = [pl.len().alias("num_deposition_obs")]
    monthly_agg.extend(_sum_or_null(pl.col(c) * pl.col("weight")).alias(c) for c in flux_cols)
    if non_conc:
        monthly_agg.append(cs.by_name(*non_conc).mean())
    monthly_agg.append(_sum_or_null(pl.col("quantity") * pl.col("weight")).alias("monthly_precip"))

    df_monthly = (
        periods.with_columns(
            pl.col("month_start").dt.year().alias("year"),
            pl.col("month_start").dt.month().alias("month"),
        )
        .group_by("plot_id", "year", "month")
        .agg(monthly_agg)
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
        Whether to fill missing months and impute missing deposition values.

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
        df_monthly = fill_missing_months(df_monthly, flux_cols=flux_cols, non_conc=non_conc)
        logger.info("After month filling: %d rows", df_monthly.height)

    df_monthly.write_parquet(output_path)
    logger.info("Saved monthly deposition table to %s", output_path)
    return df_monthly


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = process_monthly_depositions()
    print(out.head())
