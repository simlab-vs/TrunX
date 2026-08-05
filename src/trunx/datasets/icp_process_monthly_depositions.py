"""Process ICP Level II deposition data into monthly plot-level series.

This script reads raw ICP deposition data, applies cleans it,
aggregates variables by plot-month, fills missing months per plot,
and writes a parquet output.

Null handling strategy
----------------------
1. Sentinel and invalid numeric values are converted to null during cleaning.
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
from trunx.datasets.icp_level2_data import (
    _DEP_NAMES,
    _DEP_NON_CONC,
    _ICP_FOLDER,
    _find_csv,
    _make_plot_id,
)

logger = logging.getLogger(__name__)


def load_deposition_monthly_base() -> tuple[pl.DataFrame, list[str], list[str]]:
    """Load and clean deposition records before monthly aggregation.

    Returns
    -------
    tuple[pl.DataFrame, list[str], list[str]]
        Cleaned deposition records, flux deposition columns, and non-flux
        deposition columns.
    """
    path = _find_csv(os.path.join(_ICP_FOLDER, "595_dp_*/dp_dem.csv"))
    src_renames = {
        "n_total": "n_tot",
        "c_total": "c_tot",
        "s_total": "s_tot",
        "p_total": "p_tot",
        "conductivity": "cond",
        "alkalinity": "alk",
    }
    dep_rename = {col: f"dep_{col}" for col in _DEP_NAMES}

    header = pl.read_csv(path, separator=";", n_rows=0).columns
    active_src = {k: v for k, v in src_renames.items() if k in header}
    post_src = (set(header) - set(active_src)) | set(active_src.values())
    active_dep = {k: v for k, v in dep_rename.items() if k in post_src}

    df = (
        pl.read_csv(path, separator=";")
        .pipe(_make_plot_id)
        .rename(active_src)
        .rename(active_dep)
        .filter(
            pl.col("date_start").is_not_null()
            & pl.col("date_end").is_not_null()
            & (pl.col("code_sampler") == 1)
        )
    )

    if "code_vsampling" in df.columns:
        df = df.filter(~pl.col("code_vsampling").is_in([2, 3, 4, 7, 9]))

    df = df.filter(~pl.col("code_sampler").eq(8))

    dep_cols = [c for c in dep_rename.values() if c in df.columns]
    non_conc = [c for c in _DEP_NON_CONC if c in dep_cols]
    flux_cols = [c for c in dep_cols if c not in non_conc]

    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in dep_cols])

    if flux_cols:
        df = df.with_columns(
            pl.when(cs.by_name(*flux_cols).ne(-1.0)).then(cs.by_name(*flux_cols)).otherwise(None)
        )

    if dep_cols:
        df = df.with_columns(cs.by_name(*dep_cols).fill_nan(None))

    if {"dep_n_tot", "dep_n_nh4", "dep_n_no3"}.issubset(set(df.columns)):
        n_org = pl.col("dep_n_org").fill_null(0) if "dep_n_org" in df.columns else pl.lit(0)
        df = df.with_columns(
            dep_n_tot=pl.when(pl.col("dep_n_tot").is_null())
            .then(pl.col("dep_n_nh4") + pl.col("dep_n_no3") + n_org)
            .otherwise(pl.col("dep_n_tot"))
        )

    if flux_cols and "quantity" in df.columns:
        df = df.with_columns(cs.by_name(*flux_cols) * pl.col("quantity") / 100)

    df = df.with_columns(
        pl.coalesce(
            pl.col("date_end").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            pl.col("date_end").str.to_datetime(strict=False).cast(pl.Date),
        ).alias("date")
    ).drop_nulls(subset=["date"])

    return df.unique(), flux_cols, non_conc


def aggregate_monthly_deposition(
    df_dep: pl.DataFrame,
    flux_cols: list[str],
    non_conc: list[str],
) -> pl.DataFrame:
    """Aggregate cleaned deposition records to monthly plot-level summaries.

    Parameters
    ----------
    df_dep : pl.DataFrame
        Cleaned deposition records.
    flux_cols : list[str]
        Deposition columns aggregated by monthly sum.
    non_conc : list[str]
        Deposition columns aggregated by monthly mean.

    Returns
    -------
    pl.DataFrame
        Monthly deposition records with one row per plot and month.
    """
    monthly_agg: list[pl.Expr] = [pl.len().alias("num_deposition_obs")]
    if flux_cols:
        monthly_agg.append(cs.by_name(*flux_cols).sum())
    if non_conc:
        monthly_agg.append(cs.by_name(*non_conc).mean())
    if "quantity" in df_dep.columns:
        monthly_agg.append(pl.col("quantity").sum().alias("monthly_precip"))

    df_monthly = (
        df_dep.with_columns(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        )
        .group_by("plot_id", "year", "month")
        .agg(monthly_agg)
        .with_columns(
            pl.date(pl.col("year"), pl.col("month"), 1)
            .dt.offset_by("1mo")
            .dt.offset_by("-1d")
            .alias("date")
        )
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

    df_dep, flux_cols, non_conc = load_deposition_monthly_base()
    logger.info("Loaded %d cleaned deposition rows", df_dep.height)

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
