"""Identify single-species ICP plots whose stem density stays stable over time."""

import os

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure

from trunx.config import clean_data_folder, raw_data_folder

MAX_RELATIVE_CHANGE = 0.10


def _single_species_plot_ids(df: pl.DataFrame, specie_col: str = "specie") -> pl.DataFrame:
    """Return the ``plot_id``s that contain exactly one species across all dates."""
    return (
        df.group_by("plot_id")
        .agg(pl.col(specie_col).n_unique().alias("n_species"))
        .filter(pl.col("n_species") == 1)
        .select("plot_id")
    )


def find_stable_single_species_plots(
    df: pl.DataFrame,
    specie_col: str = "specie",
    date_col: str = "date",
    stems_col: str | None = None,
    num_obv_required: int = 5,
    max_relative_change: float = MAX_RELATIVE_CHANGE,
) -> pl.DataFrame:
    """Find single-species plots whose stems_n stays within a relative range over time.

    Parameters
    ----------
    df : pl.DataFrame
        Data with ``plot_id``, ``specie``, ``date`` columns.
    stems_col : str | None
        Column already holding the stem count/density for a
        ``(plot_id, date)`` row, e.g. ``"n_stems"`` for data that is
        pre-aggregated to one row per plot per date (such as NFI). If
        ``None``, stems are counted as the number of tree rows per
        ``(plot_id, date)`` group, which is only correct for tree-level
        data (one row per tree, such as ICP).
    num_obv_required : int
        A plot must have more than this many distinct observation years to
        be considered.
    max_relative_change : float
        Maximum allowed ``(max - min) / min`` of stems_n across a plot's
        survey dates, e.g. 0.10 for 10%.

    Returns
    -------
    pl.DataFrame
        One row per qualifying plot, with its species, number of distinct
        years surveyed, and the min/max/relative change of stems_n.
    """
    single_species = _single_species_plot_ids(df, specie_col)
    stems_expr = pl.col(stems_col).first() if stems_col is not None else pl.len()

    stems_by_date = (
        df.join(single_species, on="plot_id", how="inner")
        .group_by("plot_id", date_col)
        .agg(
            pl.col(specie_col).first(),
            stems_expr.alias("stems_n"),
        )
    )

    return (
        stems_by_date.group_by("plot_id")
        .agg(
            pl.col(specie_col).first(),
            pl.col(date_col).dt.year().n_unique().alias("n_observations"),
            pl.col("stems_n").min().alias("stems_n_min"),
            pl.col("stems_n").max().alias("stems_n_max"),
        )
        .with_columns(
            ((pl.col("stems_n_max") - pl.col("stems_n_min")) / pl.col("stems_n_min")).alias(
                "relative_change"
            )
        )
        .filter(pl.col("relative_change") <= max_relative_change)
        .filter(pl.col("n_observations") >= num_obv_required)
        .sort("relative_change")
    )


def _species_plot(specie: str, df: pl.DataFrame, specie_col: str = "specie") -> pl.DataFrame:
    """Return a DataFrame with only the rows for a given species."""
    return df.filter(pl.col(specie_col) == specie)


def plot_max_relative_change(
    df: pl.DataFrame,
    specie_col: str = "specie",
    date_col: str = "date",
    stems_col: str | None = None,
    num_obv_required: int = 5,
    n_plots: int = 10,
) -> Figure:
    """Plot tree count over time for the single-species plots with the largest relative change.

    Plots whose tree count is monotonically increasing or decreasing across
    all survey dates are excluded, since those reflect a steady trend rather
    than the sudden jumps this is meant to surface.

    Parameters
    ----------
    df : pl.DataFrame
        Data with ``plot_id``, ``specie``, ``date`` and ``plot_size_ha`` columns.
    specie_col : str
        Name of the species column in ``df``.
    date_col : str
        Name of the date column in ``df``.
    stems_col : str | None
        Column already holding the stem count/density for a
        ``(plot_id, date)`` row, e.g. ``"n_stems"`` for data that is
        pre-aggregated to one row per plot per date (such as NFI). If
        ``None``, trees are counted as the number of rows per
        ``(plot_id, date)`` group, which is only correct for tree-level
        data (one row per tree, such as ICP).
    num_obv_required : int
        A plot must have more than this many distinct observation years to
        be considered.
    n_plots : int
        Number of plots (highest relative change first) to draw.

    Returns
    -------
    Figure
        Grid of tree-count-over-time subplots, one per plot.
    """
    single_species = _single_species_plot_ids(df, specie_col)
    tree_count_expr = pl.col(stems_col).first() if stems_col is not None else pl.len()

    counts_by_date = (
        df.join(single_species, on="plot_id", how="inner")
        .group_by("plot_id", date_col)
        .agg(
            pl.col(specie_col).first(),
            tree_count_expr.alias("tree_count"),
            pl.col("plot_size_ha").mean().round(2).alias("plot_size_ha"),
        )
    )

    trend = (
        counts_by_date.sort("plot_id", date_col)
        .with_columns(pl.col("tree_count").diff().over("plot_id").alias("diff"))
        .group_by("plot_id")
        .agg(
            (pl.col("diff").drop_nulls() >= 0).all().alias("non_decreasing"),
            (pl.col("diff").drop_nulls() <= 0).all().alias("non_increasing"),
        )
        .filter(~(pl.col("non_decreasing") | pl.col("non_increasing")))
        .select("plot_id")
    )

    ranked_plots = (
        counts_by_date.join(trend, on="plot_id", how="inner")
        .group_by("plot_id")
        .agg(
            pl.col(specie_col).first(),
            pl.col(date_col).dt.year().n_unique().alias("n_observations"),
            pl.col("tree_count").min().alias("tree_count_min"),
            pl.col("tree_count").max().alias("tree_count_max"),
        )
        .with_columns(
            (
                (pl.col("tree_count_max") - pl.col("tree_count_min")) / pl.col("tree_count_min")
            ).alias("relative_change")
        )
        .filter(pl.col("n_observations") > num_obv_required)
    )

    top_plots = ranked_plots.sort("relative_change", descending=True).head(n_plots)

    n_cols = min(5, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for ax, row in zip(axes, top_plots.iter_rows(named=True), strict=False):
        plot_counts = counts_by_date.filter(pl.col("plot_id") == row["plot_id"]).sort(date_col)
        ax.plot(plot_counts[date_col], plot_counts["tree_count"], "o-", color="tab:blue")
        ax.set_title(f"{row['plot_id']} ({row[specie_col]})", fontsize=9)
        ax.set_xlabel("Date")
        ax.set_ylabel("# trees", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(plot_counts[date_col], plot_counts["plot_size_ha"], "s--", color="tab:red")
        ax2.set_ylabel("Plot size (ha)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.ticklabel_format(useOffset=False, axis="y")

    for ax in axes[top_plots.height :]:
        ax.axis("off")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    FORESTS = "NFI"  # "NFI" or "ICP"

    if FORESTS == "NFI":
        df = pl.read_parquet(os.path.join(clean_data_folder, "nfi_cleaned.parquet"))
        species_col = "species"
        date_col = "date"
        stems_col = "n_stems"
    elif FORESTS == "ICP":
        df = pl.read_parquet(os.path.join(clean_data_folder, "icp_tree_data.parquet"))
        species_col = "specie"
        date_col = "date"
        stems_col = None
    elif FORESTS == "OLD ICP":
        df = pl.read_parquet(
            os.path.join(raw_data_folder, "ICP/icpf/03_tidy/cpf-level2_cleaned.parquet")
        )
        species_col = "specie"
        date_col = "period_end"
        stems_col = None
    else:
        raise ValueError(f"Unsupported forest: {FORESTS}")

    stable_plots = find_stable_single_species_plots(
        df,
        specie_col=species_col,
        num_obv_required=5,
        date_col=date_col,
        stems_col=stems_col,
        max_relative_change=MAX_RELATIVE_CHANGE,
    )
    print(
        f"Found {stable_plots.height} single-species plots with stems_n changing "
        f"<={MAX_RELATIVE_CHANGE:.0%}"
    )

    species_plot = _species_plot("Picea abies", stable_plots, specie_col=species_col)
    print(species_plot)

    if FORESTS == "ICP":
        max_rel_change_sp = _species_plot("Picea abies", df, specie_col=species_col)
        plot_max_relative_change(df, specie_col=species_col, date_col=date_col, n_plots=15)
    plt.show()
