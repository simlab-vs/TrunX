"""ICP plot selection criteria.

1. Single species plot.
2. QMD and arthimetic mean should be silmilar (within 10% of each other), substitute of even aged.
3. Plots with at least 5 distinct observation years.
4. No sudden increase or decrease in tree count (change between consecutive years should be
   less than 40%) over time.
"""

import os

import plotly.graph_objects as go
import polars as pl

from trunx.config import SPECIES_INDICES, clean_data_folder

DOMINANT_SPECIES_THRESHOLD = 1
MAX_QMD_MEAN_RELATIVE_DIFF = 0.10
MIN_OBSERVATION_YEARS = 5
MAX_YEARLY_TREE_COUNT_CHANGE = 0.40


def _dominant_species_plots(
    df: pl.DataFrame, threshold: float = DOMINANT_SPECIES_THRESHOLD
) -> pl.DataFrame:
    """Find plots where one species accounts for at least `threshold` of trees.

    Only considers species covered by the 3PG model (`SPECIES_INDICES`).

    Parameters
    ----------
    df : pl.DataFrame
        Tree-level data with `plot_id`, `specie`, `tree_id` columns.
    threshold : float
        Minimum fraction of trees (by unique `tree_id`) that must belong to
        the dominant species, e.g. 0.95 for 95%.

    Returns
    -------
    pl.DataFrame
        One row per qualifying plot, with its dominant species and fraction.
    """
    trees = df.select("plot_id", "specie", "tree_id").unique()

    species_counts = trees.group_by("plot_id", "specie").agg(pl.len().alias("n_trees"))
    plot_totals = trees.group_by("plot_id").agg(pl.len().alias("n_total"))

    return (
        species_counts.sort("n_trees", descending=True)
        .group_by("plot_id", maintain_order=True)
        .first()
        .join(plot_totals, on="plot_id")
        .with_columns((pl.col("n_trees") / pl.col("n_total")).alias("dominant_fraction"))
        .filter(pl.col("dominant_fraction") >= threshold)
        .filter(pl.col("specie").is_in(list(SPECIES_INDICES)))
        .select("plot_id", "specie", "dominant_fraction")
    )


def _similar_qmd_and_mean_dbh_plots(
    df: pl.DataFrame, max_relative_diff: float = MAX_QMD_MEAN_RELATIVE_DIFF
) -> pl.DataFrame:
    """Find plots whose QMD stays close to the arithmetic mean DBH in every survey year.

    A small QMD/mean gap is a common substitute for "even-aged", since
    diameter distributions widen with age variation. Checked in every
    survey year (not just a baseline), so the plot must stay even-aged-like
    throughout its whole survey history.

    Parameters
    ----------
    df : pl.DataFrame
        Tree-level data with `plot_id`, `survey_year`, `dbh_cm` columns.
    max_relative_diff : float
        Maximum allowed `abs(qmd - mean_dbh) / mean_dbh` in any survey year,
        e.g. 0.10 for 10%.

    Returns
    -------
    pl.DataFrame
        One row per qualifying plot, with the largest QMD/mean relative
        difference observed across its survey years.
    """
    by_year = (
        df.group_by("plot_id", "survey_year")
        .agg(
            pl.col("dbh_cm").mean().alias("mean_dbh"),
            pl.col("dbh_cm").pow(2).mean().sqrt().alias("qmd"),
        )
        .with_columns(
            ((pl.col("qmd") - pl.col("mean_dbh")).abs() / pl.col("mean_dbh")).alias(
                "relative_diff"
            )
        )
    )

    return (
        by_year.group_by("plot_id")
        .agg(pl.col("relative_diff").max().alias("max_relative_diff"))
        .filter(pl.col("max_relative_diff") <= max_relative_diff)
    )


def _stable_tree_count_plots(
    df: pl.DataFrame,
    date_col: str = "date",
    min_years: int = MIN_OBSERVATION_YEARS,
    max_relative_change: float = MAX_YEARLY_TREE_COUNT_CHANGE,
) -> pl.DataFrame:
    """Find plots surveyed in enough years whose tree count never jumps between years.

    Tree count is compared year-over-year: every pair of consecutive
    surveyed calendar years must stay within `max_relative_change` of each
    other, rather than just the overall min/max across the full history.

    Parameters
    ----------
    df : pl.DataFrame
        Tree-level data with `plot_id`, `date`, `tree_id` columns.
    date_col : str
        Name of the date column in `df`.
    min_years : int
        Minimum number of distinct calendar years required.
    max_relative_change : float
        Maximum allowed `abs(count - prev_count) / prev_count` between any
        two consecutive surveyed years, e.g. 0.40 for 40%.

    Returns
    -------
    pl.DataFrame
        One row per qualifying plot, with its number of observation years
        and the largest year-over-year relative change observed.
    """
    counts_by_year = (
        df.group_by("plot_id", pl.col(date_col).dt.year().alias("year"))
        .agg(pl.col("tree_id").n_unique().cast(pl.Int64).alias("n_trees"))
        .sort("plot_id", "year")
    )

    changes = counts_by_year.with_columns(
        pl.col("n_trees").shift(1).over("plot_id").alias("prev_n_trees")
    ).with_columns(
        ((pl.col("n_trees") - pl.col("prev_n_trees")).abs() / pl.col("prev_n_trees")).alias(
            "relative_change"
        )
    )

    return (
        changes.group_by("plot_id")
        .agg(
            pl.col("year").n_unique().alias("n_observations"),
            pl.col("relative_change").drop_nulls().max().alias("max_relative_change"),
        )
        .filter(pl.col("max_relative_change") <= max_relative_change)
        .filter(pl.col("n_observations") >= min_years)
    )


def select_icp_plots(df: pl.DataFrame) -> pl.DataFrame:
    """Select ICP plots meeting the dominant-species, even-aged, and stability criteria.

    Parameters
    ----------
    df : pl.DataFrame
        Tree-level ICP data with `plot_id`, `specie`, `tree_id`,
        `survey_year`, `date`, `dbh_cm` columns.

    Returns
    -------
    pl.DataFrame
        One row per qualifying plot, with its dominant species, dominant
        fraction, QMD/mean relative diff, number of observation years, and
        tree count relative change.
    """
    dominant = _dominant_species_plots(df)
    even_aged = _similar_qmd_and_mean_dbh_plots(df)
    stable_counts = _stable_tree_count_plots(df)

    return (
        dominant.join(even_aged, on="plot_id", how="inner")
        .join(stable_counts, on="plot_id", how="inner")
        .rename(
            {
                "max_relative_diff": "max_qmd_relative_diff",
                "max_relative_change": "max_tree_count_relative_change",
            }
        )
        .sort("plot_id")
    )


def plot_ids_by_species(selected: pl.DataFrame) -> dict[str, list[str]]:
    """Group selected plot ids by their dominant species.

    Parameters
    ----------
    selected : pl.DataFrame
        Output of `select_icp_plots`, with `plot_id` and `specie` columns.

    Returns
    -------
    dict[str, list[str]]
        Dominant species mapped to the list of plot ids with that species.
    """
    return {
        specie: group["plot_id"].to_list()
        for (specie,), group in selected.group_by("specie", maintain_order=True)
    }


def _dms_to_decimal(col: str) -> pl.Expr:
    """Convert a signed DDMMSS string column (e.g. ``"+464900"``) to decimal degrees."""
    sign = pl.when(pl.col(col).str.starts_with("-")).then(-1).otherwise(1)
    digits = pl.col(col).str.slice(1)
    degrees = digits.str.slice(0, 2).cast(pl.Float64)
    minutes = digits.str.slice(2, 2).cast(pl.Float64)
    seconds = digits.str.slice(4, 2).cast(pl.Float64)
    return sign * (degrees + minutes / 60 + seconds / 3600)


def plot_plot_locations(selected: pl.DataFrame, icp_df: pl.DataFrame) -> go.Figure:
    """Plot the location of selected plots on an interactive map, colored by species.

    Parameters
    ----------
    selected : pl.DataFrame
        Output of `select_icp_plots`, with `plot_id` and `specie` columns.
    icp_df : pl.DataFrame
        Tree-level ICP data, with its `plot_latitude`/`plot_longitude`
        columns (signed DDMMSS strings).

    Returns
    -------
    go.Figure
        Map of plot locations, colored by dominant species.
    """
    locations = (
        icp_df.select("plot_id", "plot_latitude", "plot_longitude")
        .unique()
        .with_columns(
            _dms_to_decimal("plot_latitude").alias("lat"),
            _dms_to_decimal("plot_longitude").alias("lon"),
        )
    )
    plotted = selected.join(locations, on="plot_id", how="left")

    fig = go.Figure()
    for specie in sorted(plotted["specie"].unique().to_list()):
        specie_df = plotted.filter(pl.col("specie") == specie)
        fig.add_trace(
            go.Scattermap(
                lat=specie_df["lat"].to_list(),
                lon=specie_df["lon"].to_list(),
                mode="markers",
                marker={"size": 10},
                name=specie,
                text=specie_df["plot_id"].to_list(),
                hovertemplate=(
                    "<b>Plot id:</b> %{text}<br>"
                    "<b>Lat:</b> %{lat}<br><b>Lon:</b> %{lon}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        map={
            "style": "open-street-map",
            "zoom": 4,
            "center": {"lat": plotted["lat"].mean(), "lon": plotted["lon"].mean()},
        },
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        title="Selected ICP plot locations by dominant species",
        legend={"x": 0, "y": 1},
    )

    return fig


if __name__ == "__main__":
    icp_df = pl.read_parquet(os.path.join(clean_data_folder, "icp_tree_data.parquet"))

    selected_plots = select_icp_plots(icp_df)
    print(f"Selected {selected_plots.height} of {icp_df['plot_id'].n_unique()} plots")
    print(selected_plots)

    print(
        selected_plots.group_by("specie").agg(
            n_plots=pl.col("plot_id").n_unique(),
            plot_ids=pl.col("plot_id").unique(),
        )
    )

    print(plot_ids_by_species(selected_plots))

    plot_plot_locations(selected_plots, icp_df).show()
