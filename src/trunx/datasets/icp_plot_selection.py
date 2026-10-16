"""ICP plot selection criteria.

1. One species makes up at least 90% of the stems per ha, summed over all surveys (the share
   `create_combined_inputs._drop_minor_species` uses).
2. QMD and arthimetic mean should be silmilar (within 10% of each other), substitute of even aged.
3. Plots with at least 4 surveys.
4. No sudden increase or decrease in tree count (change between consecutive surveys should be
   less than 40%) over time.
5. At most `MAX_TOTAL_STEM_LOSS` of the stems per ha lost from the first to the last survey,
   as 3-PG is not given the stem loss.
6. Plots within 500 km of the mean location of the selected plots of their dominant species.
"""

import os

import numpy as np
import plotly.graph_objects as go
import polars as pl
from sklearn.metrics.pairwise import haversine_distances

from trunx.config import SPECIES_INDICES, clean_data_folder
from trunx.gp3.allometrics import dms_to_decimal

DOMINANT_SPECIES_THRESHOLD = 0.9
MAX_QMD_MEAN_RELATIVE_DIFF = 0.15
MIN_OBSERVATION_YEARS = 5
MAX_YEARLY_TREE_COUNT_CHANGE = 0.3
MAX_TOTAL_STEM_LOSS = 0.3
MAX_CENTROID_DISTANCE_KM = 5000.0
EARTH_RADIUS_KM = 6371.0


def _dominant_species_plots(
    df: pl.DataFrame, threshold: float = DOMINANT_SPECIES_THRESHOLD
) -> pl.DataFrame:
    """Find plots where one species accounts for at least `threshold` of the stems.

    The share is the species' stems per ha summed over all surveys over the plot's, each
    tree standing for its `stems_per_ha`, as in the plot-level data
    `create_combined_inputs._drop_minor_species` uses. Only considers species covered by
    the 3PG model (`SPECIES_INDICES`).

    Parameters
    ----------
    df : pl.DataFrame
        Tree-level data with `plot_id`, `specie`, `stems_per_ha` columns.
    threshold : float
        Minimum share of the stems that must belong to the dominant species,
        e.g. 0.95 for 95%.

    Returns
    -------
    pl.DataFrame
        One row per qualifying plot, with its dominant species and fraction.
    """
    species_stems = df.group_by("plot_id", "specie").agg(
        pl.col("stems_per_ha").sum().alias("n_stems")
    )

    return (
        species_stems.with_columns(
            (pl.col("n_stems") / pl.col("n_stems").sum().over("plot_id")).alias(
                "dominant_fraction"
            )
        )
        .sort("n_stems", descending=True)
        .group_by("plot_id", maintain_order=True)
        .first()
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
    min_years: int = MIN_OBSERVATION_YEARS,
    max_relative_change: float = MAX_YEARLY_TREE_COUNT_CHANGE,
) -> pl.DataFrame:
    """Find plots with enough surveys whose tree count never jumps between surveys.

    Tree count is compared survey-over-survey: every pair of consecutive
    surveys must stay within `max_relative_change` of each other, rather than
    just the overall min/max across the full history. A survey is one
    `survey_year` campaign, as its trees can be measured on several dates and
    two surveys can fall in the same calendar year.

    Parameters
    ----------
    df : pl.DataFrame
        Tree-level data with `plot_id`, `survey_year`, `date`, `tree_id` columns.
    min_years : int
        Minimum number of surveys required.
    max_relative_change : float
        Maximum allowed `abs(count - prev_count) / prev_count` between any
        two consecutive surveys, e.g. 0.40 for 40%.

    Returns
    -------
    pl.DataFrame
        One row per qualifying plot, with its number of surveys and the
        largest survey-over-survey relative change observed.
    """
    counts_by_survey = (
        df.group_by("plot_id", "survey_year")
        .agg(
            pl.col("tree_id").n_unique().cast(pl.Int64).alias("n_trees"),
            pl.col("date").min().alias("survey_date"),
        )
        .sort("plot_id", "survey_date")
    )

    changes = counts_by_survey.with_columns(
        pl.col("n_trees").shift(1).over("plot_id").alias("prev_n_trees")
    ).with_columns(
        ((pl.col("n_trees") - pl.col("prev_n_trees")).abs() / pl.col("prev_n_trees")).alias(
            "relative_change"
        )
    )

    return (
        changes.group_by("plot_id")
        .agg(
            pl.col("survey_year").n_unique().alias("n_observations"),
            pl.col("relative_change").drop_nulls().max().alias("max_relative_change"),
        )
        .filter(pl.col("max_relative_change") <= max_relative_change)
        .filter(pl.col("n_observations") >= min_years)
    )


def _low_stem_loss_plots(df: pl.DataFrame, max_loss: float = MAX_TOTAL_STEM_LOSS) -> pl.DataFrame:
    """Find plots losing at most `max_loss` of their stems from the first to the last survey.

    3-PG is not given the stem loss (thinning, mortality), so plots that lose many stems
    have their DBH under-predicted and their biomass over-predicted. Stems per ha are summed
    over the trees of each survey, each standing for its `stems_per_ha`.

    Parameters
    ----------
    df : pl.DataFrame
        Tree-level data with `plot_id`, `survey_year`, `date`, `stems_per_ha` columns.
    max_loss : float
        Maximum allowed `1 - last_stems / first_stems`, e.g. 0.3 for 30%.

    Returns
    -------
    pl.DataFrame
        One row per qualifying plot, with its total stem loss.
    """
    return (
        df.group_by("plot_id", "survey_year")
        .agg(pl.col("stems_per_ha").sum().alias("stems"), pl.col("date").min())
        .sort("plot_id", "date")
        .group_by("plot_id")
        .agg((1 - pl.col("stems").last() / pl.col("stems").first()).alias("total_stem_loss"))
        .filter(pl.col("total_stem_loss") <= max_loss)
    )


def _dms_to_decimal(col: str) -> pl.Expr:
    """Convert a signed DMS string column (e.g. ``"+464900"``) to decimal degrees."""
    return pl.col(col).map_elements(dms_to_decimal, return_dtype=pl.Float64)


def _plot_locations(df: pl.DataFrame) -> pl.DataFrame:
    """Decimal `lat` and `lon` of each plot from its signed DDMMSS coordinate columns."""
    return (
        df.group_by("plot_id")
        .agg(pl.col("plot_latitude", "plot_longitude").first())
        .with_columns(
            _dms_to_decimal("plot_latitude").alias("lat"),
            _dms_to_decimal("plot_longitude").alias("lon"),
        )
        .select("plot_id", "lat", "lon")
    )


def _close_to_species_centroid_plots(
    selected: pl.DataFrame, df: pl.DataFrame, max_distance_km: float = MAX_CENTROID_DISTANCE_KM
) -> pl.DataFrame:
    """Find plots within `max_distance_km` of the mean location of their species' plots.

    Parameters
    ----------
    selected : pl.DataFrame
        Plots with `plot_id` and `specie` columns, forming the species groups.
    df : pl.DataFrame
        Tree-level data with `plot_id`, `plot_latitude`, `plot_longitude` columns.
    max_distance_km : float
        Maximum great-circle distance to the species centroid, in km.

    Returns
    -------
    pl.DataFrame
        One row per qualifying plot, with its distance to the species centroid.
    """
    locations = selected.select("plot_id", "specie").join(
        _plot_locations(df), on="plot_id", how="left"
    )
    # Haversine distance between each plot and its species' mean location, from (lat, lon)
    # in radians
    distances = pl.concat(
        group.select(
            "plot_id",
            pl.Series(
                "centroid_distance_km",
                EARTH_RADIUS_KM
                * haversine_distances(
                    np.radians(group.select("lat", "lon").to_numpy()),
                    np.radians(group.select(pl.col("lat", "lon").mean()).to_numpy()),
                )[:, 0],
            ),
        )
        for _, group in locations.group_by("specie")
    )
    return distances.filter(pl.col("centroid_distance_km") <= max_distance_km)


def select_icp_plots(df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Select ICP plots meeting the dominant-species, even-aged, and stability criteria.

    Prints the number of plots left after each criterion.

    Parameters
    ----------
    df : pl.DataFrame
        Tree-level ICP data with `plot_id`, `specie`, `tree_id`,
        `survey_year`, `date`, `dbh_cm`, `stems_per_ha` columns.

    Returns
    -------
    pl.DataFrame
        One row per qualifying plot, with its dominant species, dominant
        fraction, QMD/mean relative diff, number of observation years, tree
        count relative change and total stem loss.
    """
    if df is None:
        df = pl.read_parquet(os.path.join(clean_data_folder, "icp_tree_data.parquet"))
    print(f"All plots: {df['plot_id'].n_unique()}")

    selected = _dominant_species_plots(df)
    print(f"Dominant species (>= {DOMINANT_SPECIES_THRESHOLD:.0%} of stems): {selected.height}")
    selected = selected.join(_similar_qmd_and_mean_dbh_plots(df), on="plot_id", how="inner")
    print(f"QMD within {MAX_QMD_MEAN_RELATIVE_DIFF:.0%} of mean DBH: {selected.height}")
    enough_surveys = _stable_tree_count_plots(df, max_relative_change=float("inf"))
    selected = selected.join(enough_surveys.select("plot_id"), on="plot_id", how="inner")
    print(f"At least {MIN_OBSERVATION_YEARS} surveys: {selected.height}")
    selected = selected.join(_stable_tree_count_plots(df), on="plot_id", how="inner")
    print(f"Tree count change below {MAX_YEARLY_TREE_COUNT_CHANGE:.0%}: {selected.height}")
    selected = selected.join(_low_stem_loss_plots(df), on="plot_id", how="inner")
    print(f"Total stem loss at most {MAX_TOTAL_STEM_LOSS:.0%}: {selected.height}")
    selected = selected.join(
        _close_to_species_centroid_plots(selected, df), on="plot_id", how="inner"
    )
    print(f"Within {MAX_CENTROID_DISTANCE_KM:.0f} km of species centroid: {selected.height}")

    return selected.rename(
        {
            "max_relative_diff": "max_qmd_relative_diff",
            "max_relative_change": "max_tree_count_relative_change",
        }
    ).sort("plot_id")


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


def dominant_species_plot_ids(
    species: str,
    df: pl.DataFrame | None = None,
    threshold: float = DOMINANT_SPECIES_THRESHOLD,
) -> list[str]:
    """Find the plots where `species` makes up at least `threshold` of the stems.

    Only the dominant-species criterion is applied (see `_dominant_species_plots`);
    use `select_icp_plots` for all selection criteria.

    Parameters
    ----------
    species : str
        Species name, e.g. "Picea abies".
    df : pl.DataFrame | None
        Tree-level ICP data with `plot_id`, `specie`, `stems_per_ha` columns; read from
        `icp_tree_data.parquet` when not given.
    threshold : float
        Minimum share of the stems that must belong to `species`, e.g. 0.9 for 90%.

    Returns
    -------
    list[str]
        Sorted plot ids.
    """
    if df is None:
        df = pl.read_parquet(os.path.join(clean_data_folder, "icp_tree_data.parquet"))
    dominant = _dominant_species_plots(df, threshold).filter(pl.col("specie") == species)
    return sorted(dominant["plot_id"].to_list())


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
    plotted = selected.join(_plot_locations(icp_df), on="plot_id", how="left")

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
        title="Selected ICP plot locations by species",
        legend={"x": 0, "y": 1},
    )

    return fig


if __name__ == "__main__":
    icp_df = pl.read_parquet(os.path.join(clean_data_folder, "icp_tree_data.parquet"))

    selected_plots = select_icp_plots()
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
