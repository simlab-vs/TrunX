"""Identify single-species ICP plots whose stem density stays stable over time."""

import os

import polars as pl

from trunx.config import clean_data_folder, raw_data_folder

MAX_RELATIVE_CHANGE = 0.40


def find_stable_single_species_plots(
    df: pl.DataFrame,
    specie_col="specie",
    date_col="date",
    num_obv_required: int = 5,
    max_relative_change: float = MAX_RELATIVE_CHANGE,
) -> pl.DataFrame:
    """Find single-species plots whose stems_n stays within a relative range over time.

    Parameters
    ----------
    df : pl.DataFrame
        Tree-level data with ``plot_id``, ``specie``, ``date`` columns.
    num_obv_required : int
        Minimum number of observations required for a plot to be considered.
    max_relative_change : float
        Maximum allowed ``(max - min) / min`` of stems_n across a plot's
        survey dates, e.g. 0.10 for 10%.

    Returns
    -------
    pl.DataFrame
        One row per qualifying plot, with its species, number of distinct
        years surveyed, and the min/max/relative change of stems_n.
    """
    single_species = (
        df.group_by("plot_id")
        .agg(pl.col(specie_col).n_unique().alias("n_species"))
        .filter(pl.col("n_species") == 1)
        .select("plot_id")
    )

    stems_by_date = (
        df.join(single_species, on="plot_id", how="inner")
        .group_by("plot_id", date_col)
        .agg(
            pl.col(specie_col).first(),
            pl.len().alias("stems_n"),
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
        .sort("relative_change")
        .filter(pl.col("n_observations") >= num_obv_required)
    )


def _species_plot(specie: str, df: pl.DataFrame, specie_col: str = "specie") -> pl.DataFrame:
    """Return a DataFrame with only the rows for a given species."""
    return df.filter(pl.col(specie_col) == specie)


if __name__ == "__main__":
    FORESTS = "ICP"  # "NFI" or "ICP"

    if FORESTS == "NFI":
        df = pl.read_parquet(os.path.join(clean_data_folder, "nfi_cleaned.parquet"))
        species_col = "species"
        date_col = "date"
    elif FORESTS == "ICP":
        df = pl.read_parquet(os.path.join(clean_data_folder, "icp_tree_data.parquet"))
        species_col = "specie"
        date_col = "date"
    elif FORESTS == "OLD ICP":
        df = pl.read_parquet(
            os.path.join(raw_data_folder, "ICP/icpf/03_tidy/cpf-level2_cleaned.parquet")
        )
        species_col = "specie"
        date_col = "period_end"
    else:
        raise ValueError(f"Unsupported forest: {FORESTS}")

    stable_plots = find_stable_single_species_plots(
        df,
        specie_col=species_col,
        num_obv_required=4,
        date_col=date_col,
        max_relative_change=MAX_RELATIVE_CHANGE,
    )
    print(
        f"Found {stable_plots.height} single-species plots with stems_n changing "
        f"<={MAX_RELATIVE_CHANGE:.0%}"
    )

    species_plot = _species_plot("Picea abies", stable_plots, specie_col=species_col)
    print(species_plot)
