"""Visualize how tree count and DBH values change across ICP preprocessing stages."""

import os

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from trunx.datasets.icp_level2_data import (
    _COUNTRIES_EXCLUDE,
    _ICP_FOLDER,
    _find_csv,
    _load_dictionaries,
    _make_plot_id,
    _make_tree_id,
)


def run_pipeline_stages(plot_id: str | None = None) -> list[tuple[str, pl.DataFrame]]:
    """Replay ``_load_trees``'s transformations one step at a time.

    Parameters
    ----------
    plot_id : str | None
        If given (as ``"CC.PPPP"``, e.g. ``"01.0038"``), restrict every
        stage to this plot, including the stages before ``plot_id`` exists
        as a column (filtered via the raw ``code_country``/``code_plot``).

    Returns
    -------
    list[tuple[str, pl.DataFrame]]
        ``(stage_name, dataframe)`` pairs in pipeline order.
    """
    species_df, country_df = _load_dictionaries()
    path = _find_csv(os.path.join(_ICP_FOLDER, "595_gr_*/gr_ipm.csv"))

    stages: list[tuple[str, pl.DataFrame]] = []

    df = pl.read_csv(path, separator=";", ignore_errors=True)
    if plot_id is not None:
        code_country_str, code_plot_str = plot_id.split(".")
        df = df.filter(
            (pl.col("code_country") == int(code_country_str))
            & (pl.col("code_plot") == int(code_plot_str))
        )
    stages.append(("raw", df))

    df = df.with_columns(pl.col("date_assessment").str.to_datetime().alias("date")).with_columns(
        pl.when(pl.col("date").is_null())
        .then(pl.date(pl.col("survey_year"), 7, 1).cast(pl.Datetime))
        .otherwise(pl.col("date"))
        .alias("date")
    )
    stages.append(("add date", df))

    df = df.pipe(_make_plot_id)
    stages.append(("add plot_id", df))

    df = df.pipe(_make_tree_id)
    stages.append(("add tree_id", df))

    df = df.join(species_df.select(["code_tree_species", "specie"]), on="code_tree_species")
    stages.append(("join species_df", df))

    df = df.join(country_df.select(["code_country", "country"]), on="code_country")
    stages.append(("join country_df", df))

    df = df.filter(~pl.col("country").is_in(_COUNTRIES_EXCLUDE))
    stages.append(("exclude countries", df))

    df = df.drop_nulls(subset="diameter")
    stages.append(("drop_nulls(diameter)", df))

    df = df.filter(pl.col("diameter").gt(0))
    stages.append(("diameter > 0", df))

    df = df.filter(
        pl.col("code_diameter_qc").cast(pl.Int64, strict=False).is_null()
        | ~pl.col("code_diameter_qc").cast(pl.Int64, strict=False).gt(2)
    )
    stages.append(("code_diameter_qc filter", df))

    df = df.filter(
        pl.col("code_diameter").cast(pl.Int64, strict=False).is_null()
        | ~pl.col("code_diameter").cast(pl.Int64, strict=False).is_in([7])
    )
    stages.append(("code_diameter filter", df))

    df = df.filter(
        pl.col("code_removal").cast(pl.Int64, strict=False).is_null()
        | ~pl.col("code_removal").cast(pl.Int64, strict=False).gt(10)
    )
    stages.append(("code_removal filter", df))

    df = df.sort("tree_id", "date", "survey_year").unique(subset=["tree_id", "date"], keep="last")
    stages.append(("dedup(tree_id, date)", df))

    return stages


def _stage_increase_flags(stages: list[tuple[str, pl.DataFrame]]) -> list[bool]:
    """Flag stages whose row count grew relative to the previous stage."""
    counts = [frame.height for _, frame in stages]
    return [False] + [counts[i] > counts[i - 1] for i in range(1, len(counts))]


def plot_tree_count_by_stage(stages: list[tuple[str, pl.DataFrame]]) -> Figure:
    """Plot the number of tree rows remaining after each preprocessing stage.

    Stages where the row count *increases* relative to the previous stage
    (e.g. a join fan-out adding rows instead of a filter removing them) are
    highlighted in red, since counts are expected to only shrink or stay
    flat through this pipeline.

    Parameters
    ----------
    stages : list[tuple[str, pl.DataFrame]]
        Output of :func:`run_pipeline_stages`.

    Returns
    -------
    Figure
        Bar chart of tree count per stage.
    """
    names = [name for name, _ in stages]
    counts = [frame.height for _, frame in stages]
    increased = _stage_increase_flags(stages)
    colors = ["tab:red" if flag else "tab:blue" for flag in increased]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    ax.bar(names, counts, color=colors)
    for i, count in enumerate(counts):
        ax.text(i, count, str(count), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Number of trees (rows)")
    ax.set_xlabel("Pipeline stage")
    ax.set_title("Tree count through preprocessing stages")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_dbh_by_stage(stages: list[tuple[str, pl.DataFrame]]) -> Figure:
    """Boxplot the DBH distribution remaining after each preprocessing stage.

    Stages where the row count increases relative to the previous stage
    (see :func:`plot_tree_count_by_stage`) are highlighted in red.

    Parameters
    ----------
    stages : list[tuple[str, pl.DataFrame]]
        Output of :func:`run_pipeline_stages`.

    Returns
    -------
    Figure
        Boxplot of DBH values per stage.
    """
    names = [name for name, _ in stages]
    increased = _stage_increase_flags(stages)
    values = []
    for _name, frame in stages:
        dbh_col = "dbh_cm" if "dbh_cm" in frame.columns else "diameter"
        values.append(frame[dbh_col].drop_nulls().to_numpy())

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    boxes = ax.boxplot(values, tick_labels=names, patch_artist=True)
    for patch, flag in zip(boxes["boxes"], increased, strict=True):
        patch.set_facecolor("tab:red" if flag else "tab:blue")
        patch.set_alpha(0.6)
    ax.set_ylabel("DBH (cm)")
    ax.set_xlabel("Pipeline stage")
    ax.set_title("DBH distribution through preprocessing stages")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_dbh_by_stage_and_year(stages: list[tuple[str, pl.DataFrame]]) -> Figure:
    """Boxplot DBH distribution per stage, split and colored by survey year.

    Parameters
    ----------
    stages : list[tuple[str, pl.DataFrame]]
        Output of :func:`run_pipeline_stages`.

    Returns
    -------
    Figure
        Boxplot of DBH values, grouped by stage on the x-axis with one box
        per survey year, colored by year.
    """
    _name, final = stages[-1]
    years = sorted(final["survey_year"].unique().to_list())
    cmap = plt.get_cmap("gist_rainbow")
    year_colors = {year: cmap(i / max(len(years) - 1, 1)) for i, year in enumerate(years)}

    stage_names = [name for name, _ in stages]
    n_years = len(years)
    group_width = 0.8
    box_width = group_width / n_years

    fig, ax = plt.subplots(figsize=(max(10, len(stage_names) * n_years * 0.3), 6))

    for stage_idx, (_name, frame) in enumerate(stages):
        dbh_col = "dbh_cm" if "dbh_cm" in frame.columns else "diameter"
        for year_idx, year in enumerate(years):
            values = frame.filter(pl.col("survey_year") == year)[dbh_col].drop_nulls().to_numpy()
            if len(values) == 0:
                continue
            position = stage_idx + (year_idx - (n_years - 1) / 2) * box_width
            box = ax.boxplot(
                values, positions=[position], widths=box_width * 0.9, patch_artist=True
            )
            for patch in box["boxes"]:
                patch.set_facecolor(year_colors[year])

    ax.set_xticks(range(len(stage_names)))
    ax.set_xticklabels(stage_names, rotation=45, ha="right")
    ax.set_ylabel("DBH (cm)")
    ax.set_xlabel("Pipeline stage")
    ax.set_title("DBH distribution by stage, colored by survey year")

    handles = [Line2D([0], [0], color=year_colors[year], lw=6, label=str(year)) for year in years]
    ax.legend(handles=handles, title="Year", fontsize=8, loc="upper left", ncol=min(n_years, 6))
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    PLOT_ID = "50.0018"  # "01.0038", "50.0015", "01.0041", "01.0046", "50.0008"

    stages = run_pipeline_stages(plot_id=PLOT_ID)
    for name, frame in stages:
        print(f"{name}: {frame.height} rows")

    plot_tree_count_by_stage(stages)
    plot_dbh_by_stage(stages)
    plot_dbh_by_stage_and_year(stages)
    plt.show()
