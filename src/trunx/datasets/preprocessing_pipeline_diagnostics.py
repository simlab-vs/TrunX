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
    df = df.pipe(_make_tree_id)

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


def find_plots_with_largest_yearly_tree_count_change(n_plots: int = 10) -> pl.DataFrame:
    """Find the plot x year transitions with the largest change in unique tree count.

    Uses the raw stage (before any preprocessing), across all plots.

    Parameters
    ----------
    n_plots : int
        Number of plot x year transitions (largest change first) to return.

    Returns
    -------
    pl.DataFrame
        One row per qualifying plot x year transition, with the previous and
        current survey year, their tree counts, and the change between them.
    """
    path = _find_csv(os.path.join(_ICP_FOLDER, "595_gr_*/gr_ipm.csv"))
    raw = pl.read_csv(path, separator=";", ignore_errors=True).pipe(_make_plot_id)

    counts = (
        raw.group_by("plot_id", "survey_year")
        .agg(pl.col("tree_number").n_unique().cast(pl.Int64).alias("count"))
        .sort("plot_id", "survey_year")
    )

    changes = (
        counts.with_columns(
            pl.col("survey_year").shift(1).over("plot_id").alias("prev_year"),
            pl.col("count").shift(1).over("plot_id").alias("prev_count"),
        )
        .drop_nulls("prev_year")
        .with_columns((pl.col("count") - pl.col("prev_count")).alias("count_diff"))
        .with_columns(pl.col("count_diff").abs().alias("abs_change"))
    )

    return changes.sort("abs_change", descending=True).head(n_plots)


def _stage_increase_flags(counts: list[int]) -> list[bool]:
    """Flag stages whose count grew relative to the previous stage."""
    return [False] + [counts[i] > counts[i - 1] for i in range(1, len(counts))]


def _tree_id_col(frame: pl.DataFrame) -> str:
    """Return the column that uniquely identifies a tree in a stage frame.

    ``tree_id`` only exists from the ``"join species_df"`` stage onward;
    earlier stages (``"raw"``, ``"add date"``) are identified by
    ``tree_number`` instead (unique within a single plot).
    """
    return "tree_id" if "tree_id" in frame.columns else "tree_number"


def plot_tree_count_by_stage(stages: list[tuple[str, pl.DataFrame]]) -> Figure:
    """Plot the number of unique trees remaining after each preprocessing stage.

    Stages where the unique tree count *increases* relative to the previous
    stage are highlighted in red, since counts are expected to only shrink
    or stay flat through this pipeline.

    Parameters
    ----------
    stages : list[tuple[str, pl.DataFrame]]
        Output of :func:`run_pipeline_stages`.

    Returns
    -------
    Figure
        Bar chart of unique tree count per stage.
    """
    names = [name for name, _ in stages]
    counts = [frame[_tree_id_col(frame)].n_unique() for _, frame in stages]
    increased = _stage_increase_flags(counts)
    colors = ["tab:red" if flag else "tab:blue" for flag in increased]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    ax.bar(names, counts, color=colors)
    for i, count in enumerate(counts):
        ax.text(i, count, str(count), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Number of trees (unique)")
    ax.set_xlabel("Pipeline stage")
    ax.set_title("Tree count through preprocessing stages")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_tree_count_by_stage_and_year(stages: list[tuple[str, pl.DataFrame]]) -> Figure:
    """Bar chart of tree count per stage, split and colored by survey year.

    Parameters
    ----------
    stages : list[tuple[str, pl.DataFrame]]
        Output of :func:`run_pipeline_stages`.

    Returns
    -------
    Figure
        Grouped bar chart of unique tree counts, one group per stage with
        one bar per survey year, colored by year.
    """
    _name, final = stages[-1]
    years = sorted(final["survey_year"].unique().to_list())
    cmap = plt.get_cmap("gist_rainbow")
    year_colors = {year: cmap(i / max(len(years) - 1, 1)) for i, year in enumerate(years)}

    stage_names = [name for name, _ in stages]
    n_years = len(years)
    group_width = 0.8
    bar_width = group_width / n_years

    fig, ax = plt.subplots(figsize=(max(10, len(stage_names) * n_years * 0.3), 6))

    for stage_idx, (_name, frame) in enumerate(stages):
        tree_col = _tree_id_col(frame)
        counts = frame.group_by("survey_year").agg(pl.col(tree_col).n_unique().alias("count"))
        count_by_year = dict(
            zip(counts["survey_year"].to_list(), counts["count"].to_list(), strict=True)
        )
        for year_idx, year in enumerate(years):
            count = count_by_year.get(year, 0)
            if count == 0:
                continue
            position = stage_idx + (year_idx - (n_years - 1) / 2) * bar_width
            ax.bar(position, count, width=bar_width * 0.9, color=year_colors[year])

    ax.set_xticks(range(len(stage_names)))
    ax.set_xticklabels(stage_names, rotation=45, ha="right")
    ax.set_ylabel("Number of trees (unique)")
    ax.set_xlabel("Pipeline stage")
    ax.set_title("Tree count by stage, colored by survey year")

    handles = [Line2D([0], [0], color=year_colors[year], lw=6, label=str(year)) for year in years]
    ax.legend(handles=handles, title="Year", fontsize=8, loc="upper left", ncol=min(n_years, 6))
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_raw_tree_count_by_plot_and_year(plot_ids: list[str]) -> Figure:
    """Bar chart of raw-stage tree count per plot, split and colored by survey year.

    Parameters
    ----------
    plot_ids : list[str]
        Plot ids (as ``"CC.PPPP"``) to compare.

    Returns
    -------
    Figure
        Grouped bar chart of unique tree counts, one group per plot with one
        bar per survey year, colored by year.
    """
    raw_by_plot = {plot_id: run_pipeline_stages(plot_id=plot_id)[0][1] for plot_id in plot_ids}

    years = sorted(
        {year for raw in raw_by_plot.values() for year in raw["survey_year"].unique().to_list()}
    )
    cmap = plt.get_cmap("turbo")
    year_colors = {year: cmap(i / max(len(years) - 1, 1)) for i, year in enumerate(years)}

    n_years = len(years)
    group_width = 0.6

    fig, ax = plt.subplots(figsize=(max(10, len(plot_ids) * 1.5), 6))

    for plot_idx, plot_id in enumerate(plot_ids):
        raw = raw_by_plot[plot_id]
        tree_col = _tree_id_col(raw)
        counts = (
            raw.group_by("survey_year")
            .agg(pl.col(tree_col).n_unique().alias("count"))
            .sort("survey_year")
        )
        plot_years = counts["survey_year"].to_list()
        plot_counts = counts["count"].to_list()
        n_plot_years = len(plot_years)
        plot_bar_width = group_width / n_plot_years
        for bar_idx, (year, count) in enumerate(zip(plot_years, plot_counts, strict=True)):
            position = plot_idx + (bar_idx - (n_plot_years - 1) / 2) * plot_bar_width
            ax.bar(position, count, width=plot_bar_width * 0.9, color=year_colors[year])
            ax.text(position, count, str(year), ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xticks(range(len(plot_ids)))
    ax.set_xticklabels(plot_ids, rotation=45, ha="right")
    ax.set_ylabel("Number of trees (unique)")
    ax.set_xlabel("Plot ID")
    ax.set_title("Raw tree count by plot, colored by survey year")

    handles = [Line2D([0], [0], color=year_colors[year], lw=6, label=str(year)) for year in years]
    ax.legend(handles=handles, title="Year", fontsize=8, loc="upper left", ncol=min(n_years, 6))
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("./images/raw_tree_count_by_plot_and_year.png", dpi=300)
    return fig


def plot_raw_stage_time_evolution(plot_ids: list[str]) -> Figure:
    """Plot the raw-stage tree count and DBH over survey years, one subplot per plot.

    Parameters
    ----------
    plot_ids : list[str]
        Plot ids (as ``"CC.PPPP"``) to plot.

    Returns
    -------
    Figure
        Grid of subplots, one per plot. Tree count (solid blue, left axis)
        vs survey year; each individual tree's raw DBH is scattered on the
        right axis, with a dashed line tracing the per-year mean.
    """
    n_cols = min(5, len(plot_ids))
    n_rows = (len(plot_ids) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for ax, plot_id in zip(axes, plot_ids, strict=False):
        raw = run_pipeline_stages(plot_id=plot_id)[0][1]
        tree_col = _tree_id_col(raw)
        by_year = (
            raw.group_by("survey_year")
            .agg(
                pl.col(tree_col).n_unique().alias("count"),
                pl.col("diameter").mean().alias("dbh_cm"),
            )
            .sort("survey_year")
        )
        ax.plot(by_year["survey_year"], by_year["count"], "o-", color="tab:blue")
        ax.set_title(plot_id, fontsize=9)
        ax.set_xlabel("Survey year")
        ax.set_ylabel("# trees", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.scatter(raw["survey_year"], raw["diameter"], color="tab:red", alpha=0.2, s=10)
        ax2.plot(by_year["survey_year"], by_year["dbh_cm"], "s--", color="tab:red")
        ax2.set_ylabel("DBH (cm)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

    for ax in axes[len(plot_ids) :]:
        ax.axis("off")

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
    increased = _stage_increase_flags([frame.height for _, frame in stages])
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

    fig, ax = plt.subplots(figsize=(max(8, len(stage_names) * n_years * 0.3), 6))

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
    PLOT_ID = "01.0005"  # "50.0018", "01.0038", "50.0015", "01.0041", "01.0046", "50.0008"
    stages = run_pipeline_stages(plot_id=PLOT_ID)
    for name, frame in stages:
        print(f"{name}: {frame.height} rows")

    plot_tree_count_by_stage(stages)
    plot_tree_count_by_stage_and_year(stages)
    plot_dbh_by_stage(stages)
    plot_dbh_by_stage_and_year(stages)

    print(find_plots_with_largest_yearly_tree_count_change(n_plots=10))

    PLOT_IDS = find_plots_with_largest_yearly_tree_count_change(n_plots=10)["plot_id"].to_list()
    plot_raw_tree_count_by_plot_and_year(PLOT_IDS)
    # Raw stage time evolution for multiple plots
    plot_raw_stage_time_evolution(PLOT_IDS)

    plt.show()
