"""Marimo EDA for individual LWF foliage dry-weight measurements."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    return Path, mo, pl


@app.cell
def _(Path, pl):
    foliage_path = Path("data/lwf_foliage_dw100_i_2026-07-30.csv")

    foliage = pl.read_csv(
        foliage_path,
        separator=";",
        schema_overrides={
            "sample_id": pl.String,
        },
        null_values="NA",
    )
    return (foliage,)


@app.cell
def _(foliage, mo, pl):
    null_percent = foliage.null_count().transpose(
        include_header=True,
        header_name="Column",
        column_names=["Null percentage"],
    )

    null_percent = null_percent.with_columns(
        (pl.col("Null percentage") / foliage.height * 100).round(2)
    )

    mo.ui.table(null_percent)
    return


@app.cell
def _():
    return


@app.cell
def _(foliage, pl):
    foliage.group_by(
        ["plot_id", "survey_date", "leaf_type", "species", "leaf_age_class"],
        maintain_order=True,
    ).agg(pl.col("gew100").mean())
    return


@app.cell
def _(foliage, pl):
    foliage_dated = foliage.with_columns(
        pl.col("survey_date").cast(pl.Utf8, strict=False).str.to_datetime(strict=False)
    )
    return (foliage_dated,)


@app.cell
def _(foliage_dated, mo):
    plot_selector = mo.ui.dropdown(
        options=["All"] + sorted(foliage_dated["plot_id"].drop_nulls().unique().to_list()),
        value="All",
        label="LWF site",
    )

    species_selector = mo.ui.dropdown(
        options=["All"] + sorted(foliage_dated["species"].drop_nulls().unique().to_list()),
        value="All",
        label="Species",
    )

    leaf_type_selector = mo.ui.dropdown(
        options=["All"] + sorted(foliage_dated["leaf_type"].drop_nulls().unique().to_list()),
        value="All",
        label="Leaf type",
    )

    age_selector = mo.ui.dropdown(
        options=["All"] + sorted(foliage_dated["leaf_age_class"].drop_nulls().unique().to_list()),
        value="All",
        label="Leaf age class",
    )

    mo.vstack(
        [
            mo.hstack(
                [
                    plot_selector,
                    species_selector,
                ]
            ),
            mo.hstack(
                [
                    leaf_type_selector,
                    age_selector,
                ]
            ),
        ]
    )
    return age_selector, leaf_type_selector, plot_selector, species_selector


@app.cell
def _(
    age_selector,
    foliage_dated,
    leaf_type_selector,
    pl,
    plot_selector,
    species_selector,
):
    filtered_foliage = foliage_dated

    if plot_selector.value != "All":
        filtered_foliage = filtered_foliage.filter(pl.col("plot_id") == plot_selector.value)

    if species_selector.value != "All":
        filtered_foliage = filtered_foliage.filter(pl.col("species") == species_selector.value)

    if leaf_type_selector.value != "All":
        filtered_foliage = filtered_foliage.filter(pl.col("leaf_type") == leaf_type_selector.value)

    if age_selector.value != "All":
        filtered_foliage = filtered_foliage.filter(pl.col("leaf_age_class") == age_selector.value)

    filtered_foliage  # noqa: B018
    return (filtered_foliage,)


@app.cell
def _(filtered_foliage):
    filtered_foliage["gew100"].describe()
    return


@app.cell
def _(filtered_foliage):
    filtered_foliage["gew100"].is_null().sum()
    return


@app.cell
def _(filtered_foliage):
    import plotly.graph_objects as go

    # Keep the three data-quality cases separate:
    # 1. valid survey date + valid gew100
    # 2. valid survey date + missing gew100
    # 3. missing survey date (NaT)

    valid_plot = (
        filtered_foliage.select(["survey_date", "gew100", "sample_id"])
        .drop_nulls(subset=["survey_date", "gew100"])
        .sort("survey_date")
    )

    missing_gew100 = (
        filtered_foliage.select(["survey_date", "gew100", "sample_id"])
        .filter(
            filtered_foliage["survey_date"].is_not_null() & filtered_foliage["gew100"].is_null()
        )
        .sort("survey_date")
    )

    missing_survey_date = filtered_foliage.filter(filtered_foliage["survey_date"].is_null())

    fig = go.Figure()

    # ============================================================
    # 1. NORMAL MEASUREMENTS
    # ============================================================

    fig.add_trace(
        go.Scatter(
            x=valid_plot["survey_date"].to_list(),
            y=valid_plot["gew100"].to_list(),
            mode="markers",
            name="Individual measurement",
            marker={"size": 7},
            customdata=valid_plot["sample_id"].to_list(),
            hovertemplate=(
                "<b>Date:</b> %{x|%d %B %Y}"
                "<br><b>Sample ID:</b> %{customdata}"
                "<br><b>gew100:</b> %{y:.2f} g"
                "<extra></extra>"
            ),
        )
    )

    # ============================================================
    # 2. MISSING gew100, BUT DATE IS KNOWN
    # ============================================================

    if not missing_gew100.is_empty():
        if not valid_plot.is_empty():
            y_min = valid_plot["gew100"].min()
            y_max = valid_plot["gew100"].max()
            y_range = y_max - y_min

            if y_range == 0:
                y_range = max(abs(y_min) * 0.1, 1.0)

            missing_marker_y = y_min - 0.08 * y_range

        else:
            missing_marker_y = 0

        fig.add_trace(
            go.Scatter(
                x=missing_gew100["survey_date"].to_list(),
                y=[missing_marker_y] * missing_gew100.height,
                mode="markers",
                name="Missing gew100",
                marker={
                    "size": 10,
                    "symbol": "x",
                },
                customdata=missing_gew100["sample_id"].to_list(),
                hovertemplate=(
                    "<b>Date:</b> %{x|%d %B %Y}"
                    "<br><b>Sample ID:</b> %{customdata}"
                    "<br><b>gew100:</b> NaN"
                    "<extra></extra>"
                ),
            )
        )

    # ============================================================
    # 3. MISSING SURVEY DATE (NaT)
    # ============================================================

    # These observations cannot be positioned on a datetime x-axis,
    # so report their count rather than silently dropping them.

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1,
        y=1.08,
        xanchor="right",
        yanchor="bottom",
        text=(
            f"Missing survey date (NaT): {missing_survey_date.height}"
            f" | Missing gew100: {missing_gew100.height}"
        ),
        showarrow=False,
    )

    fig.update_layout(
        title="Individual foliage dry weight over time",
        xaxis_title="Survey date",
        yaxis_title="gew100 (g)",
        hovermode="closest",
    )

    fig  # noqa: B018
    return (go,)


@app.cell
def _(foliage_dated, pl):
    foliage_dated.group_by(
        [
            "survey_date",
            "species",
            "leaf_type",
            "leaf_age_class",
        ],
        maintain_order=True,
    ).agg(
        n_samples=pl.col("sample_id").n_unique(),
        mean_gew100=pl.col("gew100").mean(),
        median_gew100=pl.col("gew100").median(),
        std_gew100=pl.col("gew100").std(),
    ).sort("survey_date")
    return


@app.cell
def _(foliage_dated, mo, pl):
    # --------------------------------------------------------
    # Compare LWF plots
    # --------------------------------------------------------

    comparison_plot_options = sorted(
        foliage_dated["plot_id"].drop_nulls().cast(pl.Utf8).unique().to_list()
    )

    comparison_plot_selector = mo.ui.multiselect(
        options=comparison_plot_options,
        value=comparison_plot_options[:2],
        label="LWF sites to compare",
    )

    mo.vstack(
        [
            mo.md("## Compare LWF plots"),
            comparison_plot_selector,
        ]
    )
    return (comparison_plot_selector,)


@app.cell
def _(comparison_plot_selector, foliage_dated, go, pl):
    import plotly.graph_objects as goo

    # One line per selected LWF site, so any number of sites can be
    # compared at once rather than being limited to a fixed pair.

    comparison_selected_plots = comparison_plot_selector.value

    comparison_figure = goo.Figure()

    for comparison_plot_id in comparison_selected_plots:
        comparison_plot_data = (
            foliage_dated.filter(pl.col("plot_id").cast(pl.Utf8) == comparison_plot_id)
            .drop_nulls(subset=["survey_date", "gew100"])
            .group_by("survey_date", maintain_order=True)
            .agg(pl.col("gew100").mean())
            .sort("survey_date")
        )

        comparison_figure.add_trace(
            go.Scatter(
                x=comparison_plot_data["survey_date"].to_list(),
                y=comparison_plot_data["gew100"].to_list(),
                mode="lines+markers",
                name=comparison_plot_id,
                marker={"size": 6},
                hovertemplate=(
                    f"<b>Plot:</b> {comparison_plot_id}"
                    "<br><b>Date:</b> %{x|%d %B %Y}"
                    "<br><b>Mean gew100:</b> %{y:.2f} g"
                    "<extra></extra>"
                ),
            )
        )

    comparison_figure.update_layout(
        title="Foliage dry weight comparison across LWF plots",
        xaxis_title="Survey date",
        yaxis_title="Mean gew100 (g)",
        hovermode="x unified",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Key findings

    - Mean foliage dry weight differs substantially between LWF plots.
    - Samples averaged according to site, date and leaf age type so we can ignore the pooled data.
    - CHI and JUS generally show the highest mean values across the measurement years.
    - BET has highly fluctuating values
    - The plot-level measurements are irregular over time, with several gaps between survey years.
    """)
    return


if __name__ == "__main__":
    app.run()
