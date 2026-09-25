"""Marimo EDA for plot-level LWF foliage dry-weight measurements."""

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
    foliage_path = Path("data/lwf_foliage_dw100_plot_2026-07-30.csv")

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


@app.cell
def _(foliage):
    foliage  # noqa: B018


@app.cell
def _(foliage):
    foliage_1 = foliage.sort(["plot_id", "survey_year"])
    foliage_1  # noqa: B018


@app.cell
def _(Path, pl):
    foliage2_path = Path("data/lwf_foliage_dw100_i_2026-07-30.csv")

    foliage2 = pl.read_csv(
        foliage2_path,
        separator=";",
        schema_overrides={
            "sample_id": pl.String,
        },
        null_values="NA",
    )
    return (foliage2,)


@app.cell
def _(foliage2, pl):
    foliage3 = foliage2.group_by(
        ["plot_id", "survey_date", "leaf_type", "species", "leaf_age_class"],
        maintain_order=True,
    ).agg(pl.col("gew100").mean())

    foliage3 = foliage3.sort(["plot_id", "survey_date", "species"])
    foliage3  # noqa: B018


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
def _(foliage_dated):
    foliage_dated["plot_id"].unique()


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


@app.cell
def _(filtered_foliage):
    import plotly.graph_objects as _go

    # Keep the three data-quality cases separate:
    # 1. valid survey date + valid gew100 -> normal measurements
    # 2. valid survey date + missing gew100 -> missing-value markers
    # 3. missing survey date (NaT) -> reported separately because it cannot
    #    be placed on a time axis.
    valid_plot = (
        filtered_foliage.select(["survey_date", "gew100"])
        .drop_nulls(subset=["survey_date", "gew100"])
        .sort("survey_date")
    )

    missing_gew100 = (
        filtered_foliage.select(["survey_date", "gew100"])
        .filter(
            filtered_foliage["survey_date"].is_not_null() & filtered_foliage["gew100"].is_null()
        )
        .sort("survey_date")
    )

    missing_survey_date = filtered_foliage.filter(filtered_foliage["survey_date"].is_null())

    _fig = _go.Figure()

    # Normal measurements.
    _fig.add_trace(
        _go.Scatter(
            x=valid_plot["survey_date"].to_list(),
            y=valid_plot["gew100"].to_list(),
            mode="markers",
            name="Individual measurement",
            marker={"size": 7},
            hovertemplate=(
                "<b>Date:</b> %{x|%d %B %Y}<br><b>gew100:</b> %{y:.2f} g<extra></extra>"
            ),
        )
    )

    # Missing gew100 values can be located on the time axis because their
    # survey date is known. Place an X below the observed range; this is a
    # visual marker only and is explicitly labelled as missing.
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

        _fig.add_trace(
            _go.Scatter(
                x=missing_gew100["survey_date"].to_list(),
                y=[missing_marker_y] * missing_gew100.height,
                mode="markers",
                name="Missing gew100",
                marker={"size": 10, "symbol": "x"},
                customdata=["Missing gew100"] * missing_gew100.height,
                hovertemplate=("<b>Date:</b> %{x|%d %B %Y}<br><b>gew100:</b> NaN<extra></extra>"),
            )
        )

    # Missing survey dates cannot be plotted on a datetime x-axis

    _fig.add_annotation(
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

    _fig.update_layout(
        title="Foliage dry weight over time",
        xaxis_title="Survey date",
        yaxis_title="gew100 (g)",
        hovermode="closest",
    )

    _fig  # noqa: B018


@app.cell
def _(foliage_dated, mo, pl):
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
            mo.md("### Compare LWF plots"),
            comparison_plot_selector,
        ]
    )
    return (comparison_plot_selector,)


@app.cell
def _(comparison_plot_selector, foliage_dated, pl):
    comparison_data_by_plot = {
        comparison_plot_id: (
            foliage_dated.filter(pl.col("plot_id").cast(pl.Utf8) == comparison_plot_id).sort(
                "survey_date"
            )
        )
        for comparison_plot_id in comparison_plot_selector.value
    }
    return (comparison_data_by_plot,)


@app.cell
def _(comparison_data_by_plot, comparison_plot_selector, mo):
    import plotly.graph_objects as go

    fig = go.Figure()

    for comparison_plot_id, comparison_plot_data in comparison_data_by_plot.items():
        valid_comparison_data = comparison_plot_data.drop_nulls(subset=["survey_date", "gew100"])

        fig.add_trace(
            go.Scatter(
                x=valid_comparison_data["survey_date"].to_list(),
                y=valid_comparison_data["gew100"].to_list(),
                mode="markers",
                name=comparison_plot_id,
                marker={"size": 7},
                hovertemplate=(
                    f"<b>{comparison_plot_id}</b>"
                    "<br>Date: %{x|%d %B %Y}"
                    "<br>gew100: %{y:.2f} g"
                    "<extra></extra>"
                ),
            )
        )

    comparison_sites_label = (
        ", ".join(comparison_plot_selector.value)
        if comparison_plot_selector.value
        else "no sites selected"
    )

    fig.update_layout(
        title=f"Foliage dry weight comparison: {comparison_sites_label}",
        xaxis_title="Survey date",
        yaxis_title="gew100 (g)",
        hovermode="closest",
    )

    mo.vstack(
        [
            mo.md(f"### {comparison_sites_label}"),
            fig,
        ]
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Key findings
    1. Much of the Dates column is Empty.

    2. We can simply use individual dry weight dataset and completely ignore this.
    """)


if __name__ == "__main__":
    app.run()
