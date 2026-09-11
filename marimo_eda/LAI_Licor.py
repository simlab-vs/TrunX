"""Marimo exploratory data analysis for the LI-COR LAI dataset."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    ### Dataset overview
    The LI-COR LAI dataset contains Leaf Area Index (LAI) measurements for LWF
    plots over time. `plot`, `subplot`, and `plot type` identify the sampling
    location; `date` and `season` describe when the measurement was taken;
    `mean LAI Miller` and `mean LAI Norman&Campbell` give LAI estimates from the
    two methods; `mean angle Miller` and `mean angle Norman&Campbell` give mean
    leaf angles; the corresponding `standard error` columns quantify measurement
    uncertainty; and `number of points` gives the number of measurement points.
    The plots show LAI and related measurements across dates and seasons.
    """)


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return mo, pl


@app.cell
def _():
    from pathlib import Path

    data_dir = Path("data")

    files = sorted(data_dir.iterdir())

    files  # noqa: B018
    return Path, files


@app.cell
def _(files):
    for file in files:
        print(f"{file.name:60} {file.suffix}")


@app.cell
def _(Path, pl):
    lai_path = Path("data/LAI_Licor_3_rings_all_years.xlsx")

    raw_lai = pl.read_excel(
        lai_path,
        has_header=False,
    )

    raw_lai.head()
    return (raw_lai,)


@app.cell
def _(pl, raw_lai):
    lai = raw_lai.slice(3)

    lai.columns = [
        "plot",
        "subplot",
        "date",
        "plot_type",
        "lai_miller",
        "se_miller",
        "lai_norman_campbell",
        "se_norman_campbell",
        "angle_miller",
        "se_angle_miller",
        "angle_norman_campbell",
        "se_angle_norman_campbell",
        "number_of_points",
        "season",
    ]

    # Strip whitespace from text fields.
    # Casting to Utf8 (instead of leaving mixed/other dtypes) matters here:
    # it keeps real missing values as actual nulls rather than turning them
    # into literal text, which would otherwise survive .drop_nulls() and
    # show up as a bogus option in every filter dropdown below.
    for column in ["plot", "subplot", "plot_type", "season"]:
        lai = lai.with_columns(pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars())

    # Convert "." to missing values
    lai = lai.with_columns(
        [
            pl.when(pl.col(c).cast(pl.Utf8, strict=False) == ".")
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in lai.columns
        ]
    )

    # Parse dates
    lai = lai.with_columns(
        pl.col("date").cast(pl.Utf8, strict=False).str.to_datetime(strict=False)
    )

    # Numeric variables
    numeric_columns = [
        "lai_miller",
        "se_miller",
        "lai_norman_campbell",
        "se_norman_campbell",
        "angle_miller",
        "se_angle_miller",
        "angle_norman_campbell",
        "se_angle_norman_campbell",
        "number_of_points",
    ]

    lai = lai.with_columns(
        [pl.col(column).cast(pl.Float64, strict=False) for column in numeric_columns]
    )

    lai.head()
    return (lai,)


@app.cell
def _(lai, mo, pl):
    null_percent = lai.null_count().transpose(
        include_header=True,
        header_name="Column",
        column_names=["Null percentage"],
    )

    null_percent = null_percent.with_columns(
        (pl.col("Null percentage") / lai.height * 100).round(2)
    )

    mo.ui.table(null_percent)


@app.cell
def _(lai, mo):
    plot_selector = mo.ui.dropdown(
        options=["All"] + sorted(lai["plot"].drop_nulls().unique().to_list()),
        value="All",
        label="Plot",
    )

    subplot_selector = mo.ui.dropdown(
        options=["All"] + sorted(lai["subplot"].drop_nulls().unique().to_list()),
        value="All",
        label="Subplot",
    )

    plot_type_selector = mo.ui.dropdown(
        options=["All"] + sorted(lai["plot_type"].drop_nulls().unique().to_list()),
        value="All",
        label="Plot type",
    )

    season_selector = mo.ui.dropdown(
        options=["All"] + sorted(lai["season"].drop_nulls().unique().to_list()),
        value="All",
        label="Season",
    )

    mo.hstack(
        [
            plot_selector,
            subplot_selector,
            plot_type_selector,
            season_selector,
        ]
    )
    return plot_selector, plot_type_selector, season_selector, subplot_selector


@app.cell
def _(
    lai,
    pl,
    plot_selector,
    plot_type_selector,
    season_selector,
    subplot_selector,
):
    filtered_lai = lai

    if plot_selector.value != "All":
        filtered_lai = filtered_lai.filter(pl.col("plot") == plot_selector.value)

    if subplot_selector.value != "All":
        filtered_lai = filtered_lai.filter(pl.col("subplot") == subplot_selector.value)

    if plot_type_selector.value != "All":
        filtered_lai = filtered_lai.filter(pl.col("plot_type") == plot_type_selector.value)

    if season_selector.value != "All":
        filtered_lai = filtered_lai.filter(pl.col("season") == season_selector.value)

    filtered_lai  # noqa: B018
    return (filtered_lai,)


@app.cell
def _(lai, pl, plot_selector, plot_type_selector, subplot_selector):
    # Same as filtered_lai, but deliberately ignores the season filter so
    # the seasonal-comparison chart below always has all seasons to
    # compare, no matter what the top season dropdown is set to.
    lai_for_season = lai

    if plot_selector.value != "All":
        lai_for_season = lai_for_season.filter(pl.col("plot") == plot_selector.value)

    if subplot_selector.value != "All":
        lai_for_season = lai_for_season.filter(pl.col("subplot") == subplot_selector.value)

    if plot_type_selector.value != "All":
        lai_for_season = lai_for_season.filter(pl.col("plot_type") == plot_type_selector.value)

    lai_for_season  # noqa: B018
    return (lai_for_season,)


@app.cell
def _(mo):
    measurement_options = {
        "LAI — Miller": "lai_miller",
        "LAI — Norman & Campbell": "lai_norman_campbell",
        "Mean angle — Miller": "angle_miller",
        "Mean angle — Norman & Campbell": "angle_norman_campbell",
    }

    measurement_selector = mo.ui.dropdown(
        options=list(measurement_options.keys()),
        value="LAI — Miller",
        label="Measurement",
    )

    measurement_selector  # noqa: B018
    return measurement_options, measurement_selector


@app.cell
def _(measurement_options, measurement_selector):
    selected_measurement = measurement_options[measurement_selector.value]

    selected_measurement  # noqa: B018
    return (selected_measurement,)


@app.cell
def _():
    import plotly.express as px
    import plotly.graph_objects as go

    return go, px


@app.cell
def _(filtered_lai, go, measurement_selector, pl, selected_measurement):
    plot_data = filtered_lai.select(["date", selected_measurement]).sort("date")

    # ============================================================
    # SETTINGS
    # ============================================================

    # Gaps longer than this are shown as dashed lines
    max_gap_days = 365 * 2

    # A single shared color for the trend line + markers. Previously every
    # connecting segment was its own Scatter trace, so Plotly's default
    # color cycling painted the line a different color per segment
    # (a "rainbow" line). Building one solid-segment trace and one
    # dashed-segment trace with an explicit color fixes that.
    line_color = "#3366CC"

    # ============================================================
    # FIGURE
    # ============================================================

    fig = go.Figure()

    # ============================================================
    # 1. MEASURED OBSERVATIONS
    # ============================================================

    measured = plot_data.drop_nulls(subset=[selected_measurement]).sort("date").to_dicts()

    # ============================================================
    # 2. CONNECTING LINES (solid vs. dashed segments)
    # ============================================================

    solid_x, solid_y = [], []
    dashed_x, dashed_y = [], []

    for i in range(len(measured) - 1):
        current = measured[i]
        next_row = measured[i + 1]

        gap_days = (next_row["date"] - current["date"]).days

        target_x = solid_x if gap_days <= max_gap_days else dashed_x
        target_y = solid_y if gap_days <= max_gap_days else dashed_y

        # None breaks the line between unrelated segments while still
        # letting them share a single trace (and a single color/legend
        # entry).
        target_x += [current["date"], next_row["date"], None]
        target_y += [
            current[selected_measurement],
            next_row[selected_measurement],
            None,
        ]

    if solid_x:
        fig.add_trace(
            go.Scatter(
                x=solid_x,
                y=solid_y,
                mode="lines",
                line={"width": 2, "color": line_color},
                name="Trend",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if dashed_x:
        fig.add_trace(
            go.Scatter(
                x=dashed_x,
                y=dashed_y,
                mode="lines",
                line={"width": 2, "color": line_color, "dash": "dash"},
                name="Long data gap",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    # ============================================================
    # 3. MARKERS (drawn last so they sit on top of the lines)
    # ============================================================

    fig.add_trace(
        go.Scatter(
            x=[row["date"] for row in measured],
            y=[row[selected_measurement] for row in measured],
            mode="markers",
            name="Measured",
            marker={"size": 8, "color": line_color},
            # Full date shown when hovering
            hovertemplate=(
                "<b>Date:</b> %{x|%d %B %Y}"
                "<br>"
                f"<b>{measurement_selector.value}:</b> "
                "%{y:.2f}"
                "<extra></extra>"
            ),
            showlegend=True,
        )
    )

    # ============================================================
    # 4. HIGHLIGHT NaN DATES
    # ============================================================

    missing_dates = (
        plot_data.filter(pl.col(selected_measurement).is_null())
        .select("date")
        .drop_nulls()
        .unique()
        .sort("date")
        .to_series()
        .to_list()
    )

    for date in missing_dates:
        # add_vline draws a shape, not a trace — shapes don't reliably
        # accept showlegend across Plotly versions, so it's left off here.
        # The dummy trace below (step 5) provides the legend entry instead.
        fig.add_vline(
            x=date,
            line_width=2,
            line_dash="dot",
            line_color="red",
        )

    # ============================================================
    # 5. LEGEND ENTRY FOR MISSING OBSERVATIONS
    # ============================================================

    if len(missing_dates) > 0:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={
                    "width": 2,
                    "dash": "dot",
                    "color": "red",
                },
                name="Missing observation",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    # ============================================================
    # 6. LAYOUT
    # ============================================================

    fig.update_layout(
        title=f"{measurement_selector.value} over time",
        xaxis_title="Date",
        yaxis_title=measurement_selector.value,
        # Important: inspect individual observations
        hovermode="closest",
        template="plotly_white",
    )

    fig  # noqa: B018


@app.cell
def _(lai, measurement_options, mo, pl):
    # --------------------------------------------------------
    # Compare any number of LWF plots (multiplot)
    # --------------------------------------------------------

    multiplot_options = sorted(lai["plot"].drop_nulls().cast(pl.Utf8).unique().to_list())

    multiplot_selector = mo.ui.multiselect(
        options=multiplot_options,
        value=multiplot_options[:2] if len(multiplot_options) > 1 else multiplot_options,
        label="Plots to compare",
    )

    multiplot_measurement_selector = mo.ui.dropdown(
        options=list(measurement_options.keys()),
        value="LAI — Miller",
        label="Measurement",
    )

    mo.vstack(
        [
            mo.md("## Compare LWF plots"),
            mo.hstack([multiplot_selector, multiplot_measurement_selector]),
        ]
    )
    return multiplot_measurement_selector, multiplot_selector


@app.cell
def _(
    go,
    lai,
    measurement_options,
    multiplot_measurement_selector,
    multiplot_selector,
    pl,
    px,
):
    multiplot_selected_measurement = measurement_options[multiplot_measurement_selector.value]

    multiplot_palette = px.colors.qualitative.Set2

    fig_multiplot = go.Figure()

    for idx, plot_id in enumerate(multiplot_selector.value):
        subset = (
            lai.filter(pl.col("plot").cast(pl.Utf8) == plot_id)
            .select(["date", multiplot_selected_measurement])
            .drop_nulls()
            .sort("date")
        )

        color = multiplot_palette[idx % len(multiplot_palette)]

        fig_multiplot.add_trace(
            go.Scatter(
                x=subset["date"].to_list(),
                y=subset[multiplot_selected_measurement].to_list(),
                mode="lines+markers",
                name=plot_id,
                marker={"size": 7, "color": color},
                line={"width": 2, "color": color},
                hovertemplate=(
                    f"<b>Plot:</b> {plot_id}"
                    "<br><b>Date:</b> %{x|%d %B %Y}"
                    f"<br><b>{multiplot_measurement_selector.value}:</b> "
                    "%{y:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    if not multiplot_selector.value:
        fig_multiplot.add_annotation(
            text="Select one or more plots above to compare them",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            font={"size": 14, "color": "gray"},
        )

    fig_multiplot.update_layout(
        title=f"{multiplot_measurement_selector.value} across selected plots",
        xaxis_title="Date",
        yaxis_title=multiplot_measurement_selector.value,
        hovermode="x unified",
        legend_title="LWF plot",
        template="plotly_white",
    )

    fig_multiplot  # noqa: B018


@app.cell
def _(
    go,
    lai_for_season,
    measurement_selector,
    mo,
    pl,
    px,
    selected_measurement,
):
    # --------------------------------------------------------
    # Compare the currently selected measurement across seasons
    # --------------------------------------------------------

    season_data = lai_for_season.drop_nulls(subset=[selected_measurement, "season"])

    seasons = sorted(season_data["season"].drop_nulls().unique().to_list())
    season_palette = px.colors.qualitative.Set2

    fig_season = go.Figure()

    for s_idx, szn in enumerate(seasons):
        values = (
            season_data.filter(pl.col("season") == szn)
            .select(selected_measurement)
            .to_series()
            .to_list()
        )
        s_color = season_palette[s_idx % len(season_palette)]

        fig_season.add_trace(
            go.Box(
                y=values,
                name=szn,
                marker={"color": s_color, "size": 5, "opacity": 0.6},
                line={"color": s_color, "width": 1.5},
                fillcolor=s_color,
                opacity=0.55,
                boxmean=True,
                boxpoints="all",
                jitter=0.45,
                pointpos=-1.8,
            )
        )

    if not seasons:
        fig_season.add_annotation(
            text="No data available for the current filters",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            font={"size": 14, "color": "gray"},
        )

    fig_season.update_layout(
        title=f"{measurement_selector.value} by season",
        xaxis_title="Season",
        yaxis_title=measurement_selector.value,
        template="plotly_white",
        showlegend=False,
        boxgap=0.35,
        boxgroupgap=0.2,
    )

    mo.vstack(
        [
            mo.md("## Seasonal comparison"),
            fig_season,
        ]
    )


@app.cell
def _(mo):
    mo.md("""
    ### Key findings

    - LAI Miller values range from about 1 to 8 across measurements and years.
    - Summer measurements generally show higher LAI than winter.
    - Spring observations are more variable.
    - Long periods without measurements occur, particularly between survey years.
    """)


if __name__ == "__main__":
    app.run()
