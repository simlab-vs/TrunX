# ruff: noqa: E501
"""Marimo EDA for monthly litterfall measurements at the Davos site."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    ### Dataset overview
    The monthly litterfall dataset contains litterfall measurements from the Davos (DAV) site for 2018–2025. `Network` identifies the data network (ICOS), `LWF_plot` gives the site name, and `LWF_plot_code` gives its LWF site code. `Year` and `Month` identify the sampling month, while `Start_date` and `End_date` define the corresponding sampling period. `Code_fraction` is the numeric code identifying the litterfall component, and `Text_Fraction` gives its description: spruce needles, spruce cones and seeds, twigs/branches/bark/wood, fine litter/flowers/budscales/lichens, or Total. `Weight` is the interpolated monthly litterfall weight in kg/ha at the 65°C reference temperature. The plots show temporal, seasonal, annual, and component-level variation in litterfall.
    """)


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import plotly.graph_objects as go
    import polars as pl

    return Path, go, mo, pl


@app.cell
def _(Path, pl):
    # ------------------------------------------------------------
    # Load source data
    # ------------------------------------------------------------

    litterfall_path = Path("data/monthly_litterfall_Davos_ICOS_2026-07-30.csv")

    litterfall = pl.read_csv(
        litterfall_path,
        separator=";",
        skip_rows=2,
        encoding="windows-1252",
        null_values=".",
    )
    return (litterfall,)


@app.cell
def _(litterfall, mo, pl):
    row_count = max(len(litterfall), 1)
    null_percent = pl.DataFrame(
        {
            "Column": litterfall.columns,
            "Null percentage": [
                round(
                    100 * litterfall[column].is_null().sum() / row_count,
                    2,
                )
                for column in litterfall.columns
            ],
        }
    )

    mo.ui.table(null_percent)


@app.cell
def _(litterfall, pl):
    # ------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------

    cleaned_litterfall = litterfall.with_columns(
        pl.col("LWF_plot_code").cast(pl.String).str.strip_chars(),
        pl.col("LWF_plot").cast(pl.String).str.strip_chars(),
        pl.col("Network").cast(pl.String).str.strip_chars(),
        pl.col("Text_Fraction").cast(pl.String).str.strip_chars(),
        pl.col("Year").cast(pl.Float64, strict=False),
        pl.col("Month").cast(pl.Float64, strict=False),
        pl.col("Code_fraction").cast(pl.Float64, strict=False),
        pl.col("Weight").cast(pl.Float64, strict=False),
        pl.col("Start_date").str.strptime(
            pl.Date,
            "%d/%m/%Y",
            strict=False,
        ),
        pl.col("End_date").str.strptime(
            pl.Date,
            "%d/%m/%Y",
            strict=False,
        ),
    ).with_columns(
        (pl.col("Start_date") + (pl.col("End_date") - pl.col("Start_date")) / 2).alias(
            "period_midpoint"
        ),
        (pl.col("End_date") - pl.col("Start_date")).dt.total_days().alias("duration_days"),
        pl.col("Code_fraction").alias("fraction_order"),
    )
    return (cleaned_litterfall,)


@app.cell
def _(cleaned_litterfall, mo):
    # ------------------------------------------------------------
    # Dataset overview
    # ------------------------------------------------------------

    total_rows = len(cleaned_litterfall)
    total_periods = cleaned_litterfall["period_midpoint"].drop_nulls().n_unique()
    total_fractions = cleaned_litterfall["Text_Fraction"].drop_nulls().n_unique()
    valid_weights = cleaned_litterfall["Weight"].is_not_null().sum()
    missing_weights = cleaned_litterfall["Weight"].is_null().sum()

    mo.vstack(
        [
            mo.md(
                """
                # Davos ICOS — Monthly Litterfall

                This notebook explores the **Davos (`DAV`) monthly
                litterfall dataset** associated with the ICOS network.

                The source contains five litterfall fractions and a `Total`
                category. `Weight` is the reported litterfall weight for the
                corresponding monthly period. Missing values in the source
                are represented by `.` and are treated as missing values.
                """
            ),
            mo.hstack(
                [
                    mo.stat(
                        label="LWF site",
                        value="DAV",
                    ),
                    mo.stat(
                        label="Network",
                        value="ICOS",
                    ),
                    mo.stat(
                        label="Rows",
                        value=f"{total_rows:,}",
                    ),
                    mo.stat(
                        label="Periods",
                        value=f"{total_periods:,}",
                    ),
                    mo.stat(
                        label="Fractions",
                        value=f"{total_fractions:,}",
                    ),
                    mo.stat(
                        label="Missing weights",
                        value=f"{missing_weights:,}",
                    ),
                ]
            ),
            mo.md(
                f"""
                **Coverage:** {cleaned_litterfall["Start_date"].min().strftime("%d %B %Y")}
                to {cleaned_litterfall["End_date"].max().strftime("%d %B %Y")}

                **Valid weights:** {valid_weights:,}
                """
            ),
        ]
    )


@app.cell
def _(cleaned_litterfall, mo, pl):
    # ------------------------------------------------------------
    # Main interactive filters
    # ------------------------------------------------------------

    fraction_options = sorted(
        cleaned_litterfall["Text_Fraction"].drop_nulls().cast(pl.String).unique().to_list()
    )

    year_options = sorted(
        cleaned_litterfall["Year"].drop_nulls().cast(pl.Int64).unique().to_list()
    )

    month_options = sorted(
        cleaned_litterfall["Month"].drop_nulls().cast(pl.Int64).unique().to_list()
    )

    fraction_selector = mo.ui.dropdown(
        options=["All"] + fraction_options,
        value="Total",
        label="Litter fraction",
    )

    year_selector = mo.ui.dropdown(
        options=["All"] + year_options,
        value="All",
        label="Year",
    )

    month_selector = mo.ui.dropdown(
        options=["All"] + month_options,
        value="All",
        label="Month",
    )

    mo.vstack(
        [
            mo.md("## Explore the data"),
            mo.hstack(
                [
                    fraction_selector,
                    year_selector,
                    month_selector,
                ]
            ),
        ]
    )
    return fraction_selector, month_selector, year_selector


@app.cell
def _(
    cleaned_litterfall,
    fraction_selector,
    month_selector,
    pl,
    year_selector,
):
    filtered_litterfall = cleaned_litterfall

    if fraction_selector.value != "All":
        filtered_litterfall = filtered_litterfall.filter(
            pl.col("Text_Fraction").cast(pl.String) == fraction_selector.value
        )

    if year_selector.value != "All":
        filtered_litterfall = filtered_litterfall.filter(
            pl.col("Year") == int(year_selector.value)
        )

    if month_selector.value != "All":
        filtered_litterfall = filtered_litterfall.filter(
            pl.col("Month") == int(month_selector.value)
        )

    filtered_litterfall = filtered_litterfall.sort(["Start_date", "End_date"])

    filtered_litterfall  # noqa: B018
    return (filtered_litterfall,)


@app.cell
def _(filtered_litterfall, mo, pl):
    # ------------------------------------------------------------
    # Data quality
    # ------------------------------------------------------------

    overview = pl.DataFrame(
        {
            "Quantity": [
                "Total observations",
                "Valid weights",
                "Missing weights (NaN)",
                "Missing start dates (NaT)",
                "Missing end dates (NaT)",
            ],
            "Count": [
                len(filtered_litterfall),
                filtered_litterfall["Weight"].is_not_null().sum(),
                filtered_litterfall["Weight"].is_null().sum(),
                filtered_litterfall["Start_date"].is_null().sum(),
                filtered_litterfall["End_date"].is_null().sum(),
            ],
        }
    )

    mo.vstack(
        [
            mo.md("## Data quality"),
            overview,
        ]
    )


@app.cell
def _(filtered_litterfall, pl):
    # Missingness across the complete filtered table.
    missingness_table = (
        pl.DataFrame(
            {
                "Variable": filtered_litterfall.columns,
                "Missing count": [
                    filtered_litterfall[column].is_null().sum()
                    for column in filtered_litterfall.columns
                ],
            }
        )
        .with_columns(
            (100 * pl.col("Missing count") / max(len(filtered_litterfall), 1)).alias("Missing %")
        )
        .sort(
            ["Missing count", "Variable"],
            descending=[True, False],
        )
    )

    missingness_table  # noqa: B018


@app.cell
def _(filtered_litterfall, go, mo, pl):
    # ------------------------------------------------------------
    # Main time-series view
    # ------------------------------------------------------------

    valid_series = filtered_litterfall.drop_nulls(["period_midpoint", "Weight"]).sort(
        "period_midpoint"
    )

    missing_series = filtered_litterfall.filter(
        pl.col("period_midpoint").is_not_null() & pl.col("Weight").is_null()
    ).sort("period_midpoint")

    figure = go.Figure()

    if len(valid_series) > 0:
        figure.add_trace(
            go.Scatter(
                x=valid_series["period_midpoint"],
                y=valid_series["Weight"],
                mode="lines+markers",
                name="Weight",
                marker={"size": 7},
                customdata=valid_series[["Start_date", "End_date", "Text_Fraction"]],
                hovertemplate=(
                    "<b>Period:</b> "
                    "%{customdata[0]|%d %B %Y}"
                    " → "
                    "%{customdata[1]|%d %B %Y}"
                    "<br><b>Fraction:</b> %{customdata[2]}"
                    "<br><b>Weight:</b> %{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    if len(missing_series) > 0:
        if len(valid_series) > 0:
            min_weight = valid_series["Weight"].min()
            max_weight = valid_series["Weight"].max()
            weight_range = max_weight - min_weight

            if weight_range == 0:
                weight_range = max(abs(min_weight) * 0.1, 1.0)

            missing_y = min_weight - 0.08 * weight_range
        else:
            missing_y = 0

        figure.add_trace(
            go.Scatter(
                x=missing_series["period_midpoint"],
                y=[missing_y] * len(missing_series),
                mode="markers",
                name="Missing weight",
                marker={
                    "size": 10,
                    "symbol": "x",
                },
                customdata=missing_series[["Start_date", "End_date", "Text_Fraction"]],
                hovertemplate=(
                    "<b>Period:</b> "
                    "%{customdata[0]|%d %B %Y}"
                    " → "
                    "%{customdata[1]|%d %B %Y}"
                    "<br><b>Fraction:</b> %{customdata[2]}"
                    "<br><b>Weight:</b> NaN"
                    "<extra></extra>"
                ),
            )
        )

    selected_fraction = (
        "all fractions"
        if filtered_litterfall["Text_Fraction"].n_unique() != 1
        else filtered_litterfall["Text_Fraction"][0]
    )

    figure.update_layout(
        title=f"Litterfall weight over time — {selected_fraction}",
        xaxis_title="Sampling-period midpoint",
        yaxis_title="Weight",
        hovermode="closest",
    )

    mo.vstack(
        [
            figure,
            mo.md(
                f"""
                **Valid observations plotted:** {len(valid_series):,}  
                **Missing weights marked as X:** {len(missing_series):,}
                """
            ),
        ]
    )


@app.cell
def _(cleaned_litterfall, go, pl):
    # ------------------------------------------------------------
    # Seasonal pattern
    # ------------------------------------------------------------

    seasonal_data = cleaned_litterfall.drop_nulls(
        ["Month", "Weight", "Text_Fraction", "Start_date", "End_date"]
    )

    # Pivot so that each row is a single sampling period (identified by
    # its Start_date/End_date, i.e. the same site/sampling occasion),
    # with one column per litterfall fraction, Total included. This
    # keeps each fraction's observation tied to the specific period it
    # came from, instead of letting every fraction be averaged over
    # whichever periods happen to have data for it.
    seasonal_wide = seasonal_data.pivot(
        on="Text_Fraction",
        index=["Start_date", "End_date", "Month"],
        values="Weight",
        aggregate_function="mean",
    )

    # Keep only sampling periods where every fraction, including
    # Total, has a reported weight. This guarantees that Total and
    # its components are averaged over an identical set of periods,
    # so Total is directly comparable to the components on the plot.
    seasonal_complete = seasonal_wide.drop_nulls()

    # Average each fraction (and Total) by calendar month, using that
    # same shared set of complete periods for every line.
    fraction_columns = [
        column
        for column in seasonal_complete.columns
        if column not in {"Start_date", "End_date", "Month"}
    ]
    seasonal_monthly_means = (
        seasonal_complete.group_by("Month")
        .agg([pl.col(column).mean().alias(column) for column in fraction_columns])
        .sort("Month")
    )

    seasonal_figure = go.Figure()

    for fraction_name in fraction_columns:
        seasonal_figure.add_trace(
            go.Scatter(
                x=seasonal_monthly_means["Month"],
                y=seasonal_monthly_means[fraction_name],
                mode="lines+markers",
                name=fraction_name,
                marker={"size": 6},
                hovertemplate=(
                    "<b>Month:</b> %{x}<br><b>Mean weight:</b> %{y:.3f}<extra></extra>"
                ),
            )
        )

    seasonal_figure.update_layout(
        title="Seasonal pattern of litterfall components",
        xaxis_title="Calendar month",
        yaxis_title="Mean weight per monthly period",
        xaxis={"dtick": 1},
        hovermode="x unified",
    )

    seasonal_figure  # noqa: B018


@app.cell
def _(cleaned_litterfall, go, mo, pl):
    # ------------------------------------------------------------
    # Annual total
    # ------------------------------------------------------------

    annual_total = cleaned_litterfall.filter(pl.col("Text_Fraction") == "Total").drop_nulls(
        ["Year", "Weight"]
    )

    annual_summary = annual_total.group_by("Year").agg(pl.col("Weight").sum()).sort("Year")

    annual_figure = go.Figure(
        go.Bar(
            x=annual_summary["Year"],
            y=annual_summary["Weight"],
            hovertemplate=("<b>Year:</b> %{x}<br><b>Annual total:</b> %{y:.3f}<extra></extra>"),
        )
    )

    annual_figure.update_layout(
        title="Annual litterfall from the reported Total fraction",
        xaxis_title="Year",
        yaxis_title="Annual sum of reported weight",
    )

    # 2018 starts in August in the source, so flag it rather than
    # silently treating it as a complete calendar year.
    annual_note = mo.md(
        """
        **Side note:** 2018 begins in August in this dataset, so its
        annual value represents only the available part of that year and
        should not be interpreted as a full-year total.
        """
    )

    mo.vstack([annual_figure, annual_note])


@app.cell
def _(cleaned_litterfall, go, mo, pl):
    # ------------------------------------------------------------
    # Distribution of sampling-period weights by fraction
    # ------------------------------------------------------------

    fraction_distribution_figure = go.Figure()

    distribution_fraction_options = sorted(
        cleaned_litterfall["Text_Fraction"].drop_nulls().unique().to_list()
    )

    for distribution_fraction_name in distribution_fraction_options:
        distribution_fraction_part = cleaned_litterfall.filter(
            pl.col("Text_Fraction") == distribution_fraction_name
        ).drop_nulls(["Weight"])

        fraction_distribution_figure.add_trace(
            go.Box(
                y=distribution_fraction_part["Weight"],
                name=distribution_fraction_name,
                boxmean=True,
                hovertemplate=(
                    f"<b>Fraction:</b> {distribution_fraction_name}"
                    "<br><b>Weight:</b> %{y:.3f} kg/ha"
                    "<extra></extra>"
                ),
            )
        )

    fraction_distribution_figure.update_layout(
        title="Distribution of weight by litterfall fraction",
        yaxis_title="Weight per sampling period (kg/ha)",
        showlegend=False,
    )

    mo.vstack(
        [
            fraction_distribution_figure,
            mo.md(
                """
                `Total` is an aggregate category. It is shown here for
                comparison, but should not be treated as an independent
                component of the other fractions.
                """
            ),
        ]
    )


@app.cell
def _(filtered_litterfall, pl):
    # ------------------------------------------------------------
    # Detailed records with missing Weight
    # ------------------------------------------------------------

    filtered_litterfall.filter(pl.col("Weight").is_null())


@app.cell
def _(filtered_litterfall, pl):
    # ------------------------------------------------------------
    # Summary statistics for the selected Weight values
    # ------------------------------------------------------------

    filtered_litterfall.select(pl.col("Weight")).describe()


@app.cell
def _(mo):
    mo.md("""
    ### Key findings

    - Total litterfall shows a clear seasonal pattern, with the largest weights generally occurring in October (autumn).
    - Spruce needles are the dominant litterfall component.
    - Fine litter, cones and seeds, and woody material contribute smaller and more variable amounts.
    - Annual total litterfall varies substantially between years.
    """)


if __name__ == "__main__":
    app.run()
