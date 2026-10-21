# ruff: noqa: E501
"""Marimo EDA for periodic litterfall measurements at the Davos site."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    ### Dataset overview
    The periodic litterfall dataset contains litterfall measurements from the Davos (DAV) site over defined sampling periods. `Network`, `LWF_plot`, and `LWF_plot_code` identify the data source and site; `Period_id` identifies each sampling period; `Start_date` and `End_date` define the period; `Code_fraction` is the numeric code for the litterfall component and `Text_Fraction` gives its description; `Weight` is the litterfall weight in kg/ha for that period at the 65°C reference temperature; `period_midpoint` represents the centre of the sampling period; `duration_days` gives its length; and `start_month` and `start_year` describe when the period began. The plots show temporal, seasonal, and fraction-level variation in litterfall.
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

    period_litterfall_path = Path("data/period_litterfall_Davos_ICOS_2026-07-30.csv")

    raw_period_litterfall = pl.read_csv(
        period_litterfall_path,
        separator=";",
        skip_rows=2,
        encoding="windows-1252",
        null_values=["."],
    )
    return (raw_period_litterfall,)


@app.cell
def _(raw_period_litterfall):
    raw_period_litterfall  # noqa: B018


@app.cell
def _(mo, pl, raw_period_litterfall):
    null_percent = raw_period_litterfall.select(
        (pl.all().null_count() / max(raw_period_litterfall.height, 1) * 100).round(2)
    ).transpose(
        include_header=True,
        header_name="Column",
        column_names=["Null percentage"],
    )

    mo.ui.table(null_percent)


@app.cell
def _(pl, raw_period_litterfall):
    # ------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------

    prepared_period_litterfall = raw_period_litterfall.with_columns(
        [
            pl.col(column).cast(pl.String).str.strip_chars().alias(column)
            for column in [
                "Network",
                "LWF_plot",
                "LWF_plot_code",
                "Text_Fraction",
            ]
        ]
        + [
            pl.col(column).cast(pl.Float64, strict=False).alias(column)
            for column in ["Period_id", "Code_fraction", "Weight"]
        ]
        + [
            pl.col(column)
            .str.strptime(
                pl.Date,
                format="%d/%m/%Y",
                strict=False,
            )
            .alias(column)
            for column in ["Start_date", "End_date"]
        ]
    ).with_columns(
        [
            (pl.col("Start_date") + (pl.col("End_date") - pl.col("Start_date")) / 2).alias(
                "period_midpoint"
            ),
            (pl.col("End_date") - pl.col("Start_date")).dt.total_days().alias("duration_days"),
            pl.col("Start_date").dt.month().alias("start_month"),
            pl.col("Start_date").dt.year().alias("start_year"),
        ]
    )
    return (prepared_period_litterfall,)


@app.cell
def _(mo, prepared_period_litterfall):
    # ------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------

    overview_rows = len(prepared_period_litterfall)
    overview_periods = prepared_period_litterfall["Period_id"].n_unique()
    overview_fractions = prepared_period_litterfall["Text_Fraction"].n_unique()
    overview_valid_weights = prepared_period_litterfall["Weight"].is_not_null().sum()
    overview_missing_weights = prepared_period_litterfall["Weight"].is_null().sum()

    mo.vstack(
        [
            mo.md(
                """
                # Davos ICOS — Period Litterfall

                This notebook explores the **period-based litterfall
                dataset for the Davos (`DAV`) LWF site** in the ICOS
                network.

                Unlike the monthly litterfall product, this dataset reports
                **weight per defined sampling period**. Each observation is
                associated with a `Start_date`, `End_date`, and `Period_id`.

                The source defines five litterfall fractions:
                spruce needles; spruce cones and seeds; twigs, branches
                D<2cm, bark, wood; fine litter, flowers, budscales, lichens;
                and Total.
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
                        value=f"{overview_rows:,}",
                    ),
                    mo.stat(
                        label="Sampling periods",
                        value=f"{overview_periods:,}",
                    ),
                    mo.stat(
                        label="Fractions",
                        value=f"{overview_fractions:,}",
                    ),
                    mo.stat(
                        label="Missing weights",
                        value=f"{overview_missing_weights:,}",
                    ),
                ]
            ),
            mo.md(
                f"""
                **Coverage:** {prepared_period_litterfall["Start_date"].min().strftime("%d %B %Y")}
                to {prepared_period_litterfall["End_date"].max().strftime("%d %B %Y")}

                **Valid weights:** {overview_valid_weights:,}
                """
            ),
        ]
    )


@app.cell
def _(mo, pl, prepared_period_litterfall):
    # ------------------------------------------------------------
    # Interactive filters
    # ------------------------------------------------------------

    period_fraction_options = sorted(
        prepared_period_litterfall["Text_Fraction"].drop_nulls().cast(pl.String).unique().to_list()
    )

    period_year_options = sorted(
        prepared_period_litterfall["start_year"].drop_nulls().cast(pl.Int64).unique().to_list()
    )

    period_month_options = sorted(
        prepared_period_litterfall["start_month"].drop_nulls().cast(pl.Int64).unique().to_list()
    )

    period_fraction_selector = mo.ui.dropdown(
        options=["All"] + period_fraction_options,
        value="Total",
        label="Litter fraction",
    )

    period_year_selector = mo.ui.dropdown(
        options=["All"] + period_year_options,
        value="All",
        label="Start year",
    )

    period_month_selector = mo.ui.dropdown(
        options=["All"] + period_month_options,
        value="All",
        label="Start month",
    )

    mo.vstack(
        [
            mo.md("## Explore the data"),
            mo.hstack(
                [
                    period_fraction_selector,
                    period_year_selector,
                    period_month_selector,
                ]
            ),
        ]
    )
    return (
        period_fraction_selector,
        period_month_selector,
        period_year_selector,
    )


@app.cell
def _(
    period_fraction_selector,
    period_month_selector,
    period_year_selector,
    pl,
    prepared_period_litterfall,
):
    filtered_period_litterfall = prepared_period_litterfall

    if period_fraction_selector.value != "All":
        filtered_period_litterfall = filtered_period_litterfall.filter(
            pl.col("Text_Fraction").cast(pl.String) == period_fraction_selector.value
        )

    if period_year_selector.value != "All":
        filtered_period_litterfall = filtered_period_litterfall.filter(
            pl.col("start_year") == int(period_year_selector.value)
        )

    if period_month_selector.value != "All":
        filtered_period_litterfall = filtered_period_litterfall.filter(
            pl.col("start_month") == int(period_month_selector.value)
        )

    filtered_period_litterfall = filtered_period_litterfall.sort(
        ["Start_date", "End_date", "Period_id"]
    )

    filtered_period_litterfall  # noqa: B018
    return (filtered_period_litterfall,)


@app.cell
def _(filtered_period_litterfall, mo, pl):
    # ------------------------------------------------------------
    # Data quality
    # ------------------------------------------------------------

    period_quality = pl.DataFrame(
        {
            "Quantity": [
                "Total observations",
                "Valid weights",
                "Missing weights (NaN)",
                "Missing start dates (NaT)",
                "Missing end dates (NaT)",
                "Missing period IDs",
                "Periods represented",
            ],
            "Count": [
                filtered_period_litterfall.height,
                filtered_period_litterfall["Weight"].is_not_null().sum(),
                filtered_period_litterfall["Weight"].is_null().sum(),
                filtered_period_litterfall["Start_date"].is_null().sum(),
                filtered_period_litterfall["End_date"].is_null().sum(),
                filtered_period_litterfall["Period_id"].is_null().sum(),
                filtered_period_litterfall["Period_id"].n_unique(),
            ],
        }
    )

    mo.vstack(
        [
            mo.md("## Data quality"),
            period_quality,
        ]
    )


@app.cell
def _(filtered_period_litterfall, pl):
    # Missingness in the selected table.
    period_missingness = (
        pl.DataFrame(
            {
                "Variable": filtered_period_litterfall.columns,
                "Missing count": [
                    filtered_period_litterfall[column].null_count()
                    for column in filtered_period_litterfall.columns
                ],
            }
        )
        .with_columns(
            (100 * pl.col("Missing count") / max(filtered_period_litterfall.height, 1)).alias(
                "Missing %"
            )
        )
        .sort(
            ["Missing count", "Variable"],
            descending=[True, False],
        )
    )

    period_missingness  # noqa: B018


@app.cell
def _(filtered_period_litterfall, go, mo, pl):
    # ------------------------------------------------------------
    # Main period-by-period time series
    # ------------------------------------------------------------

    period_valid_series = filtered_period_litterfall.filter(
        pl.col("period_midpoint").is_not_null() & pl.col("Weight").is_not_null()
    ).sort("period_midpoint")

    period_missing_series = filtered_period_litterfall.filter(
        pl.col("period_midpoint").is_not_null() & pl.col("Weight").is_null()
    ).sort("period_midpoint")

    period_time_figure = go.Figure()

    if not period_valid_series.is_empty():
        period_time_figure.add_trace(
            go.Scatter(
                x=period_valid_series["period_midpoint"].to_list(),
                y=period_valid_series["Weight"].to_list(),
                mode="lines+markers",
                name="Weight",
                marker={"size": 7},
                customdata=period_valid_series.select(
                    [
                        "Period_id",
                        "Start_date",
                        "End_date",
                        "duration_days",
                    ]
                ).to_numpy(),
                hovertemplate=(
                    "<b>Period ID:</b> %{customdata[0]}"
                    "<br><b>Period:</b> "
                    "%{customdata[1]|%d %B %Y}"
                    " → "
                    "%{customdata[2]|%d %B %Y}"
                    "<br><b>Duration:</b> %{customdata[3]} days"
                    "<br><b>Weight:</b> %{y:.3f} kg/ha"
                    "<extra></extra>"
                ),
            )
        )

    if not period_missing_series.is_empty():
        if not period_valid_series.is_empty():
            period_min_weight = period_valid_series["Weight"].min()
            period_max_weight = period_valid_series["Weight"].max()
            period_range = period_max_weight - period_min_weight

            if period_range == 0:
                period_range = max(
                    abs(period_min_weight) * 0.1,
                    1.0,
                )

            period_missing_y = period_min_weight - 0.08 * period_range
        else:
            period_missing_y = 0

        period_time_figure.add_trace(
            go.Scatter(
                x=period_missing_series["period_midpoint"].to_list(),
                y=[period_missing_y] * period_missing_series.height,
                mode="markers",
                name="Missing weight",
                marker={"size": 10, "symbol": "x"},
                customdata=period_missing_series.select(
                    [
                        "Period_id",
                        "Start_date",
                        "End_date",
                    ]
                ).to_numpy(),
                hovertemplate=(
                    "<b>Period ID:</b> %{customdata[0]}"
                    "<br><b>Period:</b> "
                    "%{customdata[1]|%d %B %Y}"
                    " → "
                    "%{customdata[2]|%d %B %Y}"
                    "<br><b>Weight:</b> NaN"
                    "<extra></extra>"
                ),
            )
        )

    period_time_figure.update_layout(
        title="Litterfall weight by sampling period",
        xaxis_title="Sampling-period midpoint",
        yaxis_title="Weight per sampling period (kg/ha)",
        hovermode="closest",
    )

    mo.vstack(
        [
            period_time_figure,
            mo.md(
                f"""
                **Valid observations plotted:** {period_valid_series.height:,}  
                **Missing weights marked as X:** {period_missing_series.height:,}
                """
            ),
        ]
    )


@app.cell
def _(go, pl, prepared_period_litterfall):
    # ------------------------------------------------------------
    # Fraction-level seasonal pattern
    # ------------------------------------------------------------

    component_period_data = prepared_period_litterfall.filter(
        (pl.col("Text_Fraction") != "Total")
        & pl.col("start_month").is_not_null()
        & pl.col("Weight").is_not_null()
    )

    seasonal_fraction_figure = go.Figure()

    for seasonal_fraction_name in sorted(
        component_period_data["Text_Fraction"].drop_nulls().unique().to_list()
    ):
        seasonal_fraction_part = component_period_data.filter(
            pl.col("Text_Fraction") == seasonal_fraction_name
        )

        fraction_monthly = (
            seasonal_fraction_part.group_by("start_month")
            .agg(pl.col("Weight").mean())
            .sort("start_month")
        )

        seasonal_fraction_figure.add_trace(
            go.Scatter(
                x=fraction_monthly["start_month"].to_list(),
                y=fraction_monthly["Weight"].to_list(),
                mode="lines+markers",
                name=seasonal_fraction_name,
                marker={"size": 6},
                hovertemplate=(
                    "<b>Start month:</b> %{x}"
                    "<br><b>Mean period weight:</b> %{y:.3f} kg/ha"
                    "<extra></extra>"
                ),
            )
        )

    seasonal_fraction_figure.update_layout(
        title="Seasonal pattern of litterfall components",
        xaxis_title="Calendar month of sampling-period start",
        yaxis_title="Mean weight per sampling period (kg/ha)",
        xaxis={"dtick": 1},
        hovermode="x unified",
    )

    seasonal_fraction_figure  # noqa: B018


@app.cell
def _(go, mo, pl, prepared_period_litterfall):
    # ------------------------------------------------------------
    # Distribution of sampling-period weights by fraction
    # ------------------------------------------------------------

    fraction_distribution_figure = go.Figure()

    for distribution_fraction_name in sorted(
        prepared_period_litterfall["Text_Fraction"].drop_nulls().unique().to_list()
    ):
        distribution_fraction_part = prepared_period_litterfall.filter(
            pl.col("Text_Fraction") == distribution_fraction_name
        ).filter(pl.col("Weight").is_not_null())

        fraction_distribution_figure.add_trace(
            go.Box(
                y=distribution_fraction_part["Weight"].to_list(),
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
def _(mo, prepared_period_litterfall):
    # ------------------------------------------------------------
    # Sampling-period duration
    # ------------------------------------------------------------

    duration_summary = (
        prepared_period_litterfall.select(
            [
                "Period_id",
                "Start_date",
                "End_date",
                "duration_days",
            ]
        )
        .unique()
        .sort("Start_date")
    )

    mo.vstack(
        [
            mo.md(
                """
                ## Sampling-period duration

                Period litterfall observations are associated with explicit
                collection intervals. This table makes the interval lengths
                visible rather than assuming that every period has the same
                duration.
                """
            ),
            duration_summary,
        ]
    )


@app.cell
def _(pl, prepared_period_litterfall):
    # ------------------------------------------------------------
    # Missing-weight records
    # ------------------------------------------------------------

    prepared_period_litterfall.filter(pl.col("Weight").is_null())


@app.cell
def _(filtered_period_litterfall):
    # ------------------------------------------------------------
    # Descriptive statistics for the selected data
    # ------------------------------------------------------------

    filtered_period_litterfall.select("Weight").describe()


@app.cell
def _(filtered_period_litterfall):
    # ------------------------------------------------------------
    # Final filtered records
    # ------------------------------------------------------------

    filtered_period_litterfall  # noqa: B018


@app.cell
def _(mo):
    mo.md("""
    ### Key findings

    - Litterfall weight varies strongly between sampling periods, with several pronounced peaks and generally lower values between peaks.
    - Spruce needles are the dominant litterfall component.
    - Fine litter, cones and seeds, and woody material contribute smaller amounts.
    - Component weights vary with sampling-period start month, with particularly large spruce-needle and woody-fraction values in later autumn periods.
    """)


if __name__ == "__main__":
    app.run()
