# ruff: noqa: E501
"""Marimo EDA for periodic litterfall measurements at LWF sites."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    ### Dataset overview
    The periodic LWF litterfall dataset contains litterfall measurements from multiple LWF sites over defined sampling periods. `Network`, `LWF_plot`, and `LWF_plot_code` identify the source and site, while `Sub_plot` identifies the sampling subplot. `Start_date`, `End_date`, and `Number_of_days` describe the sampling period. `Code_fraction_1` and `Code_fraction_2` are numeric identifiers for two levels of litterfall classification, with `Text_Fraction_1` and `Text_Fraction_2` giving their descriptions. `Tree species` identifies the associated species, `Weight` gives litterfall mass in kg/ha, and `period_midpoint` gives the centre of the sampling period. `derived_days` and `Year_from_start` provide derived temporal measures used in the analysis.
    """)


@app.cell
def _():
    from io import StringIO
    from pathlib import Path

    import marimo as mo
    import plotly.graph_objects as go
    import polars as pl

    return Path, StringIO, go, mo, pl


@app.cell
def _(Path, StringIO, pl):
    # Source file metadata states:
    # "Weight per sampling period in kg/ha (ref temp 65°C), . is missing value,
    # fraction_1 is the most detailed fraction, fraction_2 is the less detailed
    # fraction to which fraction_1 belongs".
    litterfall_period_path = Path("data/period_litterfall_LWF_2026-07-30.csv")

    raw_period_litterfall = pl.read_csv(
        StringIO(litterfall_period_path.read_text(encoding="windows-1252")),
        separator=";",
        skip_rows=2,
        null_values=".",
        schema_overrides={
            "Code_fraction_1": pl.Float64,
            "Code_fraction_2": pl.Float64,
        },
    )
    return (raw_period_litterfall,)


@app.cell
def _(mo, pl, raw_period_litterfall):
    null_percent = raw_period_litterfall.select(
        [
            (pl.col(column).is_null().mean() * 100).round(2).alias(column)
            for column in raw_period_litterfall.columns
        ]
    ).transpose(
        include_header=True,
        header_name="Column",
        column_names=["Null percentage"],
    )

    mo.ui.table(null_percent)


@app.cell
def _(pl, raw_period_litterfall):
    period_litterfall = raw_period_litterfall

    period_text_columns = [
        "Network",
        "LWF_plot",
        "LWF_plot_code",
        "Sub_plot",
        "Text_Fraction_1",
        "Text_Fraction_2",
        "Tree_species",
    ]

    period_litterfall = period_litterfall.with_columns(
        pl.col(period_text_columns).cast(pl.String).str.strip_chars()
    )

    period_numeric_columns = [
        "Number_of_days",
        "Code_fraction_1",
        "Code_fraction_2",
        "Weight",
    ]

    period_litterfall = period_litterfall.with_columns(
        pl.col(period_numeric_columns).cast(pl.Float64, strict=False)
    )

    period_litterfall = period_litterfall.with_columns(
        pl.col("Start_date").str.strptime(pl.Datetime, format="%d/%m/%y", strict=False),
        pl.col("End_date").str.strptime(pl.Datetime, format="%d/%m/%y", strict=False),
    )

    period_litterfall = period_litterfall.with_columns(
        (pl.col("Start_date") + (pl.col("End_date") - pl.col("Start_date")) / 2).alias(
            "period_midpoint"
        ),
        (pl.col("End_date") - pl.col("Start_date")).dt.total_days().alias("derived_days"),
        pl.col("Start_date").dt.year().alias("Year_from_start"),
    )
    return (period_litterfall,)


@app.cell
def _(mo, period_litterfall):
    mo.vstack(
        [
            mo.md(
                """
                ## Period litterfall — LWF

                This dataset reports **weight per sampling period** in
                **kg/ha at a 65°C reference temperature**. It is not the
                interpolated monthly dataset: each row corresponds to a
                defined sampling period from `Start_date` to `End_date`.

                The source also defines `fraction_1` as the **most detailed
                fraction** and `fraction_2` as the **less detailed fraction**
                to which fraction_1 belongs. `.` is treated as missing.
                """
            ),
            mo.md(
                f"""
                **Rows:** {len(period_litterfall):,}  
                **LWF sites:** {period_litterfall["LWF_plot_code"].n_unique()}  
                **Date range:** {period_litterfall["Start_date"].min().strftime("%d %B %Y")} to {
                    period_litterfall["End_date"].max().strftime("%d %B %Y")
                }  
                **Valid weights:** {period_litterfall["Weight"].is_not_null().sum():,}
                """
            ),
        ]
    )


@app.cell
def _(mo, period_litterfall, pl):
    period_site_options = sorted(
        period_litterfall.filter(pl.col("LWF_plot_code").is_not_null())
        .get_column("LWF_plot_code")
        .cast(pl.String)
        .unique()
        .to_list()
    )

    period_fraction1_options = sorted(
        period_litterfall.filter(
            pl.col("Text_Fraction_1").is_not_null()
            & (pl.col("Text_Fraction_1").cast(pl.String).str.len_chars() > 0)
        )
        .get_column("Text_Fraction_1")
        .cast(pl.String)
        .unique()
        .to_list()
    )

    period_fraction2_options = sorted(
        period_litterfall.filter(
            pl.col("Text_Fraction_2").is_not_null()
            & (pl.col("Text_Fraction_2").cast(pl.String).str.len_chars() > 0)
        )
        .get_column("Text_Fraction_2")
        .cast(pl.String)
        .unique()
        .to_list()
    )

    period_species_options = sorted(
        period_litterfall.filter(
            pl.col("Tree_species").is_not_null()
            & (pl.col("Tree_species").cast(pl.String).str.len_chars() > 0)
        )
        .get_column("Tree_species")
        .cast(pl.String)
        .unique()
        .to_list()
    )

    period_year_options = sorted(
        period_litterfall.filter(pl.col("Year_from_start").is_not_null())
        .get_column("Year_from_start")
        .cast(pl.Int64)
        .unique()
        .to_list()
    )

    period_site_selector = mo.ui.dropdown(
        options=["All"] + period_site_options,
        value="All",
        label="LWF site",
    )

    period_fraction1_selector = mo.ui.dropdown(
        options=["All"] + period_fraction1_options,
        value="All",
        label="Detailed fraction (fraction_1)",
    )

    period_fraction2_selector = mo.ui.dropdown(
        options=["All"] + period_fraction2_options,
        value="All",
        label="Parent fraction (fraction_2)",
    )

    period_species_selector = mo.ui.dropdown(
        options=["All"] + period_species_options,
        value="All",
        label="Tree species",
    )

    period_year_selector = mo.ui.dropdown(
        options=["All"] + period_year_options,
        value="All",
        label="Start year",
    )

    mo.vstack(
        [
            mo.md("### Filters"),
            mo.hstack(
                [
                    period_site_selector,
                    period_fraction1_selector,
                ]
            ),
            mo.hstack(
                [
                    period_fraction2_selector,
                    period_species_selector,
                    period_year_selector,
                ]
            ),
        ]
    )
    return (
        period_fraction1_selector,
        period_fraction2_selector,
        period_site_selector,
        period_species_selector,
        period_year_selector,
    )


@app.cell
def _(
    period_fraction1_selector,
    period_fraction2_selector,
    period_litterfall,
    period_site_selector,
    period_species_selector,
    period_year_selector,
    pl,
):
    filtered_period_litterfall = period_litterfall

    if period_site_selector.value != "All":
        filtered_period_litterfall = filtered_period_litterfall.filter(
            pl.col("LWF_plot_code").cast(pl.String) == period_site_selector.value
        )

    if period_fraction1_selector.value != "All":
        filtered_period_litterfall = filtered_period_litterfall.filter(
            pl.col("Text_Fraction_1").cast(pl.String) == period_fraction1_selector.value
        )

    if period_fraction2_selector.value != "All":
        filtered_period_litterfall = filtered_period_litterfall.filter(
            pl.col("Text_Fraction_2").cast(pl.String) == period_fraction2_selector.value
        )

    if period_species_selector.value != "All":
        filtered_period_litterfall = filtered_period_litterfall.filter(
            pl.col("Tree_species").cast(pl.String) == period_species_selector.value
        )

    if period_year_selector.value != "All":
        filtered_period_litterfall = filtered_period_litterfall.filter(
            pl.col("Year_from_start") == int(period_year_selector.value)
        )

    filtered_period_litterfall = filtered_period_litterfall.sort(
        ["Start_date", "End_date", "LWF_plot_code"]
    )

    filtered_period_litterfall  # noqa: B018
    return (filtered_period_litterfall,)


@app.cell
def _(filtered_period_litterfall, mo, pl):
    period_total_rows = len(filtered_period_litterfall)
    period_valid_weight = filtered_period_litterfall["Weight"].is_not_null().sum()
    period_missing_weight = filtered_period_litterfall["Weight"].is_null().sum()
    period_missing_start = filtered_period_litterfall["Start_date"].is_null().sum()
    period_missing_end = filtered_period_litterfall["End_date"].is_null().sum()
    period_missing_days = filtered_period_litterfall["Number_of_days"].is_null().sum()

    period_quality_summary = pl.DataFrame(
        {
            "Quantity": [
                "Total observations",
                "Valid weights",
                "Missing weight (NaN)",
                "Missing start date (NaT)",
                "Missing end date (NaT)",
                "Missing stated number of days",
            ],
            "Count": [
                period_total_rows,
                period_valid_weight,
                period_missing_weight,
                period_missing_start,
                period_missing_end,
                period_missing_days,
            ],
        }
    )

    mo.vstack(
        [
            mo.md("### Data quality"),
            period_quality_summary,
        ]
    )


@app.cell
def _(filtered_period_litterfall, pl):
    period_missingness = (
        pl.DataFrame(
            {
                "variable": filtered_period_litterfall.columns,
                "missing_count": [
                    filtered_period_litterfall[column].is_null().sum()
                    for column in filtered_period_litterfall.columns
                ],
            }
        )
        .with_columns(
            (100 * pl.col("missing_count") / max(len(filtered_period_litterfall), 1)).alias(
                "missing_percent"
            )
        )
        .sort(
            ["missing_count", "variable"],
            descending=[True, False],
        )
    )

    period_missingness  # noqa: B018


@app.cell
def _(go, period_litterfall, pl):
    # ------------------------------------------------------------
    # Fraction-level seasonal pattern
    # ------------------------------------------------------------

    # Exclude aggregate "Total..." categories so only individual
    # litterfall components are plotted.
    component_period_data = (
        period_litterfall.filter(
            ~pl.col("Text_Fraction_2")
            .cast(pl.String)
            .str.strip_chars()
            .str.to_lowercase()
            .str.contains("total")
        )
        .drop_nulls(["Start_date", "Weight"])
        .with_columns(pl.col("Start_date").dt.month().alias("start_month"))
    )

    seasonal_fraction_figure = go.Figure()

    for seasonal_fraction_name in sorted(
        component_period_data["Text_Fraction_2"].drop_nulls().unique().to_list()
    ):
        seasonal_fraction_part = component_period_data.filter(
            pl.col("Text_Fraction_2") == seasonal_fraction_name
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
def _(go, period_litterfall, pl):
    # ------------------------------------------------------------
    # Distribution of sampling-period weights by fraction
    # ------------------------------------------------------------

    fraction_distribution_figure = go.Figure()

    # Exclude aggregate "Total..." categories (e.g. "Total" and "Total
    # without branches >2cm") so only individual components are shown.
    distribution_fraction_options = sorted(
        period_litterfall.filter(
            pl.col("Text_Fraction_2").is_not_null()
            & ~pl.col("Text_Fraction_2")
            .cast(pl.String)
            .str.strip_chars()
            .str.to_lowercase()
            .str.contains("total")
        )
        .get_column("Text_Fraction_2")
        .unique()
        .to_list()
    )

    for distribution_fraction_name in distribution_fraction_options:
        distribution_fraction_part = period_litterfall.filter(
            (pl.col("Text_Fraction_2") == distribution_fraction_name)
            & pl.col("Weight").is_not_null()
        )

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

    fraction_distribution_figure  # noqa: B018


@app.cell
def _(filtered_period_litterfall, pl):
    period_missing_weight_rows = filtered_period_litterfall.filter(pl.col("Weight").is_null())

    period_missing_weight_rows  # noqa: B018


@app.cell
def _(filtered_period_litterfall):
    filtered_period_litterfall["Weight"].describe()


@app.cell
def _(mo, period_litterfall, pl):
    periodcmp_site_codes = sorted(
        period_litterfall.filter(pl.col("LWF_plot_code").is_not_null())
        .get_column("LWF_plot_code")
        .cast(pl.String)
        .unique()
        .to_list()
    )

    periodcmp_fraction1_options = sorted(
        period_litterfall.filter(
            pl.col("Text_Fraction_1").is_not_null()
            & (pl.col("Text_Fraction_1").cast(pl.String).str.len_chars() > 0)
        )
        .get_column("Text_Fraction_1")
        .cast(pl.String)
        .unique()
        .to_list()
    )

    periodcmp_fraction2_options = sorted(
        period_litterfall.filter(
            pl.col("Text_Fraction_2").is_not_null()
            & (pl.col("Text_Fraction_2").cast(pl.String).str.len_chars() > 0)
        )
        .get_column("Text_Fraction_2")
        .cast(pl.String)
        .unique()
        .to_list()
    )

    periodcmp_species_options = sorted(
        period_litterfall.filter(
            pl.col("Tree_species").is_not_null()
            & (pl.col("Tree_species").cast(pl.String).str.len_chars() > 0)
        )
        .get_column("Tree_species")
        .cast(pl.String)
        .unique()
        .to_list()
    )

    periodcmp_site_a_selector = mo.ui.dropdown(
        options=periodcmp_site_codes,
        value=periodcmp_site_codes[0],
        label="Site A",
    )

    periodcmp_site_b_selector = mo.ui.dropdown(
        options=periodcmp_site_codes,
        value=(
            periodcmp_site_codes[1] if len(periodcmp_site_codes) > 1 else periodcmp_site_codes[0]
        ),
        label="Site B",
    )

    periodcmp_fraction1_selector = mo.ui.dropdown(
        options=["All"] + periodcmp_fraction1_options,
        value="All",
        label="Detailed fraction",
    )

    periodcmp_fraction2_selector = mo.ui.dropdown(
        options=["All"] + periodcmp_fraction2_options,
        value="All",
        label="Parent fraction",
    )

    periodcmp_species_selector = mo.ui.dropdown(
        options=["All"] + periodcmp_species_options,
        value="All",
        label="Tree species",
    )

    mo.vstack(
        [
            mo.md("## Compare two LWF plot sites"),
            mo.hstack(
                [
                    periodcmp_site_a_selector,
                    periodcmp_site_b_selector,
                ]
            ),
            mo.hstack(
                [
                    periodcmp_fraction1_selector,
                    periodcmp_fraction2_selector,
                    periodcmp_species_selector,
                ]
            ),
        ]
    )
    return (
        periodcmp_fraction1_selector,
        periodcmp_fraction2_selector,
        periodcmp_site_a_selector,
        periodcmp_site_b_selector,
        periodcmp_species_selector,
    )


@app.cell
def _(
    period_litterfall,
    periodcmp_fraction1_selector,
    periodcmp_fraction2_selector,
    periodcmp_site_a_selector,
    periodcmp_site_b_selector,
    periodcmp_species_selector,
    pl,
):
    periodcmp_selected_a = periodcmp_site_a_selector.value
    periodcmp_selected_b = periodcmp_site_b_selector.value

    periodcmp_data_a = period_litterfall.filter(
        pl.col("LWF_plot_code").cast(pl.String) == periodcmp_selected_a
    )

    periodcmp_data_b = period_litterfall.filter(
        pl.col("LWF_plot_code").cast(pl.String) == periodcmp_selected_b
    )

    if periodcmp_fraction1_selector.value != "All":
        periodcmp_data_a = periodcmp_data_a.filter(
            pl.col("Text_Fraction_1").cast(pl.String) == periodcmp_fraction1_selector.value
        )
        periodcmp_data_b = periodcmp_data_b.filter(
            pl.col("Text_Fraction_1").cast(pl.String) == periodcmp_fraction1_selector.value
        )

    if periodcmp_fraction2_selector.value != "All":
        periodcmp_data_a = periodcmp_data_a.filter(
            pl.col("Text_Fraction_2").cast(pl.String) == periodcmp_fraction2_selector.value
        )
        periodcmp_data_b = periodcmp_data_b.filter(
            pl.col("Text_Fraction_2").cast(pl.String) == periodcmp_fraction2_selector.value
        )

    if periodcmp_species_selector.value != "All":
        periodcmp_data_a = periodcmp_data_a.filter(
            pl.col("Tree_species").cast(pl.String) == periodcmp_species_selector.value
        )
        periodcmp_data_b = periodcmp_data_b.filter(
            pl.col("Tree_species").cast(pl.String) == periodcmp_species_selector.value
        )

    periodcmp_data_a = periodcmp_data_a.sort("period_midpoint")
    periodcmp_data_b = periodcmp_data_b.sort("period_midpoint")
    return periodcmp_data_a, periodcmp_data_b


@app.cell
def _(
    go,
    periodcmp_data_a,
    periodcmp_data_b,
    periodcmp_site_a_selector,
    periodcmp_site_b_selector,
):
    periodcmp_valid_a = periodcmp_data_a.drop_nulls(["period_midpoint", "Weight"])
    periodcmp_valid_b = periodcmp_data_b.drop_nulls(["period_midpoint", "Weight"])

    periodcmp_figure = go.Figure()

    periodcmp_figure.add_trace(
        go.Scatter(
            x=periodcmp_valid_a["period_midpoint"].to_list(),
            y=periodcmp_valid_a["Weight"].to_list(),
            mode="lines+markers",
            name=periodcmp_site_a_selector.value,
            marker={"size": 6},
            hovertemplate=(
                f"<b>Site:</b> {periodcmp_site_a_selector.value}"
                "<br><b>Period midpoint:</b> "
                "%{x|%d %B %Y}"
                "<br><b>Weight:</b> %{y:.3f} kg/ha"
                "<extra></extra>"
            ),
        )
    )

    periodcmp_figure.add_trace(
        go.Scatter(
            x=periodcmp_valid_b["period_midpoint"].to_list(),
            y=periodcmp_valid_b["Weight"].to_list(),
            mode="lines+markers",
            name=periodcmp_site_b_selector.value,
            marker={"size": 6},
            hovertemplate=(
                f"<b>Site:</b> {periodcmp_site_b_selector.value}"
                "<br><b>Period midpoint:</b> "
                "%{x|%d %B %Y}"
                "<br><b>Weight:</b> %{y:.3f} kg/ha"
                "<extra></extra>"
            ),
        )
    )

    periodcmp_figure.update_layout(
        title=(
            "Period litterfall: "
            f"{periodcmp_site_a_selector.value} vs "
            f"{periodcmp_site_b_selector.value}"
        ),
        xaxis_title="Sampling-period midpoint",
        yaxis_title="Weight per sampling period (kg/ha)",
        hovermode="x unified",
        legend_title="LWF plot",
    )

    periodcmp_figure  # noqa: B018


@app.cell
def _(mo, period_litterfall, pl):
    # ------------------------------------------------------------
    # Multi-site comparison over time
    #
    # Note: `mo.ui.multiplot` is not part of marimo's widget API.
    # The equivalent here is `mo.ui.multiselect` to pick any number
    # of LWF sites, combined with a single Plotly figure that
    # overlays one trace per selected site.
    # ------------------------------------------------------------
    multisite_site_codes = sorted(
        period_litterfall.filter(pl.col("LWF_plot_code").is_not_null())
        .get_column("LWF_plot_code")
        .cast(pl.String)
        .unique()
        .to_list()
    )

    multisite_fraction1_options = sorted(
        period_litterfall.filter(
            pl.col("Text_Fraction_1").is_not_null()
            & (pl.col("Text_Fraction_1").cast(pl.String).str.len_chars() > 0)
        )
        .get_column("Text_Fraction_1")
        .cast(pl.String)
        .unique()
        .to_list()
    )

    multisite_fraction2_options = sorted(
        period_litterfall.filter(
            pl.col("Text_Fraction_2").is_not_null()
            & (pl.col("Text_Fraction_2").cast(pl.String).str.len_chars() > 0)
        )
        .get_column("Text_Fraction_2")
        .cast(pl.String)
        .unique()
        .to_list()
    )

    multisite_species_options = sorted(
        period_litterfall.filter(
            pl.col("Tree_species").is_not_null()
            & (pl.col("Tree_species").cast(pl.String).str.len_chars() > 0)
        )
        .get_column("Tree_species")
        .cast(pl.String)
        .unique()
        .to_list()
    )

    multisite_site_selector = mo.ui.multiselect(
        options=multisite_site_codes,
        value=multisite_site_codes[: min(4, len(multisite_site_codes))],
        label="LWF sites to compare",
    )

    multisite_fraction1_selector = mo.ui.dropdown(
        options=["All"] + multisite_fraction1_options,
        value="All",
        label="Detailed fraction",
    )

    multisite_fraction2_selector = mo.ui.dropdown(
        options=["All"] + multisite_fraction2_options,
        value="All",
        label="Parent fraction",
    )

    multisite_species_selector = mo.ui.dropdown(
        options=["All"] + multisite_species_options,
        value="All",
        label="Tree species",
    )

    mo.vstack(
        [
            mo.md(
                """
                ## Multi-site comparison over time

                Pick any number of LWF sites to overlay their litterfall
                time series on one chart. When a fraction/species filter
                is left at "All", each site's line shows **summed**
                weight across all fractions for that sampling period,
                since a single period can contain multiple fraction-level
                rows.
                """
            ),
            multisite_site_selector,
            mo.hstack(
                [
                    multisite_fraction1_selector,
                    multisite_fraction2_selector,
                    multisite_species_selector,
                ]
            ),
        ]
    )
    return (
        multisite_fraction1_selector,
        multisite_fraction2_selector,
        multisite_site_selector,
        multisite_species_selector,
    )


@app.cell
def _(
    multisite_fraction1_selector,
    multisite_fraction2_selector,
    multisite_site_selector,
    multisite_species_selector,
    period_litterfall,
    pl,
):
    multisite_combined = period_litterfall.filter(
        pl.col("LWF_plot_code").cast(pl.String).is_in(multisite_site_selector.value)
    )

    if multisite_fraction1_selector.value != "All":
        multisite_combined = multisite_combined.filter(
            pl.col("Text_Fraction_1").cast(pl.String) == multisite_fraction1_selector.value
        )

    if multisite_fraction2_selector.value != "All":
        multisite_combined = multisite_combined.filter(
            pl.col("Text_Fraction_2").cast(pl.String) == multisite_fraction2_selector.value
        )

    if multisite_species_selector.value != "All":
        multisite_combined = multisite_combined.filter(
            pl.col("Tree_species").cast(pl.String) == multisite_species_selector.value
        )

    multisite_combined = multisite_combined.drop_nulls(["period_midpoint", "Weight"]).sort(
        ["LWF_plot_code", "period_midpoint"]
    )

    multisite_combined  # noqa: B018
    return (multisite_combined,)


@app.cell
def _(go, mo, multisite_combined, multisite_site_selector, pl):
    if len(multisite_site_selector.value) == 0:
        multisite_figure_output = mo.md(
            "*Select at least one LWF site above to see the comparison.*"
        )
    else:
        multisite_figure = go.Figure()

        for multisite_plot_code in multisite_site_selector.value:
            multisite_site_part = multisite_combined.filter(
                pl.col("LWF_plot_code").cast(pl.String) == multisite_plot_code
            )

            multisite_site_series = (
                multisite_site_part.group_by("period_midpoint")
                .agg(pl.col("Weight").sum())
                .sort("period_midpoint")
            )

            multisite_figure.add_trace(
                go.Scatter(
                    x=multisite_site_series["period_midpoint"].to_list(),
                    y=multisite_site_series["Weight"].to_list(),
                    mode="lines+markers",
                    name=multisite_plot_code,
                    marker={"size": 5},
                    hovertemplate=(
                        f"<b>Site:</b> {multisite_plot_code}"
                        "<br><b>Period midpoint:</b> %{x|%d %B %Y}"
                        "<br><b>Weight:</b> %{y:.3f} kg/ha"
                        "<extra></extra>"
                    ),
                )
            )

        multisite_figure.update_layout(
            title="Litterfall over time across selected LWF sites",
            xaxis_title="Sampling-period midpoint",
            yaxis_title="Summed weight per sampling period (kg/ha)",
            hovermode="x unified",
            legend_title="LWF plot",
        )

        multisite_figure_output = multisite_figure

    multisite_figure_output  # noqa: B018


@app.cell
def _(mo):
    mo.md("""
    ### Key findings

    - Broad litterfall fractions show pronounced peaks around autumn sampling periods.
    - The dominant litterfall component differs by site and species.
    - The same sampling period can contain multiple fraction-level observations.
    - Total litterfall should therefore be interpreted separately from individual component weights.
    """)


if __name__ == "__main__":
    app.run()
