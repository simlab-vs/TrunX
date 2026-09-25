# ruff: noqa: E501
"""Marimo EDA for monthly litterfall measurements at LWF sites."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    ### Dataset overview
    The monthly LWF litterfall dataset contains litterfall measurements from multiple LWF sites over time. `Network`, `LWF_plot`, and `LWF_plot_code` identify the data source and site, while `Sub_plot` identifies the sampling subplot. `Year`, `Month`, `Start_date`, and `End_date` describe the sampling period, and `Number_of_days` gives its duration. `Code_fraction_2` is the numeric code for the litterfall fraction and `Text_Fraction_2` gives its description. `Tree species` identifies the associated species, `Weight` gives litterfall weight in kg/ha for the sampling period, and `period_midpoint` gives the midpoint of the sampling period. The plots show temporal, seasonal, and fraction-level variation in litterfall.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import plotly.graph_objects as go
    import polars as pl

    return Path, go, mo, pl


@app.cell
def _(Path, pl):
    litterfall_path = Path("data/monthly_litterfall_LWF_2026-07-30.csv")

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
    null_percent = litterfall.select(
        [
            (pl.col(column).null_count() * 100 / pl.len()).round(2).alias(column)
            for column in litterfall.columns
        ]
    ).transpose(include_header=True, header_name="Column", column_names=["Null percentage"])

    mo.ui.table(null_percent)
    return


@app.cell
def _(litterfall):
    litterfall  # noqa: B018 - Marimo displays the final cell expression.
    return


@app.cell
def _(litterfall, pl):

    cleaned_litterfall = litterfall.clone()

    text_columns = [
        "Network",
        "LWF_plot",
        "LWF_plot_code",
        "Sub_plot",
        "Text_Fraction_2",
        "Tree_species",
    ]

    cleaned_litterfall = cleaned_litterfall.with_columns(
        [
            pl.col(litterfall_column)
            .cast(pl.String, strict=False)
            .str.strip_chars()
            .alias(litterfall_column)
            for litterfall_column in text_columns
        ]
    )

    numeric_columns = [
        "Year",
        "Month",
        "Number_of_days",
        "Code_fraction_2",
        "Weight",
    ]

    cleaned_litterfall = cleaned_litterfall.with_columns(
        [
            pl.col(litterfall_column).cast(pl.Float64, strict=False).alias(litterfall_column)
            for litterfall_column in numeric_columns
        ]
    )

    cleaned_litterfall = cleaned_litterfall.with_columns(
        [
            pl.col("Start_date").str.strptime(
                pl.Datetime,
                format="%d/%m/%y",
                strict=False,
            ),
            pl.col("End_date").str.strptime(
                pl.Datetime,
                format="%d/%m/%y",
                strict=False,
            ),
        ]
    ).with_columns(
        (pl.col("Start_date") + (pl.col("End_date") - pl.col("Start_date")) / 2).alias(
            "period_midpoint"
        )
    )
    return (cleaned_litterfall,)


@app.cell
def _(cleaned_litterfall, mo):
    mo.vstack(
        [
            mo.md(
                """
                ## Monthly litterfall — LWF

                The source file describes **interpolated monthly litterfall
                weight**, referenced to **65°C**, in **kg/ha**. A row
                represents a sampling period (`Start_date` → `End_date`) for
                an LWF site and a litter fraction/species category.

                The notebook keeps the source categories and treats `.` as
                missing rather than as zero.
                """
            ),
            mo.md(
                f"""
                **Rows:** {len(cleaned_litterfall):,}  
                **LWF sites:** {cleaned_litterfall["LWF_plot_code"].drop_nulls().n_unique()}  
                **Year range:** {int(cleaned_litterfall["Year"].min())}–{
                    int(cleaned_litterfall["Year"].max())
                }
                """
            ),
        ]
    )
    return


@app.cell
def _(cleaned_litterfall, mo):
    litterfall_plot_options = sorted(
        cleaned_litterfall["LWF_plot_code"].drop_nulls().cast(str).unique().to_list()
    )

    litterfall_fraction_options = sorted(
        cleaned_litterfall["Text_Fraction_2"].drop_nulls().cast(str).unique().to_list()
    )

    litterfall_species_options = sorted(
        cleaned_litterfall["Tree_species"]
        .drop_nulls()
        .cast(str)
        .filter(cleaned_litterfall["Tree_species"].drop_nulls().cast(str).str.len_chars() > 0)
        .unique()
        .to_list()
    )

    litterfall_year_options = sorted(
        cleaned_litterfall["Year"].drop_nulls().cast(int).unique().to_list()
    )

    litterfall_plot_selector = mo.ui.dropdown(
        options=["All"] + litterfall_plot_options,
        value="All",
        label="LWF site",
    )

    litterfall_fraction_selector = mo.ui.dropdown(
        options=["All"] + litterfall_fraction_options,
        value="All",
        label="Litter fraction",
    )

    litterfall_species_selector = mo.ui.dropdown(
        options=["All"] + litterfall_species_options,
        value="All",
        label="Tree species",
    )

    litterfall_year_selector = mo.ui.dropdown(
        options=["All"] + litterfall_year_options,
        value="All",
        label="Year",
    )

    mo.vstack(
        [
            mo.md("### Filters"),
            mo.hstack(
                [
                    litterfall_plot_selector,
                    litterfall_fraction_selector,
                ]
            ),
            mo.hstack(
                [
                    litterfall_species_selector,
                    litterfall_year_selector,
                ]
            ),
        ]
    )
    return (
        litterfall_fraction_selector,
        litterfall_plot_selector,
        litterfall_species_selector,
        litterfall_year_selector,
    )


@app.cell
def _(
    cleaned_litterfall,
    litterfall_fraction_selector,
    litterfall_plot_selector,
    litterfall_species_selector,
    litterfall_year_selector,
    pl,
):
    filtered_litterfall = cleaned_litterfall.clone()

    if litterfall_plot_selector.value != "All":
        filtered_litterfall = filtered_litterfall.filter(
            pl.col("LWF_plot_code").cast(pl.String, strict=False) == litterfall_plot_selector.value
        )

    if litterfall_fraction_selector.value != "All":
        filtered_litterfall = filtered_litterfall.filter(
            pl.col("Text_Fraction_2").cast(pl.String, strict=False)
            == litterfall_fraction_selector.value
        )

    if litterfall_species_selector.value != "All":
        filtered_litterfall = filtered_litterfall.filter(
            pl.col("Tree_species").cast(pl.String, strict=False)
            == litterfall_species_selector.value
        )

    if litterfall_year_selector.value != "All":
        filtered_litterfall = filtered_litterfall.filter(
            pl.col("Year") == int(litterfall_year_selector.value)
        )

    filtered_litterfall = filtered_litterfall.sort(["Start_date", "LWF_plot_code"])

    filtered_litterfall  # noqa: B018 - Marimo displays the final cell expression.
    return (filtered_litterfall,)


@app.cell
def _(filtered_litterfall, mo, pl):
    litterfall_total_rows = len(filtered_litterfall)
    litterfall_valid_weight = filtered_litterfall["Weight"].is_not_null().sum()
    litterfall_missing_weight = filtered_litterfall["Weight"].is_null().sum()
    litterfall_missing_start = filtered_litterfall["Start_date"].is_null().sum()
    litterfall_missing_end = filtered_litterfall["End_date"].is_null().sum()

    litterfall_quality = pl.DataFrame(
        {
            "Quantity": [
                "Total observations",
                "Valid weight",
                "Missing weight (NaN)",
                "Missing start date (NaT)",
                "Missing end date (NaT)",
            ],
            "Count": [
                litterfall_total_rows,
                litterfall_valid_weight,
                litterfall_missing_weight,
                litterfall_missing_start,
                litterfall_missing_end,
            ],
        }
    )

    mo.vstack(
        [
            mo.md("### Data quality"),
            litterfall_quality,
        ]
    )
    return


@app.cell
def _(filtered_litterfall, pl):
    litterfall_missingness = pl.DataFrame(
        {
            "variable": filtered_litterfall.columns,
            "missing_count": [
                filtered_litterfall[column].null_count() for column in filtered_litterfall.columns
            ],
        }
    )

    litterfall_missingness = litterfall_missingness.with_columns(
        (100 * pl.col("missing_count") / max(len(filtered_litterfall), 1)).alias("missing_percent")
    )

    litterfall_missingness = litterfall_missingness.sort(
        ["missing_count", "variable"],
        descending=[True, False],
    )

    litterfall_missingness  # noqa: B018 - Marimo displays the final cell expression.
    return


@app.cell
def _(cleaned_litterfall, go, pl):
    # ------------------------------------------------------------
    # Seasonal pattern
    # ------------------------------------------------------------

    seasonal_data = cleaned_litterfall.drop_nulls(subset=["Month", "Weight"])

    # Exclude Total so that the individual fractions can be compared
    # without treating the aggregate Total as another component.
    # Also exclude "Total without branches >2cm" (matched as a
    # substring so wording/spacing differences don't slip through).
    seasonal_components = seasonal_data.filter(
        (pl.col("Text_Fraction_2") != "Total")
        & ~pl.col("Text_Fraction_2")
        .cast(pl.String, strict=False)
        .str.strip_chars()
        .str.to_lowercase()
        .str.contains("without branches", literal=True)
    )

    seasonal_figure = go.Figure()

    for fraction_name in sorted(seasonal_components["Text_Fraction_2"].unique().to_list()):
        fraction_data = seasonal_components.filter(pl.col("Text_Fraction_2") == fraction_name)

        grouped = fraction_data.group_by("Month").agg(pl.col("Weight").mean()).sort("Month")

        seasonal_figure.add_trace(
            go.Scatter(
                x=grouped["Month"],
                y=grouped["Weight"],
                mode="lines+markers",
                name=fraction_name,
                marker=dict(size=6),
                hovertemplate=(
                    "<b>Month:</b> %{x}<br><b>Mean weight:</b> %{y:.3f}<extra></extra>"
                ),
            )
        )

    seasonal_figure.update_layout(
        title="Seasonal pattern of litterfall components",
        xaxis_title="Calendar month",
        yaxis_title="Mean weight per monthly period",
        xaxis=dict(dtick=1),
        hovermode="x unified",
    )

    seasonal_figure  # noqa: B018 - Marimo displays the final cell expression.
    return


@app.cell
def _(cleaned_litterfall, go, mo, pl):
    # ------------------------------------------------------------
    # Distribution of sampling-period weights by fraction
    # ------------------------------------------------------------

    distribution_fraction_names = (
        cleaned_litterfall.filter(
            pl.col("Text_Fraction_2").is_not_null()
            & ~pl.col("Text_Fraction_2")
            .cast(pl.String, strict=False)
            .str.strip_chars()
            .str.to_lowercase()
            .str.contains("without branches", literal=True)
        )
        .select("Text_Fraction_2")
        .unique()
        .to_series()
        .to_list()
    )

    fraction_distribution_figure = go.Figure()

    for distribution_fraction_name in sorted(distribution_fraction_names):
        distribution_fraction_part = cleaned_litterfall.filter(
            pl.col("Text_Fraction_2") == distribution_fraction_name
        ).drop_nulls(subset=["Weight"])

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
                """
            ),
        ]
    )
    return


@app.cell
def _(filtered_litterfall, pl):
    litterfall_missing_weight_rows = filtered_litterfall.filter(pl.col("Weight").is_null())

    litterfall_missing_weight_rows  # noqa: B018 - Marimo displays the final cell expression.
    return


@app.cell
def _(filtered_litterfall):
    filtered_litterfall["Weight"].describe()
    return


@app.cell
def _(cleaned_litterfall, mo):
    # ------------------------------------------------------------
    # Compare LWF sites
    # ------------------------------------------------------------

    comparison_site_options = sorted(
        cleaned_litterfall["LWF_plot_code"].drop_nulls().cast(str).unique().to_list()
    )

    comparison_fraction_options = sorted(
        cleaned_litterfall["Text_Fraction_2"].drop_nulls().cast(str).unique().to_list()
    )

    comparison_default_fraction = (
        "Total"
        if "Total" in comparison_fraction_options
        else (comparison_fraction_options[0] if comparison_fraction_options else None)
    )

    comparison_site_selector = mo.ui.multiselect(
        options=comparison_site_options,
        value=comparison_site_options[:2],
        label="LWF sites to compare",
    )

    comparison_fraction_selector = mo.ui.dropdown(
        options=comparison_fraction_options,
        value=comparison_default_fraction,
        label="Litter fraction",
    )

    mo.vstack(
        [
            mo.md("### Compare LWF sites"),
            mo.hstack(
                [
                    comparison_site_selector,
                    comparison_fraction_selector,
                ]
            ),
        ]
    )
    return comparison_fraction_selector, comparison_site_selector


@app.cell
def _(
    cleaned_litterfall,
    comparison_fraction_selector,
    comparison_site_selector,
    pl,
):
    comparison_data_by_site = {
        comparison_site_id: (
            cleaned_litterfall.filter(
                (pl.col("LWF_plot_code").cast(pl.String, strict=False) == comparison_site_id)
                & (pl.col("Text_Fraction_2") == comparison_fraction_selector.value)
            )
            .drop_nulls(subset=["Start_date", "Weight"])
            .group_by("Start_date")
            .agg(pl.col("Weight").mean())
            .sort("Start_date")
        )
        for comparison_site_id in comparison_site_selector.value
    }
    return (comparison_data_by_site,)


@app.cell
def _(
    comparison_data_by_site,
    comparison_fraction_selector,
    comparison_site_selector,
    go,
    mo,
):
    # One line per selected LWF site, so any number of sites can be
    # compared at once for the chosen litter fraction.

    comparison_figure = go.Figure()

    for comparison_site_id, comparison_site_data in comparison_data_by_site.items():
        comparison_figure.add_trace(
            go.Scatter(
                x=comparison_site_data["Start_date"],
                y=comparison_site_data["Weight"],
                mode="lines+markers",
                name=comparison_site_id,
                marker=dict(size=6),
                hovertemplate=(
                    f"<b>{comparison_site_id}</b>"
                    "<br>Date: %{x|%d %B %Y}"
                    "<br>Mean weight: %{y:.3f} kg/ha"
                    "<extra></extra>"
                ),
            )
        )

    comparison_sites_label = (
        ", ".join(comparison_site_selector.value)
        if comparison_site_selector.value
        else "no sites selected"
    )

    comparison_figure.update_layout(
        title=(
            f"Litterfall comparison ({comparison_fraction_selector.value}): "
            f"{comparison_sites_label}"
        ),
        xaxis_title="Sampling period start",
        yaxis_title="Mean weight (kg/ha)",
        hovermode="x unified",
    )

    mo.vstack(
        [
            mo.md(f"### {comparison_sites_label}"),
            comparison_figure,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Key findings

    - Litterfall composition is strongly seasonal, with leaf and needle fractions generally increasing from spring into autumn.
    - Several fractions show their largest values around October.
    - Woody litter (Twigs, wood, bark) remains comparatively smaller.
    - Total litterfall varies substantially between sampling periods and sites.
    """)
    return


if __name__ == "__main__":
    app.run()
