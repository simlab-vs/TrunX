# ruff: noqa: E501
"""Marimo EDA for monthly deposition measurements at LWF sites."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    ### Dataset overview
    The monthly deposition dataset contains monthly precipitation and water-chemistry measurements across LWF sites and sampling locations. `plot_id` and `plot_name` identify the site, `survey_year` and `survey_month` identify the sampling month, and `location` identifies the sampling location. `precip` is precipitation, while the remaining variables describe conductivity, pH, alkalinity, major ions, and total N, P, S and dissolved organic carbon concentrations in the units shown by the variable selector. The plots show how these measurements vary over time and between selected locations.
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
    deposition_path = Path("data/monthly_dep_lwf_2026-07-28.csv")

    deposition = pl.read_csv(
        deposition_path,
        separator=";",
        schema_overrides={
            "alk": pl.Float64,
        },
        null_values="NA",
    )
    return (deposition,)


@app.cell
def _(deposition, mo, pl):
    null_percent = (
        deposition.null_count()
        .transpose(
            include_header=True,
            header_name="Column",
            column_names=["Null percentage"],
        )
        .with_columns((pl.col("Null percentage") * 100 / len(deposition)).round(2))
    )

    mo.ui.table(null_percent)
    return


@app.cell
def _(Path, pl):
    location_path = Path("data/plot_locations.xlsx")

    plot_locations = pl.read_excel(location_path)
    return (plot_locations,)


@app.cell
def _(deposition, pl, plot_locations):
    cleaned_deposition = deposition.clone()

    non_measurement_columns = [
        "plot_id",
        "plot_name",
        "survey_year",
        "survey_month",
        "location",
    ]

    measurement_columns = [
        column for column in cleaned_deposition.columns if column not in non_measurement_columns
    ]

    cleaned_deposition = cleaned_deposition.with_columns(
        [pl.col(column).cast(pl.Float64, strict=False) for column in measurement_columns]
    )

    cleaned_deposition = cleaned_deposition.with_columns(
        pl.concat_str(
            [
                pl.col("survey_year").cast(pl.Int64, strict=False).cast(pl.Utf8),
                pl.col("survey_month").cast(pl.Int64, strict=False).cast(pl.Utf8).str.zfill(2),
                pl.lit("01"),
            ],
            separator="-",
        )
        .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
        .alias("period_date")
    )

    if {"plot_id", "location"}.issubset(plot_locations.columns):
        cleaned_deposition = cleaned_deposition.join(
            plot_locations,
            on=["plot_id", "location"],
            how="left",
            suffix="_location",
        )
    return cleaned_deposition, measurement_columns


@app.cell
def _():
    variable_units = {
        "precip": "mm",
        "conductivity": "µS/cm (25°C)",
        "ph": "pH",
        "alk": "meq/L",
        "nh4_n": "mg N/L",
        "no3_n": "mg N/L",
        "so4_s": "mg S/L",
        "po4_p": "mg P/L",
        "ca": "mg/L",
        "mg": "mg/L",
        "k": "mg/L",
        "na": "mg/L",
        "cl": "mg/L",
        "t_p": "mg P/L",
        "t_s": "mg S/L",
        "t_n": "mg N/L",
        "toc": "mg C/L",
    }

    variable_labels = {
        "precip": "Precipitation",
        "conductivity": "Conductivity",
        "ph": "pH",
        "alk": "Alkalinity",
        "nh4_n": "Ammonium",
        "no3_n": "Nitrate",
        "so4_s": "Sulfate",
        "po4_p": "Phosphate",
        "ca": "Calcium",
        "mg": "Magnesium",
        "k": "Potassium",
        "na": "Sodium",
        "cl": "Chloride",
        "t_p": "Total phosphorus",
        "t_s": "Total sulphur",
        "t_n": "Total nitrogen",
        "toc": "Dissolved organic carbon",
    }
    return variable_labels, variable_units


@app.cell
def _(
    deposition,
    measurement_columns,
    mo,
    pl,
    variable_labels,
    variable_units,
):
    plot_options = sorted(deposition["plot_id"].drop_nulls().cast(pl.Utf8).unique().to_list())

    location_options = sorted(deposition["location"].drop_nulls().cast(pl.Utf8).unique().to_list())

    variable_options = {
        (
            f"{variable_labels.get(column, column)} ({variable_units.get(column, '')})"
            if variable_units.get(column, "")
            else variable_labels.get(column, column)
        ): column
        for column in measurement_columns
    }

    plot_selector = mo.ui.dropdown(
        options=["All"] + plot_options,
        value="All",
        label="LWF site",
    )

    location_selector = mo.ui.dropdown(
        options=["All"] + location_options,
        value="All",
        label="Location",
    )

    variable_selector = mo.ui.dropdown(
        options=variable_options,
        value=next(iter(variable_options)),
        label="Measurement",
    )

    mo.vstack(
        [
            mo.md("### Filters"),
            mo.hstack(
                [
                    plot_selector,
                    location_selector,
                    variable_selector,
                ]
            ),
        ]
    )
    return location_selector, plot_selector, variable_selector


@app.cell
def _(cleaned_deposition, location_selector, pl, plot_selector):
    filtered_deposition = cleaned_deposition.clone()

    if plot_selector.value != "All":
        filtered_deposition = filtered_deposition.filter(
            pl.col("plot_id").cast(pl.Utf8) == plot_selector.value
        )

    if location_selector.value != "All":
        filtered_deposition = filtered_deposition.filter(
            pl.col("location").cast(pl.Utf8) == location_selector.value
        )

    filtered_deposition = filtered_deposition.sort("period_date")
    return (filtered_deposition,)


@app.cell
def _(variable_labels, variable_selector, variable_units):
    selected_variable = variable_selector.value

    selected_label = variable_labels.get(
        selected_variable,
        selected_variable,
    )

    selected_unit = variable_units.get(
        selected_variable,
        "",
    )
    return selected_label, selected_unit, selected_variable


@app.cell
def _(
    filtered_deposition,
    mo,
    pl,
    selected_label,
    selected_unit,
    selected_variable,
):
    valid_values = filtered_deposition[selected_variable].is_not_null().sum()

    missing_values = filtered_deposition[selected_variable].is_null().sum()

    missing_dates = filtered_deposition["period_date"].is_null().sum()

    total_rows = len(filtered_deposition)

    summary = pl.DataFrame(
        {
            "Quantity": [
                "Total observations",
                "Valid measurements",
                "Missing measurement (NaN)",
                "Missing/invalid time (NaT)",
            ],
            "Count": [
                total_rows,
                valid_values,
                missing_values,
                missing_dates,
            ],
        }
    )

    mo.vstack(
        [
            mo.md(
                f"""
    ### Data quality

    **Measurement:** {selected_label}

    **Unit:** {selected_unit}
    """
            ),
            summary,
        ]
    )
    return


@app.cell
def _(filtered_deposition, measurement_columns, pl):
    missingness = pl.DataFrame(
        {
            "variable": measurement_columns,
            "missing_count": [
                filtered_deposition[column].is_null().sum() for column in measurement_columns
            ],
        }
    )

    missingness = missingness.with_columns(
        (100 * pl.col("missing_count") / max(len(filtered_deposition), 1)).alias("missing_percent")
    )

    missingness = missingness.sort("missing_count", descending=True)

    missingness  # noqa: B018 - Marimo displays the final cell expression.
    return


@app.cell
def _(
    filtered_deposition,
    go,
    pl,
    selected_label,
    selected_unit,
    selected_variable,
):
    valid_plot = (
        filtered_deposition.select(["period_date", selected_variable])
        .drop_nulls(subset=["period_date", selected_variable])
        .sort("period_date")
    )

    missing_measurement = (
        filtered_deposition.select(["period_date", selected_variable])
        .filter(pl.col("period_date").is_not_null() & pl.col(selected_variable).is_null())
        .sort("period_date")
    )

    missing_period_date = filtered_deposition.filter(pl.col("period_date").is_null())

    fig = go.Figure()

    # --------------------------------------------------------
    # Valid measurements
    # --------------------------------------------------------

    if not valid_plot.is_empty():
        fig.add_trace(
            go.Scatter(
                x=valid_plot["period_date"],
                y=valid_plot[selected_variable],
                mode="markers",
                name="Measured",
                marker={"size": 7},
                hovertemplate=(
                    "<b>Date:</b> %{x|%d %B %Y}"
                    f"<br><b>{selected_label}:</b> "
                    "%{y:.3f}"
                    f" {selected_unit}"
                    "<extra></extra>"
                ),
            )
        )

    # --------------------------------------------------------
    # Missing measurement values
    # --------------------------------------------------------

    if not missing_measurement.is_empty():
        if not valid_plot.is_empty():
            y_min = valid_plot[selected_variable].min()
            y_max = valid_plot[selected_variable].max()
            y_range = y_max - y_min

            if y_range == 0:
                y_range = max(abs(y_min) * 0.1, 1.0)

            missing_marker_y = y_min - 0.08 * y_range

        else:
            missing_marker_y = 0

        fig.add_trace(
            go.Scatter(
                x=missing_measurement["period_date"],
                y=[missing_marker_y] * len(missing_measurement),
                mode="markers",
                name="Missing measurement (NaN)",
                marker={
                    "size": 10,
                    "symbol": "x",
                },
                hovertemplate=(
                    f"<b>Date:</b> %{{x|%d %B %Y}}<br><b>{selected_label}:</b> NaN<extra></extra>"
                ),
            )
        )

    # --------------------------------------------------------
    # Missing dates
    # --------------------------------------------------------

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1,
        y=1.08,
        xanchor="right",
        yanchor="bottom",
        text=(
            f"Missing measurement (NaN): "
            f"{len(missing_measurement)}"
            f" | Missing time (NaT): "
            f"{len(missing_period_date)}"
        ),
        showarrow=False,
    )

    # --------------------------------------------------------
    # Plot layout
    # --------------------------------------------------------

    fig.update_layout(
        title=f"{selected_label} over time",
        xaxis_title="Sampling month",
        yaxis_title=(f"{selected_label} ({selected_unit})" if selected_unit else selected_label),
        hovermode="closest",
    )

    fig  # noqa: B018 - Marimo displays the final cell expression.
    return


@app.cell
def _(
    cleaned_deposition,
    measurement_columns,
    mo,
    pl,
    variable_labels,
    variable_units,
):
    # --------------------------------------------------------

    # Compare two LWF plots

    # --------------------------------------------------------

    comparison_plot_options = sorted(
        cleaned_deposition["plot_id"].drop_nulls().cast(pl.Utf8).unique().to_list()
    )

    comparison_variable_options = {
        (
            f"{variable_labels.get(column, column)} ({variable_units.get(column, '')})"
            if variable_units.get(column, "")
            else variable_labels.get(column, column)
        ): column
        for column in measurement_columns
    }

    comparison_plots = mo.ui.multiselect(
        options=comparison_plot_options,
        value=comparison_plot_options[: min(2, len(comparison_plot_options))],
        label="LWF plots",
    )

    comparison_variable = mo.ui.dropdown(
        options=comparison_variable_options,
        value=next(iter(comparison_variable_options)),
        label="Measurement",
    )

    mo.vstack(
        [
            mo.md("## Compare multiple LWF plots"),
            mo.hstack(
                [
                    comparison_plots,
                    comparison_variable,
                ]
            ),
        ]
    )
    return comparison_plots, comparison_variable


@app.cell
def _(
    cleaned_deposition,
    comparison_plots,
    comparison_variable,
    go,
    mo,
    pl,
    variable_labels,
    variable_units,
):
    variable = comparison_variable.value

    label = variable_labels.get(variable, variable)
    unit = variable_units.get(variable, "")

    if len(comparison_plots.value) == 0:
        fig_compare = mo.md("*Select at least one LWF plot above to see the comparison.*")

    else:
        fig_compare = go.Figure()

        for plot_id in comparison_plots.value:
            data = cleaned_deposition.filter(pl.col("plot_id").cast(pl.Utf8) == plot_id).select(
                ["period_date", variable]
            )

            data = data.drop_nulls(subset=["period_date"]).sort("period_date")

            fig_compare.add_trace(
                go.Scatter(
                    x=data["period_date"],
                    y=data[variable],
                    mode="lines+markers",
                    name=plot_id,
                    hovertemplate=(
                        "<b>Plot:</b> "
                        + plot_id
                        + "<br><b>Date:</b> %{x|%d %B %Y}"
                        + f"<br><b>{label}:</b> %{{y:.3f}}"
                        + (f" {unit}" if unit else "")
                        + "<extra></extra>"
                    ),
                )
            )

        fig_compare.update_layout(
            title=f"{label}: " + " vs ".join(comparison_plots.value),
            xaxis_title="Sampling month",
            yaxis_title=(f"{label} ({unit})" if unit else label),
            hovermode="x unified",
            legend_title="LWF plot",
        )

    fig_compare  # noqa: B018 - Marimo displays the final cell expression.
    return


@app.cell
def _():
    from pathlib import Path as loccmp_Path

    import marimo as loccmp_mo
    import plotly.graph_objects as loccmp_go
    import polars as loccmp_pl

    return loccmp_Path, loccmp_go, loccmp_mo, loccmp_pl


@app.cell
def _(loccmp_Path, loccmp_pl):
    loccmp_source_path = loccmp_Path("data/monthly_dep_lwf_2026-07-28.csv")

    loccmp_source_data = loccmp_pl.read_csv(
        loccmp_source_path,
        separator=";",
        schema_overrides={
            "alk": loccmp_pl.Float64,
        },
        null_values="NA",
    )

    loccmp_measurement_columns = [
        loccmp_column
        for loccmp_column in loccmp_source_data.columns
        if loccmp_column
        not in {
            "plot_id",
            "plot_name",
            "survey_year",
            "survey_month",
            "location",
        }
    ]

    loccmp_source_data = loccmp_source_data.with_columns(
        [
            loccmp_pl.col(loccmp_column).cast(loccmp_pl.Float64, strict=False)
            for loccmp_column in loccmp_measurement_columns
        ]
    )

    loccmp_source_data = loccmp_source_data.with_columns(
        loccmp_pl.concat_str(
            [
                loccmp_pl.col("survey_year")
                .cast(loccmp_pl.Int64, strict=False)
                .cast(loccmp_pl.Utf8),
                loccmp_pl.col("survey_month")
                .cast(loccmp_pl.Int64, strict=False)
                .cast(loccmp_pl.Utf8)
                .str.zfill(2),
                loccmp_pl.lit("01"),
            ],
            separator="-",
        )
        .str.strptime(loccmp_pl.Date, format="%Y-%m-%d", strict=False)
        .alias("comparison_date")
    )
    return loccmp_measurement_columns, loccmp_source_data


@app.cell
def _(loccmp_measurement_columns, loccmp_mo, loccmp_pl, loccmp_source_data):
    loccmp_site_codes = sorted(
        loccmp_source_data["plot_id"].drop_nulls().cast(loccmp_pl.Utf8).unique().to_list()
    )

    loccmp_measurement_options = {
        "Precipitation (mm)": "precip",
        "Conductivity (µS/cm (25°C))": "conductivity",
        "pH (pH)": "ph",
        "Alkalinity (meq/L)": "alk",
        "Ammonium (mg N/L)": "nh4_n",
        "Nitrate (mg N/L)": "no3_n",
        "Sulfate (mg S/L)": "so4_s",
        "Phosphate (mg P/L)": "po4_p",
        "Calcium (mg/L)": "ca",
        "Magnesium (mg/L)": "mg",
        "Potassium (mg/L)": "k",
        "Sodium (mg/L)": "na",
        "Chloride (mg/L)": "cl",
        "Total phosphorus (mg P/L)": "t_p",
        "Total sulphur (mg S/L)": "t_s",
        "Total nitrogen (mg N/L)": "t_n",
        "Dissolved organic carbon (mg C/L)": "toc",
    }

    loccmp_measurement_options = {
        loccmp_label: loccmp_column
        for loccmp_label, loccmp_column in loccmp_measurement_options.items()
        if loccmp_column in loccmp_measurement_columns
    }

    loccmp_site_dropdown = loccmp_mo.ui.dropdown(
        options=loccmp_site_codes,
        value=loccmp_site_codes[0],
        label="LWF site",
    )

    loccmp_measurement_dropdown = loccmp_mo.ui.dropdown(
        options=loccmp_measurement_options,
        value=next(iter(loccmp_measurement_options)),
        label="Measurement",
    )
    return (
        loccmp_measurement_dropdown,
        loccmp_measurement_options,
        loccmp_site_dropdown,
    )


@app.cell
def _(
    loccmp_measurement_dropdown,
    loccmp_mo,
    loccmp_pl,
    loccmp_site_dropdown,
    loccmp_source_data,
):
    loccmp_selected_site = loccmp_site_dropdown.value

    loccmp_location_codes = sorted(
        loccmp_source_data.filter(
            loccmp_pl.col("plot_id").cast(loccmp_pl.Utf8) == loccmp_selected_site
        )
        .get_column("location")
        .drop_nulls()
        .cast(loccmp_pl.Utf8)
        .unique()
        .to_list()
    )

    if not loccmp_location_codes:
        loccmp_location_codes = ["No locations available"]

    loccmp_location_a_dropdown = loccmp_mo.ui.dropdown(
        options=loccmp_location_codes,
        value=loccmp_location_codes[0],
        label="Location A",
    )

    loccmp_location_b_dropdown = loccmp_mo.ui.dropdown(
        options=loccmp_location_codes,
        value=(
            loccmp_location_codes[1]
            if len(loccmp_location_codes) > 1
            else loccmp_location_codes[0]
        ),
        label="Location B",
    )

    loccmp_mo.vstack(
        [
            loccmp_mo.md("## Compare two deposition locations"),
            loccmp_mo.md(
                "Select an LWF site and any two locations available "
                "at that site, such as B vs Ba, B vs F, or Ba vs Bb."
            ),
            loccmp_mo.hstack(
                [
                    loccmp_site_dropdown,
                    loccmp_location_a_dropdown,
                    loccmp_location_b_dropdown,
                    loccmp_measurement_dropdown,
                ]
            ),
        ]
    )
    return (
        loccmp_location_a_dropdown,
        loccmp_location_b_dropdown,
        loccmp_selected_site,
    )


@app.cell
def _(
    loccmp_go,
    loccmp_location_a_dropdown,
    loccmp_location_b_dropdown,
    loccmp_measurement_dropdown,
    loccmp_measurement_options,
    loccmp_pl,
    loccmp_selected_site,
    loccmp_source_data,
):
    loccmp_location_a = loccmp_location_a_dropdown.value
    loccmp_location_b = loccmp_location_b_dropdown.value
    loccmp_measurement_column = loccmp_measurement_dropdown.value

    loccmp_a_data = (
        loccmp_source_data.filter(
            (loccmp_pl.col("plot_id").cast(loccmp_pl.Utf8) == loccmp_selected_site)
            & (loccmp_pl.col("location").cast(loccmp_pl.Utf8) == loccmp_location_a)
        )
        .select(["comparison_date", loccmp_measurement_column])
        .drop_nulls(subset=["comparison_date"])
        .sort("comparison_date")
    )

    loccmp_b_data = (
        loccmp_source_data.filter(
            (loccmp_pl.col("plot_id").cast(loccmp_pl.Utf8) == loccmp_selected_site)
            & (loccmp_pl.col("location").cast(loccmp_pl.Utf8) == loccmp_location_b)
        )
        .select(["comparison_date", loccmp_measurement_column])
        .drop_nulls(subset=["comparison_date"])
        .sort("comparison_date")
    )

    loccmp_measurement_label = next(
        (
            loccmp_label
            for loccmp_label, loccmp_column in loccmp_measurement_options.items()
            if loccmp_column == loccmp_measurement_column
        ),
        loccmp_measurement_column,
    )

    loccmp_location_comparison_figure = loccmp_go.Figure()

    for loccmp_frame, loccmp_name in (
        (loccmp_a_data, loccmp_location_a),
        (loccmp_b_data, loccmp_location_b),
    ):
        loccmp_location_comparison_figure.add_trace(
            loccmp_go.Scatter(
                x=loccmp_frame["comparison_date"],
                y=loccmp_frame[loccmp_measurement_column],
                mode="lines+markers",
                name=loccmp_name,
                marker={"size": 7},
                hovertemplate=(
                    f"<b>Site:</b> {loccmp_selected_site}"
                    f"<br><b>Location:</b> {loccmp_name}"
                    "<br><b>Date:</b> %{x|%d %B %Y}"
                    f"<br><b>{loccmp_measurement_label}:</b> "
                    "%{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    loccmp_location_comparison_figure.update_layout(
        title=(
            f"{loccmp_selected_site}: "
            f"{loccmp_location_a} vs {loccmp_location_b} — "
            f"{loccmp_measurement_label}"
        ),
        xaxis_title="Sampling month",
        yaxis_title=loccmp_measurement_label,
        hovermode="x unified",
        legend_title="Location",
    )

    loccmp_location_comparison_figure  # noqa: B018 - Marimo displays the final cell expression.
    return


@app.cell
def _(mo):
    mo.md("""
    ### Key findings

    - Precipitation and deposition chemistry show substantial month-to-month variability.
    - Pronounced peaks occur in some months rather than a smooth long-term trend.
    - Forest location B (forest stand) generally has higher conductivity, magnesium, alkalinity, and total sulphur concentrations than open location F (open area).
    - pH values are more similar between locations B and F.
    """)
    return


if __name__ == "__main__":
    app.run()
