# ruff: noqa: E501
"""Marimo EDA for periodic deposition measurements at LWF sites."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    ### Dataset overview
    The periodic deposition dataset contains water-chemistry measurements collected over sampling periods at LWF sites and locations. `plot_id` and `plot_name` identify the site, `date_start` and `date_end` define each sampling period, and `location` identifies the sampling location. `precip` records precipitation, while the remaining variables describe conductivity, pH, alkalinity, major ions, total nitrogen, phosphorus and sulphur, and dissolved organic carbon in the units shown by the variable selector. The plots show variation across sampling periods and allow comparisons between locations.
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
    deposition_path = Path("data/period_dep_lwf_2026-07-28.csv")

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
def _(deposition):
    deposition  # noqa: B018


@app.cell
def _(deposition, mo, pl):
    null_percent = (
        (deposition.null_count() / len(deposition) * 100)
        .transpose(
            include_header=True,
            header_name="Column",
            column_names=["Null percentage"],
        )
        .with_columns(pl.col("Null percentage").round(2))
    )

    mo.ui.table(null_percent)


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
        "date_start",
        "date_end",
        "location",
    ]

    measurement_columns = [
        column for column in cleaned_deposition.columns if column not in non_measurement_columns
    ]

    for column in measurement_columns:
        cleaned_deposition = cleaned_deposition.with_columns(
            pl.col(column).cast(pl.Float64, strict=False)
        )

    cleaned_deposition = cleaned_deposition.with_columns(
        pl.col("date_start").str.to_datetime(strict=False).alias("period_date")
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

    missingness  # noqa: B018


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
                x=valid_plot["period_date"].to_list(),
                y=valid_plot[selected_variable].to_list(),
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
                x=missing_measurement["period_date"].to_list(),
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
            "Missing measurement (NaN): "
            f"{len(missing_measurement)}"
            " | Missing time (NaT): "
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

    fig  # noqa: B018


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
    # Compare two LWF plots/sites
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
        fig_plot_compare = mo.md("*Select at least one LWF plot above to see the comparison.*")

    else:
        fig_plot_compare = go.Figure()

        for plot_id in comparison_plots.value:
            data = cleaned_deposition.filter(pl.col("plot_id").cast(pl.Utf8) == plot_id).select(
                ["period_date", variable]
            )

            data = data.drop_nulls(subset=["period_date"]).sort("period_date")

            fig_plot_compare.add_trace(
                go.Scatter(
                    x=data["period_date"].to_list(),
                    y=data[variable].to_list(),
                    mode="lines+markers",
                    name=plot_id,
                    hovertemplate=(
                        f"<b>Plot:</b> {plot_id}"
                        "<br><b>Date:</b> %{x|%d %B %Y}"
                        f"<br><b>{label}:</b> %{{y:.3f}}"
                        f"{(' ' + unit) if unit else ''}"
                        "<extra></extra>"
                    ),
                )
            )

        fig_plot_compare.update_layout(
            title=f"{label}: " + " vs ".join(comparison_plots.value),
            xaxis_title="Sampling date",
            yaxis_title=f"{label} ({unit})" if unit else label,
            hovermode="x unified",
            legend_title="LWF plot",
        )

    fig_plot_compare  # noqa: B018


@app.cell
def _():
    from pathlib import Path as perloccmp_Path

    import marimo as perloccmp_mo
    import plotly.graph_objects as perloccmp_go
    import polars as perloccmp_pl

    return perloccmp_Path, perloccmp_go, perloccmp_mo, perloccmp_pl


@app.cell
def _(perloccmp_Path, perloccmp_pl):
    perloccmp_source_path = perloccmp_Path("data/period_dep_lwf_2026-07-28.csv")

    perloccmp_source_data = perloccmp_pl.read_csv(
        perloccmp_source_path,
        separator=";",
        schema_overrides={
            "alk": perloccmp_pl.Float64,
        },
        null_values="NA",
    )

    perloccmp_measurement_columns = [
        perloccmp_column
        for perloccmp_column in perloccmp_source_data.columns
        if perloccmp_column
        not in {
            "plot_id",
            "plot_name",
            "date_start",
            "date_end",
            "location",
        }
    ]

    for perloccmp_column in perloccmp_measurement_columns:
        perloccmp_source_data = perloccmp_source_data.with_columns(
            perloccmp_pl.col(perloccmp_column).cast(perloccmp_pl.Float64, strict=False)
        )

    perloccmp_source_data = perloccmp_source_data.with_columns(
        perloccmp_pl.col("date_start").str.to_datetime(strict=False).alias("comparison_date")
    )
    return perloccmp_measurement_columns, perloccmp_source_data


@app.cell
def _(
    perloccmp_measurement_columns,
    perloccmp_mo,
    perloccmp_pl,
    perloccmp_source_data,
):
    perloccmp_site_codes = sorted(
        perloccmp_source_data["plot_id"].drop_nulls().cast(perloccmp_pl.Utf8).unique().to_list()
    )

    perloccmp_measurement_options = {
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

    perloccmp_measurement_options = {
        perloccmp_label: perloccmp_column
        for perloccmp_label, perloccmp_column in perloccmp_measurement_options.items()
        if perloccmp_column in perloccmp_measurement_columns
    }

    perloccmp_site_dropdown = perloccmp_mo.ui.dropdown(
        options=perloccmp_site_codes,
        value=perloccmp_site_codes[0],
        label="LWF site",
    )

    perloccmp_measurement_dropdown = perloccmp_mo.ui.dropdown(
        options=perloccmp_measurement_options,
        value=next(iter(perloccmp_measurement_options)),
        label="Measurement",
    )
    return (
        perloccmp_measurement_dropdown,
        perloccmp_measurement_options,
        perloccmp_site_dropdown,
    )


@app.cell
def _(
    perloccmp_measurement_dropdown,
    perloccmp_mo,
    perloccmp_pl,
    perloccmp_site_dropdown,
    perloccmp_source_data,
):
    perloccmp_selected_site = perloccmp_site_dropdown.value

    perloccmp_location_codes = sorted(
        perloccmp_source_data.filter(
            perloccmp_pl.col("plot_id").cast(perloccmp_pl.Utf8) == perloccmp_selected_site
        )["location"]
        .drop_nulls()
        .cast(perloccmp_pl.Utf8)
        .unique()
        .to_list()
    )

    if not perloccmp_location_codes:
        perloccmp_location_codes = ["No locations available"]

    perloccmp_location_a_dropdown = perloccmp_mo.ui.dropdown(
        options=perloccmp_location_codes,
        value=perloccmp_location_codes[0],
        label="Location A",
    )

    perloccmp_location_b_dropdown = perloccmp_mo.ui.dropdown(
        options=perloccmp_location_codes,
        value=(
            perloccmp_location_codes[1]
            if len(perloccmp_location_codes) > 1
            else perloccmp_location_codes[0]
        ),
        label="Location B",
    )

    perloccmp_mo.vstack(
        [
            perloccmp_mo.md("## Compare two deposition locations"),
            perloccmp_mo.md(
                "Choose an LWF site, then compare any two locations "
                "available there, such as B vs Ba, B vs F, or Ba vs Bb."
            ),
            perloccmp_mo.hstack(
                [
                    perloccmp_site_dropdown,
                    perloccmp_location_a_dropdown,
                    perloccmp_location_b_dropdown,
                    perloccmp_measurement_dropdown,
                ]
            ),
        ]
    )
    return (
        perloccmp_location_a_dropdown,
        perloccmp_location_b_dropdown,
        perloccmp_selected_site,
    )


@app.cell
def _(
    perloccmp_go,
    perloccmp_location_a_dropdown,
    perloccmp_location_b_dropdown,
    perloccmp_measurement_dropdown,
    perloccmp_measurement_options,
    perloccmp_pl,
    perloccmp_selected_site,
    perloccmp_source_data,
):
    perloccmp_location_a = perloccmp_location_a_dropdown.value
    perloccmp_location_b = perloccmp_location_b_dropdown.value
    perloccmp_measurement_column = perloccmp_measurement_dropdown.value

    perloccmp_a_data = (
        perloccmp_source_data.filter(
            (perloccmp_pl.col("plot_id").cast(perloccmp_pl.Utf8) == perloccmp_selected_site)
            & (perloccmp_pl.col("location").cast(perloccmp_pl.Utf8) == perloccmp_location_a)
        )
        .select(["comparison_date", perloccmp_measurement_column])
        .drop_nulls(subset=["comparison_date"])
        .sort("comparison_date")
    )

    perloccmp_b_data = (
        perloccmp_source_data.filter(
            (perloccmp_pl.col("plot_id").cast(perloccmp_pl.Utf8) == perloccmp_selected_site)
            & (perloccmp_pl.col("location").cast(perloccmp_pl.Utf8) == perloccmp_location_b)
        )
        .select(["comparison_date", perloccmp_measurement_column])
        .drop_nulls(subset=["comparison_date"])
        .sort("comparison_date")
    )

    perloccmp_measurement_label = next(
        (
            perloccmp_label
            for perloccmp_label, perloccmp_column in perloccmp_measurement_options.items()
            if perloccmp_column == perloccmp_measurement_column
        ),
        perloccmp_measurement_column,
    )

    perloccmp_figure = perloccmp_go.Figure()

    for perloccmp_frame, perloccmp_name in (
        (perloccmp_a_data, perloccmp_location_a),
        (perloccmp_b_data, perloccmp_location_b),
    ):
        perloccmp_figure.add_trace(
            perloccmp_go.Scatter(
                x=perloccmp_frame["comparison_date"].to_list(),
                y=perloccmp_frame[perloccmp_measurement_column].to_list(),
                mode="lines+markers",
                name=perloccmp_name,
                marker={"size": 7},
                hovertemplate=(
                    f"<b>Site:</b> {perloccmp_selected_site}"
                    f"<br><b>Location:</b> {perloccmp_name}"
                    "<br><b>Date:</b> %{x|%d %B %Y}"
                    f"<br><b>{perloccmp_measurement_label}:</b> "
                    "%{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    perloccmp_figure.update_layout(
        title=(
            f"{perloccmp_selected_site}: "
            f"{perloccmp_location_a} vs {perloccmp_location_b} — "
            f"{perloccmp_measurement_label}"
        ),
        xaxis_title="Sampling period start date",
        yaxis_title=perloccmp_measurement_label,
        hovermode="x unified",
        legend_title="Location",
    )

    perloccmp_figure  # noqa: B018


@app.cell
def _(mo):
    mo.md("""
    ### Key findings

    - Deposition chemistry varies considerably between sampling periods.
    - Forest and open locations show clear differences in several chemical variables.
    - pH is generally more similar between forest and open locations.
    - Unlike the monthly dataset, each observation represents a defined sampling period rather than a calendar month.
    """)


if __name__ == "__main__":
    app.run()
