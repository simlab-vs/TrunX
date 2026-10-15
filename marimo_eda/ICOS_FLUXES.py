import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Data for Half Hourly Fluxes CH-Dav Jan 2019 - Aug 2026
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import plotly.graph_objects as go
    import polars as pl

    return go, pl


@app.cell
def _(pl):
    from pathlib import Path

    file_path = Path("data/raw/ICOS/ICOSETC_CH-Dav_FLUXES_INTERIM_L2.csv")

    flux = pl.read_csv(file_path, infer_schema_length=100000, null_values="-9999")
    return (flux,)


@app.cell
def _(flux):
    flux
    return


@app.cell
def _(flux):
    flux.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Link for Column name Metadata: https://www.swissfluxnet.ethz.ch/index.php/data/variables/variable-abbreviations/
    """)
    return


@app.cell
def _(flux, mo, pl):
    column_table = pl.DataFrame(
        {
            "Column": flux.columns,
        }
    )

    mo.vstack(
        [
            mo.md("## Dataset columns"),
            mo.md(f"**Total columns:** {len(flux.columns):,}"),
            column_table,
        ]
    )
    return


@app.cell
def _(flux, mo, pl):
    row_count = max(len(flux), 1)

    null_percent = pl.DataFrame(
        {
            "Column": flux.columns,
            "Null percentage": [
                round(
                    100 * flux[column].is_null().sum() / row_count,
                    2,
                )
                for column in flux.columns
            ],
        }
    ).sort(
        ["Null percentage", "Column"],
        descending=[True, False],
    )

    mo.vstack(
        [
            mo.md("## Missing values"),
            null_percent,
        ]
    )
    null_percent
    return


@app.cell
def _(flux, pl):
    # ------------------------------------------------------------

    # Prepare data

    # ------------------------------------------------------------

    cleaned_flux = flux.with_columns(
        pl.col("TIMESTAMP_START")
        .cast(pl.String)
        .str.strptime(
            pl.Datetime,
            "%Y%m%d%H%M",
            strict=False,
        )
        .alias("TIMESTAMP_START"),
        pl.col("TIMESTAMP_END")
        .cast(pl.String)
        .str.strptime(
            pl.Datetime,
            "%Y%m%d%H%M",
            strict=False,
        )
        .alias("TIMESTAMP_END"),
    )
    return (cleaned_flux,)


@app.cell
def _(cleaned_flux, mo):
    # ------------------------------------------------------------

    # Dataset overview

    # ------------------------------------------------------------

    total_rows = len(cleaned_flux)

    start_date = cleaned_flux["TIMESTAMP_START"].min()

    end_date = cleaned_flux["TIMESTAMP_END"].max()

    total_columns = len(cleaned_flux.columns)

    valid_fc = cleaned_flux["FC"].is_not_null().sum()

    missing_fc = cleaned_flux["FC"].is_null().sum()

    valid_nee = cleaned_flux["NEE"].is_not_null().sum()

    missing_nee = cleaned_flux["NEE"].is_null().sum()

    valid_co2 = cleaned_flux["CO2"].is_not_null().sum()

    missing_co2 = cleaned_flux["CO2"].is_null().sum()

    mo.vstack(
        [
            mo.md(
                """

                # CH-Dav ICOS — FLUXES


                This notebook explores the complete ICOS Level 2

                FLUXES dataset for the CH-Dav site.


                All source columns are retained. Missing values represented

                by `-9999` in the source file are treated as missing values.

                """
            ),
            mo.hstack(
                [
                    mo.stat(
                        label="Site",
                        value="CH-Dav",
                    ),
                    mo.stat(
                        label="Rows",
                        value=f"{total_rows:,}",
                    ),
                    mo.stat(
                        label="Columns",
                        value=f"{total_columns:,}",
                    ),
                    mo.stat(
                        label="Valid FC",
                        value=f"{valid_fc:,}",
                    ),
                    mo.stat(
                        label="Valid NEE",
                        value=f"{valid_nee:,}",
                    ),
                    mo.stat(
                        label="Valid CO₂",
                        value=f"{valid_co2:,}",
                    ),
                ]
            ),
            mo.md(
                f"""

                **Coverage:** {start_date} to {end_date}


                **Missing FC:** {missing_fc:,}


                **Missing NEE:** {missing_nee:,}


                **Missing CO₂:** {missing_co2:,}

                """
            ),
        ]
    )
    return


@app.cell
def _(cleaned_flux):
    cleaned_flux
    return


@app.cell
def _(cleaned_flux, mo):

    year_options = sorted(
        cleaned_flux["TIMESTAMP_START"].drop_nulls().dt.year().unique().to_list()
    )

    month_options = sorted(
        cleaned_flux["TIMESTAMP_START"].drop_nulls().dt.month().unique().to_list()
    )

    variable_options = [
        column
        for column in cleaned_flux.columns
        if column
        not in {
            "TIMESTAMP_START",
            "TIMESTAMP_END",
        }
    ]

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

    variable_selector = mo.ui.dropdown(
        options=variable_options,
        value="FC" if "FC" in variable_options else variable_options[0],
        label="Variable",
    )

    mo.vstack(
        [
            mo.md("## Explore the data"),
            mo.hstack(
                [
                    year_selector,
                    month_selector,
                    variable_selector,
                ]
            ),
        ]
    )
    return month_selector, variable_selector, year_selector


@app.cell
def _(cleaned_flux, month_selector, pl, year_selector):

    filtered_flux = cleaned_flux

    if year_selector.value != "All":
        filtered_flux = filtered_flux.filter(
            pl.col("TIMESTAMP_START").dt.year() == int(year_selector.value)
        )

    if month_selector.value != "All":
        filtered_flux = filtered_flux.filter(
            pl.col("TIMESTAMP_START").dt.month() == int(month_selector.value)
        )

    filtered_flux = filtered_flux.sort(["TIMESTAMP_START", "TIMESTAMP_END"])
    return (filtered_flux,)


@app.cell
def _(filtered_flux, mo, pl):

    overview = pl.DataFrame(
        {
            "Quantity": [
                "Total observations",
                "Total variables",
                "Missing values",
                "Complete rows",
                "Missing start timestamps",
                "Missing end timestamps",
            ],
            "Count": [
                len(filtered_flux),
                len(filtered_flux.columns),
                filtered_flux.null_count().sum_horizontal()[0],
                filtered_flux.drop_nulls().height,
                filtered_flux["TIMESTAMP_START"].is_null().sum(),
                filtered_flux["TIMESTAMP_END"].is_null().sum(),
            ],
        }
    )

    mo.vstack(
        [
            mo.md("## Data quality"),
            overview,
        ]
    )
    return


@app.cell
def _(cleaned_flux, filtered_flux, mo, pl, variable_selector):
    # ------------------------------------------------------------
    # Selected variable information
    # ------------------------------------------------------------

    selected_variable = variable_selector.value

    dtype = cleaned_flux.schema[selected_variable]

    total_count = len(filtered_flux)

    missing_count = filtered_flux[selected_variable].is_null().sum()

    valid_count = total_count - missing_count

    unique_count = filtered_flux[selected_variable].drop_nulls().n_unique()

    variable_info = pl.DataFrame(
        {
            "Property": [
                "Variable",
                "Data type",
                "Observations",
                "Valid observations",
                "Missing observations",
                "Missing percentage",
                "Unique values",
            ],
            "Value": [
                str(selected_variable),
                str(dtype),
                str(total_count),
                str(valid_count),
                str(missing_count),
                str(
                    round(
                        100 * missing_count / max(total_count, 1),
                        2,
                    )
                ),
                str(unique_count),
            ],
        }
    )

    mo.vstack(
        [
            mo.md(f"### Selected variable: `{selected_variable}`"),
            variable_info,
        ]
    )
    return (selected_variable,)


@app.cell
def _(filtered_flux, go, mo, pl, selected_variable):
    # ------------------------------------------------------------
    # Main time-series view
    # ------------------------------------------------------------

    valid_series = filtered_flux.filter(
        pl.col("TIMESTAMP_START").is_not_null() & pl.col(selected_variable).is_not_null()
    ).sort("TIMESTAMP_START")

    missing_series = filtered_flux.filter(
        pl.col("TIMESTAMP_START").is_not_null() & pl.col(selected_variable).is_null()
    ).sort("TIMESTAMP_START")

    # Downsample only for plotting.

    # The full filtered_flux dataset remains unchanged.

    max_points = 10_000

    if len(valid_series) > max_points:
        step = max(1, len(valid_series) // max_points)

        plot_series = valid_series[::step]

    else:
        plot_series = valid_series

    figure = go.Figure()

    if len(plot_series) > 0:
        figure.add_trace(
            go.Scattergl(
                x=plot_series["TIMESTAMP_START"],
                y=plot_series[selected_variable],
                mode="lines",
                name=selected_variable,
                customdata=plot_series[["TIMESTAMP_START", "TIMESTAMP_END"]],
                hovertemplate=(
                    "<b>Start:</b> "
                    "%{customdata[0]|%d %B %Y %H:%M}"
                    "<br><b>End:</b> "
                    "%{customdata[1]|%d %B %Y %H:%M}"
                    f"<br><b>{selected_variable}:</b> "
                    "%{y}"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=f"{selected_variable} over time — CH-Dav",
        xaxis_title="Time",
        yaxis_title=selected_variable,
        hovermode="closest",
    )

    mo.vstack(
        [
            figure,
            mo.md(
                f"""

                **Valid observations:** {len(valid_series):,}


                **Missing observations:** {len(missing_series):,}


                **Points plotted:** {len(plot_series):,}

                """
            ),
        ]
    )
    return


@app.cell
def _(cleaned_flux, go, mo, pl, variable_selector):
    # ------------------------------------------------------------
    # Annual analysis
    # ------------------------------------------------------------

    selected_varaible_1 = variable_selector.value

    annual_data = cleaned_flux.filter(pl.col("TIMESTAMP_START").is_not_null()).with_columns(
        pl.col("TIMESTAMP_START").dt.year().alias("Year")
    )

    numeric_columns = annual_data.select(pl.selectors.numeric()).columns

    if selected_varaible_1 in numeric_columns:
        annual_summary = (
            annual_data.group_by("Year")
            .agg(
                [
                    pl.col(selected_varaible_1).mean().alias("Mean"),
                    pl.col(selected_varaible_1).median().alias("Median"),
                    pl.col(selected_varaible_1).min().alias("Minimum"),
                    pl.col(selected_varaible_1).max().alias("Maximum"),
                    pl.col(selected_varaible_1).count().alias("Valid observations"),
                ]
            )
            .sort("Year")
        )

        annual_figure = go.Figure(
            go.Scatter(
                x=annual_summary["Year"],
                y=annual_summary["Mean"],
                mode="lines+markers",
                marker=dict(size=7),
                name="Mean",
                hovertemplate=("<b>Year:</b> %{x}<br><b>Mean:</b> %{y:.3f}<extra></extra>"),
            )
        )

        annual_figure.update_layout(
            title=f"Annual mean — {selected_varaible_1}",
            xaxis_title="Year",
            yaxis_title=f"Mean {selected_varaible_1}",
        )

    else:
        annual_summary = (
            annual_data.group_by(
                [
                    "Year",
                    selected_varaible_1,
                ]
            )
            .agg(pl.len().alias("Count"))
            .sort(["Year", selected_varaible_1])
        )

        annual_figure = go.Figure()

        for category in annual_summary[selected_varaible_1].drop_nulls().unique().to_list():
            category_data = annual_summary.filter(pl.col(selected_varaible_1) == category)

            annual_figure.add_trace(
                go.Bar(
                    x=category_data["Year"],
                    y=category_data["Count"],
                    name=str(category),
                )
            )

        annual_figure.update_layout(
            title=f"Annual distribution — {selected_varaible_1}",
            xaxis_title="Year",
            yaxis_title="Number of observations",
            barmode="stack",
        )

    mo.vstack(
        [
            annual_figure,
            annual_summary,
        ]
    )
    return


@app.cell
def _(cleaned_flux, go, mo, pl, variable_selector):
    # ------------------------------------------------------------
    # Distribution of selected variable
    # ------------------------------------------------------------

    selected_variable_1 = variable_selector.value

    numeric_columns_1 = cleaned_flux.select(pl.selectors.numeric()).columns

    if selected_variable_1 in numeric_columns_1:
        values = cleaned_flux[selected_variable_1].drop_nulls()

        distribution_figure = go.Figure(
            go.Histogram(
                x=values,
                nbinsx=100,
                name=selected_variable_1,
                hovertemplate=("<b>Value:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>"),
            )
        )

        distribution_figure.update_layout(
            title=f"Distribution — {selected_variable_1}",
            xaxis_title=selected_variable_1,
            yaxis_title="Count",
        )

        output = distribution_figure

    else:
        value_counts = (
            cleaned_flux[selected_variable_1].value_counts().sort("count", descending=True)
        )

        distribution_figure = go.Figure(
            go.Bar(
                x=value_counts[selected_variable_1].cast(pl.String),
                y=value_counts["count"],
                hovertemplate=("<b>Value:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>"),
            )
        )

        distribution_figure.update_layout(
            title=f"Value distribution — {selected_variable_1}",
            xaxis_title=selected_variable_1,
            yaxis_title="Count",
        )

        output = mo.vstack(
            [
                distribution_figure,
                value_counts,
            ]
        )

    output
    return


@app.cell
def _(mo):
    mo.md("""
    ## Main ICOS Flux Variables

    The CH-Dav ICOS Level 2 dataset contains atmospheric,
    micrometeorological, turbulent flux, ecosystem exchange,
    and quality-control variables. The main variables used in
    this EDA are:

    | Variable | Meaning | Typical unit |
    |---|---|---|
    | **CO2** | Carbon dioxide mole fraction in wet air | µmol CO₂ mol⁻¹ |
    | **CO2_DRY** | Carbon dioxide dry-air mixing ratio | µmol CO₂ mol⁻¹ |
    | **CO2_SIGMA** | Standard deviation of CO₂ concentration | µmol CO₂ mol⁻¹ |
    | **FC** | Turbulent CO₂ flux | µmol CO₂ m⁻² s⁻¹ |
    | **SC** | CO₂ storage flux | µmol CO₂ m⁻² s⁻¹ |
    | **NEE** | Net ecosystem exchange of CO₂ | µmol CO₂ m⁻² s⁻¹ |
    | **H** | Sensible heat flux | W m⁻² |
    | **LE** | Latent heat turbulent flux | W m⁻² |
    | **H2O** | Water vapour mole fraction in wet air | mmol H₂O mol⁻¹ |
    | **H2O_DRY** | Water vapour dry mole fraction | mmol H₂O mol⁻¹ |
    | **USTAR** | Friction velocity | m s⁻¹ |
    | **WS** | Wind speed | m s⁻¹ |
    | **WD** | Wind direction | degrees |
    | **TAU** | Momentum flux / frictional stress | N m⁻² |
    | **T_SONIC** | Sonic air temperature | °C |
    | **MO_LENGTH** | Monin–Obukhov length, describing atmospheric stability | m |
    | **ZL** | Monin–Obukhov stability parameter | dimensionless |
    | **PBLH** | Planetary boundary-layer height | m |
    | **U_SIGMA** | Standard deviation of the longitudinal wind component | m s⁻¹ |
    | **V_SIGMA** | Standard deviation of the lateral wind component | m s⁻¹ |
    | **W_SIGMA** | Standard deviation of the vertical wind component | m s⁻¹ |


    ### Flux and storage terms

    The turbulent fluxes and storage fluxes represent different
    components of the exchange measured at the site:

    - **FC** — turbulent CO₂ flux
    - **SC** — CO₂ storage flux
    - **NEE** — net ecosystem exchange of CO₂
    - **H** — turbulent sensible heat flux
    - **SH** — sensible heat storage flux
    - **LE** — turbulent latent heat flux
    - **SLE** — latent heat storage flux

    ### Quality-control variables

    The dataset also contains many variables ending in:

    - **`_FLAG`** — quality-control or validity flag
    - **`_STAT`** — processing/statistical status information
    - **`_TEST`** — quality-control test result
    - **`_SIGMA`** — variability or standard-deviation measure
    - **`_UNC`** — uncertainty estimate

    These variables are retained in the dataset and are considered
    separately during the EDA rather than being discarded.

    **Source:** SwissFluxNet, Variable Abbreviations. URL: https://www.swissfluxnet.ethz.ch/index.php/data/variables/variable-abbreviations/
    """)
    return


if __name__ == "__main__":
    app.run()
