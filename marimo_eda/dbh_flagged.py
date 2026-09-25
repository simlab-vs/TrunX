"""Preprocess LWF tree inventory data for downstream analysis."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # DBH Flagged Tree Inventory

        This dataset contains **individual-tree inventory observations from the
        LWF monitoring network**.

        Each row corresponds to a tree observation and includes tree identifiers,
        location information, DBH and height measurements, crown characteristics,
        tree status, and quality-control/imputation flags.

        The main focus of this notebook is the **quality of DBH measurements**,
        particularly the `qa_flag_dbh` and `imputation_flag` fields, and how these
        vary between LWF sites and across inventory years.

        ### Main DBH-related variables

        - `dbh` — original/measured diameter at breast height
        - `dbh_corrected` — corrected DBH value
        - `qa_flag_dbh` — DBH quality-assurance flag
        - `imputation_flag` — DBH imputation indicator
        - `year` — inventory year
        - `sitename` — LWF monitoring site
        - `parcelname` — spatial parcel within the site
        """
    )
    return



@app.cell
def _():
    import marimo as mo
    import plotly.graph_objects as go
    import polars as pl

    return go, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # LWF Tree Inventory Preprocessing

    This notebook loads the raw LWF tree-inventory CSV and produces a
    cleaned Polars DataFrame in memory (`tree_cleaned`) for
    analysis.
    """)


@app.cell
def _():
    from pathlib import Path

    # Change this path if the raw file has a different name/location.
    input_path = Path("data/Givi_LWFdata_Jun26_flagged.csv")
    return (input_path,)


@app.cell
def _(input_path, mo, pl):
    if not input_path.exists():
        mo.stop(
            mo.md(
                f"""
                **Input file not found:** `{input_path}`

                Place the raw LWF tree-inventory CSV at that path, or change
                `input_path` in the cell above.
                """
            )
        )

    tree = pl.read_csv(
        input_path,
        null_values=["NA", "NaN", "nan", ""],
        try_parse_dates=False,
    )
    return (tree,)


@app.cell
def _(mo, tree):
    mo.md(f"""
    ## Raw data

    - Rows: **{tree.height:,}**
    - Columns: **{tree.width}**

    The raw CSV is loaded without changing the measurement values.
    """)


@app.cell
def _(tree):
    expected_columns = [
        "sitename",
        "parcelname",
        "year",
        "date_observation",
        "invnr",
        "banr",
        "banreti",
        "tree_species",
        "dbh",
        "height",
        "dbh_corrected",
        "height_corrected",
        "crown_base_height",
        "crown_width",
        "tree_status",
        "lat",
        "lon",
        "x_LV03",
        "y_LV03",
        "qa_flag_dbh",
        "imputation_flag",
        "qa_flag_height",
        "imputation_flag_height",
    ]

    missing_columns = [
        column for column in expected_columns if column not in tree.columns
    ]

    if missing_columns:
        raise ValueError(
            "The raw file is missing expected columns: " + ", ".join(missing_columns)
        )
    return (expected_columns,)


@app.cell
def _(expected_columns, pl, tree):
    tree_cleaned = tree.clone()

    tree_cleaned = tree_cleaned.select(expected_columns)

    tree_cleaned = tree_cleaned.with_columns(
        pl.col("date_observation")
        .cast(pl.String)
        .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
        .alias("date_observation")
    )

    numeric_columns = [
        "year",
        "invnr",
        "banr",
        "banreti",
        "dbh",
        "height",
        "dbh_corrected",
        "height_corrected",
        "crown_base_height",
        "crown_width",
        "lat",
        "lon",
        "x_LV03",
        "y_LV03",
    ]

    tree_cleaned = tree_cleaned.with_columns(
        [
            pl.col(column).cast(pl.Float64, strict=False).alias(column)
            for column in numeric_columns
        ]
    )

    integer_columns = ["year", "invnr", "banr", "banreti"]

    tree_cleaned = tree_cleaned.with_columns(
        [
            pl.col(column).cast(pl.Int64, strict=False).alias(column)
            for column in integer_columns
        ]
    )
    return (tree_cleaned,)


@app.cell
def _(mo, tree_cleaned):
    mo.md(f"""
    ## Processed data

    - Rows: **{tree_cleaned.height:,}**
    - Columns: **{tree_cleaned.width}**
    """)


@app.cell
def _(mo, tree_cleaned):
    mo.vstack(
        [
            mo.md("### Preview"),
            mo.ui.table(tree_cleaned.head(2000)),
        ]
    )


@app.cell
def _(tree_cleaned):
    tree_cleaned["year"].unique()


@app.cell
def _(mo, pl, tree_cleaned):
    null_summary = (
        tree_cleaned.null_count()
        .transpose(
            include_header=True,
            header_name="variable",
            column_names=["missing"],
        )
        .with_columns(
            (pl.col("missing") / tree_cleaned.height * 100)
            .round(2)
            .alias("missing_percent")
        )
        .filter(pl.col("missing") > 0)
        .sort("missing", descending=True)
    )

    mo.vstack(
        [
            mo.md("### Missing values"),
            mo.ui.table(null_summary),
        ]
    )


@app.cell
def _(mo, tree_cleaned):
    total_trees = tree_cleaned["banr"].n_unique()
    total_sites = tree_cleaned["sitename"].drop_nulls().n_unique()
    total_species = tree_cleaned["tree_species"].drop_nulls().n_unique()

    year_min = tree_cleaned["year"].drop_nulls().min()
    year_max = tree_cleaned["year"].drop_nulls().max()

    mo.vstack(
        [
            mo.md(
                """
                # LWF Tree Inventory — Exploratory Data Analysis

                This notebook explores individual-tree inventory measurements
                across LWF sites, species, and inventory years.
                """
            ),
            mo.hstack(
                [
                    mo.stat(label="Rows", value=f"{tree_cleaned.height:,}"),
                    mo.stat(label="Trees", value=f"{total_trees:,}"),
                    mo.stat(label="LWF sites", value=f"{total_sites:,}"),
                    mo.stat(label="Species", value=f"{total_species:,}"),
                    mo.stat(
                        label="Year range",
                        value=f"{int(year_min)}–{int(year_max)}",
                    ),
                ]
            ),
        ]
    )


@app.cell
def _(mo, pl, tree_cleaned):
    site_summary = (
        tree_cleaned
        .group_by("sitename")
        .agg(
            [
                pl.len().alias("observations"),
                pl.col("banr").n_unique().alias("trees"),
                pl.col("tree_species").n_unique().alias("species"),
                pl.col("year").min().alias("first_year"),
                pl.col("year").max().alias("last_year"),
            ]
        )
        .sort("sitename")
    )

    mo.vstack(
        [
            mo.md("## LWF site summary"),
            mo.ui.table(site_summary),
        ]
    )


@app.cell
def _(mo, pl, tree_cleaned):
    species_summary = (
        tree_cleaned
        .group_by("tree_species")
        .agg(
            [
                pl.len().alias("observations"),
                pl.col("banr").n_unique().alias("trees"),
                pl.col("sitename").n_unique().alias("sites"),
                pl.col("dbh").mean().round(2).alias("mean_dbh_cm"),
                pl.col("height").mean().round(2).alias("mean_height_m"),
            ]
        )
        .sort("observations", descending=True)
    )

    mo.vstack(
        [
            mo.md("## Tree species"),
            mo.ui.table(species_summary),
        ]
    )


@app.cell
def _():
    metric_labels = {
        "dbh": "DBH",
        "height": "Height",
        "dbh_corrected": "DBH (corrected)",
        "height_corrected": "Height (corrected)",
        "crown_base_height": "Crown base height",
        "crown_width": "Crown width",
    }

    metric_units = {
        "dbh": "cm",
        "height": "m",
        "dbh_corrected": "cm",
        "height_corrected": "m",
        "crown_base_height": "m",
        "crown_width": "m",
    }
    return metric_labels, metric_units


@app.cell
def _(metric_labels, metric_units, mo, tree_cleaned):
    site_options = sorted(
        tree_cleaned["sitename"].drop_nulls().cast(str).unique().to_list()
    )

    metric_options = {
        (
            f"{metric_labels.get(column, column)} ({metric_units.get(column, '')})"
            if metric_units.get(column, "")
            else metric_labels.get(column, column)
        ): column
        for column in metric_labels
    }

    site_selector = mo.ui.multiselect(
        options=site_options,
        value=site_options[: min(2, len(site_options))],
        label="LWF sites",
    )

    metric_selector = mo.ui.dropdown(
        options=metric_options,
        value=next(iter(metric_options)),
        label="Metric",
    )

    mo.vstack(
        [
            mo.md("## Compare multiple LWF sites"),
            mo.hstack(
                [
                    site_selector,
                    metric_selector,
                ]
            ),
        ]
    )
    return metric_selector, site_selector


@app.cell
def _(
    go,
    metric_labels,
    metric_selector,
    metric_units,
    mo,
    pl,
    site_selector,
    tree_cleaned,
):
    metric = metric_selector.value

    metric_label = metric_labels.get(metric, metric)
    metric_unit = metric_units.get(metric, "")

    if len(site_selector.value) == 0:
        fig_compare = mo.md(
            "*Select at least one LWF site above to see the comparison.*"
        )

    else:
        fig_compare = go.Figure()

        site_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        for site_index, site in enumerate(site_selector.value):
            site_color = site_palette[site_index % len(site_palette)]

            site_yearly = (
                tree_cleaned.filter(pl.col("sitename") == site)
                .filter(pl.col("year").is_not_null() & pl.col(metric).is_not_null())
                .group_by("year")
                .agg(
                    [
                        pl.col(metric).mean().round(2).alias(metric),
                        pl.col("banr").n_unique().alias("n_trees"),
                    ]
                )
                .sort("year")
            )

            fig_compare.add_trace(
                go.Scatter(
                    x=site_yearly["year"],
                    y=site_yearly[metric],
                    mode="lines+markers",
                    name=f"{site} — {metric_label}",
                    legendgroup=site,
                    line={"color": site_color},
                    hovertemplate=(
                        "<b>Site:</b> "
                        + site
                        + "<br><b>Year:</b> %{x}"
                        + f"<br><b>Mean {metric_label}:</b> %{{y:.2f}}"
                        + (f" {metric_unit}" if metric_unit else "")
                        + "<extra></extra>"
                    ),
                )
            )

            fig_compare.add_trace(
                go.Scatter(
                    x=site_yearly["year"],
                    y=site_yearly["n_trees"],
                    mode="lines+markers",
                    name=f"{site} — Trees",
                    legendgroup=site,
                    yaxis="y2",
                    line={"color": site_color, "dash": "dot"},
                    marker={"symbol": "diamond"},
                    hovertemplate=(
                        "<b>Site:</b> "
                        + site
                        + "<br><b>Year:</b> %{x}"
                        + "<br><b>Trees:</b> %{y}"
                        + "<extra></extra>"
                    ),
                )
            )

        fig_compare.update_layout(
            title=f"Mean {metric_label} by year: " + " vs ".join(site_selector.value),
            xaxis_title="Inventory year",
            yaxis_title=(
                f"{metric_label} ({metric_unit})" if metric_unit else metric_label
            ),
            yaxis2={
                "title": "Number of trees",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "rangemode": "tozero",
            },
            hovermode="x unified",
            legend_title="LWF site",
        )

    fig_compare  # noqa: B018


@app.cell
def _(metric_labels, metric_units, mo, tree_cleaned):
    subsite_options = sorted(
        tree_cleaned["sitename"].drop_nulls().cast(str).unique().to_list()
    )

    subsite_metric_options = {
        (
            f"{metric_labels.get(column, column)} ({metric_units.get(column, '')})"
            if metric_units.get(column, "")
            else metric_labels.get(column, column)
        ): column
        for column in metric_labels
    }

    subsite_selector = mo.ui.dropdown(
        options=subsite_options,
        value=subsite_options[0] if subsite_options else None,
        label="LWF site",
    )

    subsite_metric_selector = mo.ui.dropdown(
        options=subsite_metric_options,
        value=next(iter(subsite_metric_options)),
        label="Metric",
    )

    mo.vstack(
        [
            mo.md("## Compare Teilfläche vs Subfläche within a site"),
            mo.md(
                "Compares parcel-level (`parcelname`) mean values within a "
                "single LWF site."
            ),
            mo.hstack(
                [
                    subsite_selector,
                    subsite_metric_selector,
                ]
            ),
        ]
    )
    return subsite_metric_selector, subsite_selector


@app.cell
def _(
    go,
    metric_labels,
    metric_units,
    mo,
    pl,
    subsite_metric_selector,
    subsite_selector,
    tree_cleaned,
):
    parcel_metric = subsite_metric_selector.value
    selected_site = subsite_selector.value

    parcel_metric_label = metric_labels.get(parcel_metric, parcel_metric)
    parcel_metric_unit = metric_units.get(parcel_metric, "")

    if selected_site is None:
        fig_parcel_compare = mo.md(
            "*Select a LWF site above to see the comparison.*"
        )
    else:
        site_data = tree_cleaned.filter(pl.col("sitename") == selected_site)

        parcel_options = sorted(
            site_data["parcelname"].drop_nulls().cast(str).unique().to_list()
        )

        if len(parcel_options) == 0:
            fig_parcel_compare = mo.md(
                f"*No parcel data found for site **{selected_site}**.*"
            )
        else:
            fig_parcel_compare = go.Figure()

            parcel_palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            ]

            parcel_labels = {}

            for parcel_index, parcel in enumerate(parcel_options):
                parcel_color = parcel_palette[parcel_index % len(parcel_palette)]
                parcel_data = site_data.filter(pl.col("parcelname") == parcel)

                parcel_total_trees = parcel_data["banr"].n_unique()
                parcel_label = f"{parcel} ({parcel_total_trees:,} trees total)"
                parcel_labels[parcel] = parcel_label

                parcel_yearly = (
                    parcel_data
                    .filter(
                        pl.col("year").is_not_null()
                        & pl.col(parcel_metric).is_not_null()
                    )
                    .group_by("year")
                    .agg(
                        [
                            pl.col(parcel_metric)
                            .mean()
                            .round(2)
                            .alias(parcel_metric),
                            pl.col("banr").n_unique().alias("n_trees"),
                        ]
                    )
                    .sort("year")
                )

                fig_parcel_compare.add_trace(
                    go.Scatter(
                        x=parcel_yearly["year"],
                        y=parcel_yearly[parcel_metric],
                        mode="lines+markers",
                        name=f"{parcel_label} — {parcel_metric_label}",
                        legendgroup=parcel,
                        line={"color": parcel_color},
                        hovertemplate=(
                            "<b>Parcel:</b> "
                            + parcel
                            + "<br><b>Year:</b> %{x}"
                            + f"<br><b>Mean {parcel_metric_label}:</b> %{{y:.2f}}"
                            + (
                                f" {parcel_metric_unit}"
                                if parcel_metric_unit
                                else ""
                            )
                            + "<extra></extra>"
                        ),
                    )
                )

                fig_parcel_compare.add_trace(
                    go.Scatter(
                        x=parcel_yearly["year"],
                        y=parcel_yearly["n_trees"],
                        mode="lines+markers",
                        name=f"{parcel} — Trees",
                        legendgroup=parcel,
                        yaxis="y2",
                        line={"color": parcel_color, "dash": "dot"},
                        marker={"symbol": "diamond"},
                        hovertemplate=(
                            "<b>Parcel:</b> "
                            + parcel
                            + "<br><b>Year:</b> %{x}"
                            + "<br><b>Trees:</b> %{y}"
                            + "<extra></extra>"
                        ),
                    )
                )

            fig_parcel_compare.update_layout(
                title=(
                    f"Mean {parcel_metric_label} by year — {selected_site}: "
                    + " vs ".join(parcel_labels[p] for p in parcel_options)
                ),
                xaxis_title="Inventory year",
                yaxis_title=(
                    f"{parcel_metric_label} ({parcel_metric_unit})"
                    if parcel_metric_unit
                    else parcel_metric_label
                ),
                yaxis2={
                    "title": "Number of trees",
                    "overlaying": "y",
                    "side": "right",
                    "showgrid": False,
                    "rangemode": "tozero",
                },
                hovermode="x unified",
                legend_title="Parcel (Teilfläche/Subfläche)",
            )

    fig_parcel_compare  # noqa: B018


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For more info on the Plot Design, visit https://lwf.wsl.ch/en/about-wsl/instrumented-field-sites-and-laboratories/lwf-demoflaeche/plot-design/
    """)


@app.cell
def _():
    imputation_flag_columns = [
        "qa_flag_dbh",
        "imputation_flag",
        "qa_flag_height",
        "imputation_flag_height",
    ]
    return (imputation_flag_columns,)


@app.cell
def _(imputation_flag_columns, mo, tree_cleaned):
    imputation_site_options = sorted(
        tree_cleaned["sitename"].drop_nulls().cast(str).unique().to_list()
    )

    imputation_flag_options = {
        f"{flag_column} = {flag_value}": (flag_column, flag_value)
        for flag_column in imputation_flag_columns
        for flag_value in sorted(
            tree_cleaned[flag_column].drop_nulls().cast(str).unique().to_list()
        )
    }

    imputation_site_selector = mo.ui.dropdown(
        options=imputation_site_options,
        value=imputation_site_options[0] if imputation_site_options else None,
        label="LWF site",
    )

    imputation_flag_selector = mo.ui.dropdown(
        options=imputation_flag_options,
        value=next(iter(imputation_flag_options), None),
        label="QA / imputation flag",
    )

    mo.vstack(
        [
            mo.md("## Imputation percentage by year"),
            mo.md(
                "Percentage of observations carrying the selected QA/imputation "
                "flag, by year, for a single LWF site."
            ),
            mo.hstack(
                [
                    imputation_site_selector,
                    imputation_flag_selector,
                ]
            ),
        ]
    )
    return imputation_flag_selector, imputation_site_selector


@app.cell
def _(
    go,
    imputation_flag_selector,
    imputation_site_selector,
    mo,
    pl,
    tree_cleaned,
):
    imputation_site = imputation_site_selector.value
    imputation_flag_selection = imputation_flag_selector.value

    if imputation_site is None or imputation_flag_selection is None:
        fig_imputation = mo.md(
            "*Select a LWF site and a flag above to see the imputation "
            "percentage by year.*"
        )
    else:
        imputation_flag_column, imputation_flag_value = imputation_flag_selection

        imputation_yearly = (
            tree_cleaned.filter(pl.col("sitename") == imputation_site)
            .filter(pl.col("year").is_not_null())
            .group_by("year")
            .agg(
                [
                    pl.len().alias("total"),
                    (
                        pl.col(imputation_flag_column).cast(pl.String)
                        == imputation_flag_value
                    )
                    .sum()
                    .alias("flagged"),
                ]
            )
            .with_columns(
                (pl.col("flagged") / pl.col("total") * 100)
                .round(2)
                .alias("percentage")
            )
            .sort("year")
        )

        if imputation_yearly.height == 0:
            fig_imputation = mo.md(
                f"*No data found for site **{imputation_site}**.*"
            )
        else:
            fig_imputation = go.Figure()

            fig_imputation.add_trace(
                go.Scatter(
                    x=imputation_yearly["year"],
                    y=imputation_yearly["percentage"],
                    mode="lines+markers",
                    name=f"{imputation_flag_column} = {imputation_flag_value}",
                    hovertemplate=(
                        "<b>Year:</b> %{x}"
                        "<br><b>Percentage:</b> %{y:.2f}%"
                        "<extra></extra>"
                    ),
                )
            )

            fig_imputation.update_layout(
                title=(
                    f"Imputation percentage by year — {imputation_site}: "
                    f"{imputation_flag_column} = {imputation_flag_value}"
                ),
                xaxis_title="Inventory year",
                yaxis_title="Percentage (%)",
                hovermode="x unified",
            )

    fig_imputation  # noqa: B018

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Key Observations
    - DBH & height vary significantly by plot site.
    - Height & DBh flag for likely errors has a varying trend over the years.
    """)
    return

if __name__ == "__main__":
    app.run()