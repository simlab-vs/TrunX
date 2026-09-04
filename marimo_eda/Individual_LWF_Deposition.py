import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import marimo as mo
    from pathlib import Path

    return Path, mo, pd


@app.cell
def _(Path, pd):
    foliage_path = Path("lwf_foliage_dw100_i_2026-07-30.csv")

    foliage = pd.read_csv(
        foliage_path,
        sep=";",
    )
    return (foliage,)


@app.cell
def _(foliage):
    foliage.head()
    return


@app.cell
def _(foliage, pd):
    foliage["survey_date"] = pd.to_datetime(
        foliage["survey_date"],
        errors="coerce",
    )
    return


@app.cell
def _(foliage, mo):
    plot_selector = mo.ui.dropdown(
        options=["All"] + sorted(
            foliage["plot_id"]
            .dropna()
            .unique()
            .tolist()
        ),
        value="All",
        label="LWF site",
    )

    species_selector = mo.ui.dropdown(
        options=["All"] + sorted(
            foliage["species"]
            .dropna()
            .unique()
            .tolist()
        ),
        value="All",
        label="Species",
    )

    leaf_type_selector = mo.ui.dropdown(
        options=["All"] + sorted(
            foliage["leaf_type"]
            .dropna()
            .unique()
            .tolist()
        ),
        value="All",
        label="Leaf type",
    )

    age_selector = mo.ui.dropdown(
        options=["All"] + sorted(
            foliage["leaf_age_class"]
            .dropna()
            .unique()
            .tolist()
        ),
        value="All",
        label="Leaf age class",
    )

    mo.vstack([
        mo.hstack([
            plot_selector,
            species_selector,
        ]),
        mo.hstack([
            leaf_type_selector,
            age_selector,
        ]),
    ])
    return age_selector, leaf_type_selector, plot_selector, species_selector


@app.cell
def _(
    age_selector,
    foliage,
    leaf_type_selector,
    plot_selector,
    species_selector,
):
    filtered_foliage = foliage.copy()

    if plot_selector.value != "All":
        filtered_foliage = filtered_foliage[
            filtered_foliage["plot_id"] == plot_selector.value
        ]

    if species_selector.value != "All":
        filtered_foliage = filtered_foliage[
            filtered_foliage["species"] == species_selector.value
        ]

    if leaf_type_selector.value != "All":
        filtered_foliage = filtered_foliage[
            filtered_foliage["leaf_type"] == leaf_type_selector.value
        ]

    if age_selector.value != "All":
        filtered_foliage = filtered_foliage[
            filtered_foliage["leaf_age_class"] == age_selector.value
        ]

    filtered_foliage
    return (filtered_foliage,)


@app.cell
def _(filtered_foliage):
    filtered_foliage["gew100"].describe()
    return


@app.cell
def _(filtered_foliage):
    filtered_foliage["gew100"].isna().sum()
    return


@app.cell
def _(filtered_foliage):
    import plotly.graph_objects as go

    plot_data = (
        filtered_foliage[
            ["survey_date", "gew100", "sample_id"]
        ]
        .dropna(subset=["survey_date", "gew100"])
        .sort_values("survey_date")
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_data["survey_date"],
            y=plot_data["gew100"],
            mode="markers",
            name="Individual measurement",
            marker=dict(size=7),
            customdata=plot_data["sample_id"],
            hovertemplate=(
                "<b>Date:</b> %{x|%d %B %Y}"
                "<br><b>gew100:</b> %{y:.2f} g"
                "<br><b>Sample:</b> %{customdata}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Foliage dry weight over time",
        xaxis_title="Survey date",
        yaxis_title="gew100 (g)",
        hovermode="closest",
    )

    fig
    return (go,)


@app.cell
def _(foliage):
    samples_per_group = (
        foliage
        .groupby(
            [
                "survey_date",
                "species",
                "leaf_type",
                "leaf_age_class",
            ],
            dropna=False,
        )
        .agg(
            n_samples=("sample_id", "nunique"),
            mean_gew100=("gew100", "mean"),
            median_gew100=("gew100", "median"),
            std_gew100=("gew100", "std"),
        )
        .reset_index()
        .sort_values("survey_date")
    )

    samples_per_group
    return


@app.cell
def _(go, samples_per_date):
    fig_1 = go.Figure()

    fig_1.add_trace(
        go.Bar(
            x=samples_per_date["survey_date"],
            y=samples_per_date["n_samples"],
            hovertemplate=(
                "<b>Date:</b> %{x|%d %B %Y}"
                "<br><b>Samples:</b> %{y}"
                "<extra></extra>"
            ),
        )
    )

    fig_1.update_layout(
        title="Number of foliage samples per survey date",
        xaxis_title="Survey date",
        yaxis_title="Number of individual samples",
    )

    fig_1
    return


if __name__ == "__main__":
    app.run()
