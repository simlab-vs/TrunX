"""EFM data processing and EDA notebook."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os

    import geopandas as gpd
    import polars as pl

    from trunx.config import data_folder

    return data_folder, gpd, os, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot data
    """)
    return


@app.cell
def _(data_folder, gpd, os, pl):
    # Load plot data
    plot_file = os.path.join(data_folder, "SwissData/EFM/efm_geo_data.gpkg")
    print("Geo layers are:\n", gpd.list_layers(plot_file))

    plot_file_df = gpd.read_file(plot_file, layer="plot_point")

    plot_file_df = plot_file_df[plot_file_df.geometry.notna()]

    plot_file_df["lon"] = plot_file_df.geometry.x
    plot_file_df["lat"] = plot_file_df.geometry.y
    plot_file_df = plot_file_df.drop(columns=["geometry"])

    plot_file_df = pl.from_pandas(plot_file_df)
    plot_file_df = plot_file_df.select("plot", "lon", "lat", "elevation", "area")

    plot_file_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tree data
    """)
    return


@app.cell
def _(data_folder, os, pl):
    # Load tree files
    tree_file = os.path.join(data_folder, "SwissData/EFM/efm_tree_data.csv")
    tree_plot_df = pl.read_csv(tree_file)
    print("Distinct species are:", tree_plot_df["species"].unique().to_list())

    ## Plot locations with single species
    single_species_df = (
        tree_plot_df.filter(pl.col("status") == 1)
        .group_by(["plot", "year"])
        .agg(n_species=pl.col("species").n_unique())
        .filter(pl.col("n_species") == 1)
        .join(tree_plot_df, on=["plot", "year"], how="inner")
    )

    print(single_species_df.head())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
