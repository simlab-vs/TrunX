"""Geographic locations of ICP plots."""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():

    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import polars as pl

    sys.path.append(str(Path(__file__).parent.parent))
    from scripts.support_utils import load_prepare_data
    from trunx.plot_utils import plot_geographic_location_species

    return load_prepare_data, pl, plot_geographic_location_species


@app.cell
def _(load_prepare_data, pl, plot_geographic_location_species):
    _, df = load_prepare_data()

    # Single species locations
    df = df.join(
        df.group_by(["Lat", "Lon"]).agg(species_count=pl.col("Species").unique().count()),
        on=["Lat", "Lon"],
    ).filter(pl.col("species_count") == 1)

    plot_geographic_location_species(df, selected_species=["Spruce", "Beech", "Pine", "Oak"])
    return (df,)


@app.cell
def _(df, pl):
    swiss_df = df.filter(pl.col("code_country") == 50)

    print("Different species present in Switzerland are :", swiss_df["specie"].unique().to_list())

    swiss_df.group_by(["code_plot", "Lat", "Lon"], maintain_order=True).agg(
        num_trees=pl.col("tree_id").n_unique(), specie=pl.col("specie").unique()
    )
    return


if __name__ == "__main__":
    app.run()
