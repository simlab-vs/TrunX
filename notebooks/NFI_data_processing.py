"""NFI data preprocessing."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NFI data preprocessing
    """)
    return


@app.cell
def _():
    import os

    import polars as pl
    from pyproj import Transformer

    from trunx.config import data_folder

    NFI_raw_data_loc = os.path.join(data_folder, "SwissData/NFI_data_Givi_202606")
    return NFI_raw_data_loc, Transformer, os, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot level data
    """)
    return


@app.cell
def _(NFI_raw_data_loc, Transformer, os, pl):
    raw_data_plot_level = pl.read_csv(os.path.join(NFI_raw_data_loc, "Givi_plot_level.csv"))

    # Create transformer once
    transformer = Transformer.from_crs("EPSG:21781", "EPSG:4326", always_xy=True)

    x_coords = raw_data_plot_level["X"].to_numpy()
    y_coords = raw_data_plot_level["Y"].to_numpy()

    # Vectorized conversion (single call!)
    lon, lat = transformer.transform(x_coords, y_coords)

    # Add to DataFrame
    raw_data_plot_level = raw_data_plot_level.with_columns(
        [pl.Series("lon", lon), pl.Series("lat", lat)]
    )

    raw_data_plot_level = raw_data_plot_level.rename({"CLNR": "plot_id"})

    print(
        "Number of plots: ",
    )
    print()
    return (raw_data_plot_level,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tree level data
    """)
    return


@app.cell
def _(NFI_raw_data_loc, os, pl):
    raw_data_tree_level = pl.read_csv(os.path.join(NFI_raw_data_loc, "Givi_tree_level.csv"))

    print(raw_data_tree_level.schema)

    raw_data_tree_level = raw_data_tree_level.rename(
        {
            "CLNR": "plot_id",
            "BANR": "tree_id",
            "BARTLFI": "specie",
            "RPSTZ": "tree_rep_fact",
            "D13": "DBH",
            "BIOMASSE": "tree_biomass",
            "VMRDBIOM": "stem_biomass",
            "ASTDHBIOM": "branch_biomass",
            "REISIGBIOM": "brushwood_biomass",
            "WURZELN": "root_biomass",
            "NADELN": "foliage_biomass",
        }
    )

    # Species code (from metadata)
    species_mapping = {
        10: "Picea abies",
        15: "Pinus sylvestris",
        50: "Fagus sylvatica",
        51: "Quercus robur",
        52: "Quercus petraea",
    }

    # Keep only the species of interest
    raw_data_tree_level = raw_data_tree_level.filter(
        pl.col("specie").is_in(species_mapping.keys())
    )

    print("Number of tree:", raw_data_tree_level.select("tree_id").n_unique())
    return (raw_data_tree_level,)


@app.cell
def _(raw_data_plot_level, raw_data_tree_level):
    tree_growth_data = raw_data_tree_level.join(raw_data_plot_level, on=["plot_id", "INVNR"])

    tree_growth_data.head()
    return (tree_growth_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tree locations
    """)
    return


@app.cell
def _(pl, tree_growth_data):
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(
        go.Scattermap(
            lat=tree_growth_data["lat"],
            lon=tree_growth_data["lon"],
            mode="markers",
            marker=dict(size=3),
            name="Tree locations",
        )
    )

    fig.update_layout(
        map=dict(
            style="open-street-map",
            zoom=7,
            center=dict(
                lat=tree_growth_data.select(pl.mean("lat")).item(),
                lon=tree_growth_data.select(pl.mean("lon")).item(),
            ),
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        legend=dict(x=0, y=1),
        title=dict(
            text="Tree locations",
            x=0.5,
            xanchor="center",
        ),
    )
    return


@app.cell
def _(tree_growth_data):
    tree_growth_data.group_by(["plot_id", "tree_id"]).len()
    return


if __name__ == "__main__":
    app.run()
