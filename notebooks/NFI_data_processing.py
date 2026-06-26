"""NFI data preprocessing."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NFI data preprocessing
    """)
    return


@app.cell
def _():
    import os

    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from pyproj import Transformer

    from trunx.config import data_folder

    # NFI_raw_data_loc = os.path.join(data_folder, "SwissData/NFI_data_Givi_202606")
    NFI_raw_data_loc = os.path.join(data_folder, "SwissData/NFI")
    return NFI_raw_data_loc, Transformer, go, os, pl, px


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

    print("Number of plots: ", raw_data_plot_level["plot_id"].n_unique())
    return (raw_data_plot_level,)


@app.cell
def _(raw_data_plot_level):
    raw_data_plot_level["plot_id", "INVNR"].n_unique()
    return


@app.cell
def _(raw_data_plot_level):
    print(raw_data_plot_level.select("plot_id").n_unique())
    return


@app.cell
def _(go, pl, raw_data_plot_level):
    plot_locations_fig = go.Figure()

    plot_locations_fig.add_trace(
        go.Scattermap(
            lat=raw_data_plot_level["lat"],
            lon=raw_data_plot_level["lon"],
            mode="markers",
            marker=dict(size=3),
            name="Tree locations",
        )
    )

    plot_locations_fig.update_layout(
        map=dict(
            style="open-street-map",
            zoom=7,
            center=dict(
                lat=raw_data_plot_level.select(pl.mean("lat")).item(),
                lon=raw_data_plot_level.select(pl.mean("lon")).item(),
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
        # 15: "Pinus sylvestris",
        50: "Fagus sylvatica",
        # 51: "Quercus robur",
        # 52: "Quercus petraea",
    }

    # Keep only the species of interest
    raw_data_tree_level = raw_data_tree_level.filter(
        pl.col("specie").is_in(species_mapping.keys())
    )

    raw_data_tree_level = raw_data_tree_level.with_columns(
        pl.col("specie").replace_strict(species_mapping).alias("specie_name")
    )

    print("Number of trees:", raw_data_tree_level.select("tree_id").n_unique())
    return (raw_data_tree_level,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tree growth data with observation dates
    """)
    return


@app.cell
def _(raw_data_plot_level, raw_data_tree_level):
    tree_growth_data = raw_data_tree_level.join(raw_data_plot_level, on=["plot_id", "INVNR"])

    tree_growth_data.head()
    return (tree_growth_data,)


@app.cell
def _(go, pl, tree_growth_data):
    tree_locations = go.Figure()

    tree_locations.add_trace(
        go.Scattermap(
            lat=tree_growth_data["lat"],
            lon=tree_growth_data["lon"],
            mode="markers",
            marker=dict(size=3),
            name="Tree locations",
        )
    )

    tree_locations.update_layout(
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Filter single species locations
    """)
    return


@app.cell
def _(pl, px, tree_growth_data):
    species_per_inv = (
        tree_growth_data.group_by(["plot_id", "INVNR"])
        .agg(
            n_species=pl.col("specie").n_unique(),
            specie=pl.col("specie").first(),
            specie_name=pl.col("specie_name").unique(),
        )
        .filter(pl.col("n_species") == 1)
    )

    single_species_growth_data = (
        species_per_inv.group_by("plot_id")
        .agg(
            num_species=pl.col("specie").n_unique(),
            species=pl.col("specie").unique(),
            species_name=pl.col("specie_name").unique(),
        )
        .filter(pl.col("num_species") == 1)
        .join(tree_growth_data, on="plot_id", how="inner")
    )

    print(
        "Number of plots with single specie tree data:",
        single_species_growth_data.select("lat", "lon").n_unique(),
    )

    # Get unique species and assign colors
    unique_species = single_species_growth_data["specie_name"].unique().to_list()
    colors = px.colors.qualitative.Set1
    species_to_color = {
        species: colors[i % len(colors)] for i, species in enumerate(unique_species)
    }

    # Create a color column
    single_species_growth_data = single_species_growth_data.with_columns(
        pl.col("specie_name")
        .replace_strict(species_to_color, default="rgb(128,128,128)")
        .alias("color")
    ).with_columns(year=pl.col("DATUMF").str.strptime(pl.Date, "%d/%m/%Y").dt.year())
    return single_species_growth_data, species_to_color, unique_species


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Single species Tree locations
    """)
    return


@app.cell
def _(
    go,
    pl,
    single_species_growth_data,
    species_to_color,
    tree_growth_data,
    unique_species,
):
    fig = go.Figure()

    # Add a separate trace for each species
    for species in unique_species:
        # Filter data for this species
        species_data = single_species_growth_data.filter(pl.col("specie_name") == species)

        fig.add_trace(
            go.Scattermap(
                lat=species_data["lat"],
                lon=species_data["lon"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=species_to_color[species],
                ),
                name=species,
                text=[species] * len(species_data),
                hovertemplate="Species: %{text}<br>Lat: %{lat:.4f} \
                    <br>Lon: %{lon:.4f}<extra></extra>",
                legendgroup=species,  # Groups legend items together
                showlegend=True,
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
        legend=dict(
            x=0,
            y=1,
            title="Species",
            bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
            bordercolor="black",
            borderwidth=1,
        ),
        title=dict(
            text="Tree locations colored by species",
            x=0.5,
            xanchor="center",
        ),
    )

    fig.show()
    return


@app.cell
def _(pl, single_species_growth_data):
    growth_data = (
        single_species_growth_data.group_by("plot_id", "year", "specie_name")
        .agg(
            n_stems=pl.col("tree_rep_fact").sum(),
            DBH=(pl.col("DBH") * pl.col("tree_rep_fact")).sum() / pl.col("tree_rep_fact").sum(),
            biom_stem=(pl.col("stem_biomass") * pl.col("tree_rep_fact")).sum(),
            biom_foliage=(pl.col("foliage_biomass") * pl.col("tree_rep_fact")).sum(),
            biom_root=(pl.col("root_biomass") * pl.col("tree_rep_fact")).sum(),
        )
        .with_columns(
            biom_stem=(pl.col("biom_stem") / 1000).round(2),
            biom_foliage=(pl.col("biom_foliage") / 1000).round(2),
            biom_root=(pl.col("biom_root") / 1000).round(2),
        )
    )

    growth_data = growth_data.join(
        single_species_growth_data, on=["plot_id", "specie_name"]
    ).select(
        "plot_id",
        "lat",
        "lon",
        "year",
        "specie_name",
        "n_stems",
        "DBH",
        "biom_stem",
        "biom_root",
        "biom_foliage",
        "DATUMF",
    )

    print(growth_data.head())
    return (growth_data,)


@app.cell
def _(growth_data, pl):
    growth_data.group_by("plot_id", "specie_name").agg(
        num_periods=pl.col("year").n_unique()
    ).filter(pl.col("num_periods") == 5).group_by("specie_name").agg(
        count=pl.col("plot_id").n_unique()
    )
    return


if __name__ == "__main__":
    app.run()
