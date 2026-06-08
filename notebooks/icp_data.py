"""ICP data processing for 3PG model implementation."""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    from io import StringIO

    import pandas as pd
    import polars as pl
    import requests

    return StringIO, pd, pl, requests


@app.cell
def _():
    # SPECIES_MAPPING = {
    #     "Picea abies": "spruce",  # Norway Spruce
    #     "Pinus sylvestris": "pine",  # Scots Pine
    #     "Fagus sylvatica": "beech",  # Common Beech
    #     "Quercus petraea": "oak",  # Sessile Oak
    #     "Quercus robur": "oak",  # Pedunculate Oak
    # }
    return


@app.cell
def _(StringIO, pd, pl, requests):
    url = "https://icp-forests.org/documentation/Dictionaries/d_tree_spec.html"
    html = requests.get(url).text
    species_df = pd.read_html(StringIO(html))[0]

    species_df = pl.from_pandas(species_df)
    species_df = species_df.rename({"CODE": "code_tree_species", "DESCRIPTION": "specie"})

    url = "https://icp-forests.org/documentation/Dictionaries/d_country.html"
    html = requests.get(url).text
    country_df = pd.read_html(StringIO(html))[0]

    country_df = pl.from_pandas(country_df)
    country_df = country_df.rename({"CODE": "code_country", "LIB_COUNTRY": "country"})
    return country_df, species_df


@app.cell
def _(pl):
    df_plots_raw = pl.read_csv("./data/raw/ICP/595_si_20260309085250/si_plt.csv", separator=";")

    df_plots_raw = df_plots_raw.with_columns(
        (
            pl.col("code_country").cast(pl.Utf8).str.zfill(2)
            + "."
            + pl.col("code_plot").cast(pl.Utf8).str.zfill(4)
        ).alias("plot_id")
    )

    df_plots_raw = df_plots_raw.rename(
        {
            "latitude": "plot_latitude",
            "longitude": "plot_longitude",
            "slope": "plot_slope",
            "code_orientation": "plot_orientation",
            "code_altitude": "plot_altitude",
        }
    )
    return


@app.cell
def _(country_df, pl, species_df):
    df_growth_raw = pl.read_csv(
        "./data/raw/ICP/595_gr_20260306090551/gr_ipm.csv", separator=";", ignore_errors=True
    )

    df_growth_raw = df_growth_raw.with_columns(
        pl.col("date_assessment").str.to_datetime().alias("date")
    )

    df_growth_raw = df_growth_raw.with_columns(
        (
            pl.col("code_country").cast(pl.Utf8).str.zfill(2)
            + "."
            + pl.col("code_plot").cast(pl.Utf8).str.zfill(4)
            + "."
            + pl.col("tree_number").cast(pl.Utf8).str.zfill(5)
        ).alias("tree_id"),
        (
            pl.col("code_country").cast(pl.Utf8).str.zfill(2)
            + "."
            + pl.col("code_plot").cast(pl.Utf8).str.zfill(4)
        ).alias("plot_id"),
    )

    df_growth_raw = df_growth_raw.join(
        species_df.select(["code_tree_species", "specie"]), on="code_tree_species"
    )

    df_growth_raw = df_growth_raw.join(
        country_df.select(["code_country", "country"]), on="code_country"
    )
    return (df_growth_raw,)


@app.cell
def _(df_growth_raw, pl):
    print(f"Initial number of rows: {df_growth_raw.height}")

    # Drop all rows with null values in the 'diameter' column
    df_growth = df_growth_raw.drop_nulls(subset="diameter")
    print(f" `- after dropping nulls: {df_growth.height}")

    # Drop all rows where country is 'Belgium' or 'Spain'
    df_growth = df_growth.filter(~pl.col("country").is_in(["Belgium", "Spain"]))
    print(f" `- after dropping Belgium and Spain: {df_growth.height}")

    # Drop all rows with:
    # - diameter_quality_code is larger than 2 (implausible,
    #    https://icp-forests.org/documentation/Dictionaries/d_gr_quality_code.html)
    # - or diameter_method_code is in [7] (estimated diameter,
    # https://icp-forests.org/documentation/Dictionaries/d_diameter.html)
    # - or removal_code is larger than 10 (dead tree, see
    #    https://icp-forests.org/documentation/Dictionaries/d_removal_mortality_ccgr.html)
    # Keep null values in 'diameter_quality_code', 'diameter_method_code',
    # and 'removal_code' for now

    df_growth = df_growth.filter(
        pl.col("code_diameter_qc").cast(pl.Int64, strict=False).is_null()
        | ~pl.col("code_diameter_qc").cast(pl.Int64, strict=False).gt(2)
    )
    df_growth = df_growth.filter(
        pl.col("code_diameter").cast(pl.Int64, strict=False).is_null()
        | ~pl.col("code_diameter").cast(pl.Int64, strict=False).is_in([7])
    )
    df_growth = df_growth.filter(
        pl.col("code_removal").cast(pl.Int64, strict=False).is_null()
        | ~pl.col("code_removal").cast(pl.Int64, strict=False).gt(10)
    )

    print(f" `- after dropping quality codes 3-9: {df_growth.height}")

    # Drop rows with negative or zero diameter values
    df_growth = df_growth.filter(pl.col("diameter").gt(0))
    print(f" `- after dropping negative diameters: {df_growth.height}")

    df_growth = df_growth.select(
        "survey_year",
        "tree_id",
        "plot_id",
        "code_country",
        "country",
        "code_tree_species",
        "specie",
        "code_plot",
        "tree_number",
        "diameter",
        "height",
    )
    return (df_growth,)


@app.cell
def _(df_growth, pl):
    df_growth.filter(pl.col("plot_id") == "04.1401").group_by("survey_year").agg(
        avg_diameter=pl.col("diameter").mean()
    ).sort(by="survey_year")
    return


@app.cell
def _(country_df, pl):
    df_plot_info_raw = pl.read_csv(
        "./data/raw/ICP/595_gr_20260306090551/gr_pli.csv", separator=";", ignore_errors=True
    )

    df_plot_info_raw = df_plot_info_raw.with_columns(
        (
            pl.col("code_country").cast(pl.Utf8).str.zfill(2)
            + "."
            + pl.col("code_plot").cast(pl.Utf8).str.zfill(4)
        ).alias("plot_id"),
    )

    df_plot_info_raw = df_plot_info_raw.join(
        country_df.select(["code_country", "country"]), on="code_country"
    )

    df_plot_info_raw = df_plot_info_raw.drop_nulls(subset=["total_plot_size"])

    df_plot_info_raw = df_plot_info_raw.with_columns(
        pl.col("date_observation").str.to_datetime().alias("date")
    )

    df_plot_info = df_plot_info_raw.select(
        "survey_year",
        "code_country",
        "code_plot",
        "date",
        "latitude",
        "longitude",
        "total_plot_size",
        "plot_id",
        "country",
    )
    return (df_plot_info,)


@app.cell
def _(df_growth, df_plot_info):
    df_plot_info.join(df_growth, on="plot_id", how="inner")
    return


if __name__ == "__main__":
    app.run()
