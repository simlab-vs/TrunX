"""ICP data processing for 3PG model implementation."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    from io import StringIO

    import pandas as pd
    import polars as pl
    import polars.selectors as cs
    import requests

    return StringIO, cs, pd, pl, requests


@app.cell
def _():
    SPECIES_MAPPING = {
        "Picea abies": "spruce",  # Norway Spruce
        "Pinus sylvestris": "pine",  # Scots Pine
        "Fagus sylvatica": "beech",  # Common Beech
        "Quercus petraea": "oak",  # Sessile Oak
        "Quercus robur": "oak",  # Pedunculate Oak
    }
    return (SPECIES_MAPPING,)


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

    print(df_plots_raw.height)
    return (df_plots_raw,)


@app.cell
def _(SPECIES_MAPPING, country_df, pl, species_df):
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

    df_growth_raw = df_growth_raw.filter(pl.col("specie").is_in(SPECIES_MAPPING.keys()))

    print(df_growth_raw)
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
        "date",
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

    print(df_growth)
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
        "trees_number",
        "country",
    )

    print(df_plot_info)
    return


@app.cell
def _(cs, df_growth, df_plots_raw, pl):
    # Join to plot information
    PLOT_COLS = [
        "plot_latitude",
        "plot_longitude",
        "plot_slope",
        "plot_orientation",
        "plot_size",
        "plot_altitude",
    ]

    # We require at least one of the plot columns to be non-null
    df_growth_plot = df_growth.join(
        df_plots_raw.select("plot_id", *PLOT_COLS),
        on="plot_id",
        how="left",
    ).filter(pl.any_horizontal(cs.by_name(*PLOT_COLS).is_not_null()))

    print(f"Number of rows after joining with plot information: {df_growth_plot.height}")

    df_growth.head()
    return (df_growth_plot,)


@app.cell
def _():
    return


@app.cell
def _(pl):
    import os

    from trunx.config import clean_data_folder, data_folder

    df = pl.read_parquet(os.path.join(clean_data_folder, "icp_level2_cleaned.parquet"))

    df = df.filter(pl.col("plot_id") == "5.0010")

    df.select("period_end", "specie").unique()
    return data_folder, os


@app.cell
def _(cs, data_folder, df_growth_plot, os, pl):
    df_soil_raw = pl.read_csv(
        os.path.join(data_folder, "raw/ICP/595_ss_20260309084900/ss_ssm.csv"), separator=";"
    )

    df_soil_raw = df_soil_raw.with_columns(
        [
            pl.col("date_start").str.to_datetime().alias("date_start"),
            pl.col("date_end").str.to_datetime().alias("date_end"),
        ]
    )

    df_soil_raw = df_soil_raw.with_columns(
        (
            pl.col("code_country").cast(pl.Utf8).str.zfill(2)
            + "."
            + pl.col("code_plot").cast(pl.Utf8).str.zfill(4)
        ).alias("plot_id")
    )

    print("Number of rows in raw soil solutions data:", df_soil_raw.height)

    df_soil_raw = df_soil_raw.rename(
        {"conductivity": "cond", "alkalinity": "alk", "n_total": "n_tot"}
    )

    soil_solution_names = [
        "ph",
        "cond",
        "k",
        "ca",
        "mg",
        "n_no3",
        "s_so4",
        "alk",
        "al",
        "doc",
        "na",
        "n_nh4",
        "cl",
        "n_tot",
        "fe",
        "mn",
        "al_labile",
        "p",
        "cr",
        "ni",
        "zn",
        "cu",
        "pb",
        "cd",
        "si",
        "n_no2",
        "n_no3_plus_n_no2",
    ]

    rename_soil_dict = {col: f"ss_{col}" for col in soil_solution_names}

    # Rename columns
    df_soil_raw = df_soil_raw.rename(rename_soil_dict)

    # Convert to float types
    df_soil_raw = df_soil_raw.with_columns(
        [pl.col(col_name).cast(pl.Float64) for col_name in rename_soil_dict.values()]
    )

    # Drop rows with null or negative values of sample_vol
    df_soil = (
        df_soil_raw.filter(
            pl.col("sample_vol").cast(pl.Float64, strict=False).is_null()
            | pl.col("sample_vol").cast(pl.Float64, strict=False).gt(0)
        )
        # Replace invalid values in soil solutions data
        .with_columns(
            pl.when(cs.starts_with("ss_").is_between(0.0001, 10000))
            .then(cs.starts_with("ss_"))
            .otherwise(None)
        )
        .group_by("plot_id", "survey_year")
        .agg(
            pl.mean("sample_vol").cast(pl.Float64, strict=False).alias("sample_vol"),
            cs.starts_with("ss_").mean().name.keep(),
            pl.len().alias("num_soil_obs"),
        )
        .select(
            "plot_id",
            "survey_year",
            "sample_vol",
            "num_soil_obs",
            "ss_ph",
            "ss_cond",
            "ss_k",
            "ss_ca",
            "ss_mg",
            "ss_n_no3",
            "ss_s_so4",
            "ss_alk",
            "ss_al",
            "ss_doc",
            "ss_na",
            "ss_n_nh4",
            "ss_cl",
            "ss_n_tot",
            "ss_fe",
            "ss_mn",
            "ss_al_labile",
            "ss_p",
            "ss_cr",
            "ss_ni",
            "ss_zn",
            "ss_cu",
            "ss_pb",
            "ss_cd",
            "ss_si",
        )
    )

    df_soil_with_period = (
        df_soil.join(
            df_growth_plot.select("plot_id", "tree_id", "date"),
            on="plot_id",
            how="inner",
        )
        .with_columns(
            period_start_year=pl.col("date").dt.year() - 5,
            period_end_year=pl.col("date").dt.year() + 5,
        )
        .filter(
            pl.col("survey_year").is_between(
                pl.col("period_start_year"), pl.col("period_end_year")
            )
        )
        .group_by("tree_id", "date")
        .agg(
            cs.starts_with("ss_").mean(),
            pl.sum("num_soil_obs").alias("num_soil_obs"),
        )
        .select("tree_id", "date", "num_soil_obs", cs.starts_with("ss_"))
    )
    return (df_soil_with_period,)


@app.cell
def _(cs, data_folder, df_growth_plot, os, pl):
    df_deposition_raw = pl.read_csv(
        os.path.join(data_folder, "raw/ICP/595_dp_20260309084812/dp_dem.csv"), separator=";"
    )

    df_deposition_raw = df_deposition_raw.with_columns(
        [
            pl.col("date_start").str.to_datetime().alias("date_start"),
            pl.col("date_end").str.to_datetime().alias("date_end"),
        ]
    )

    df_deposition_raw = df_deposition_raw.with_columns(
        (
            pl.col("code_country").cast(pl.Utf8).str.zfill(2)
            + "."
            + pl.col("code_plot").cast(pl.Utf8).str.zfill(4)
        ).alias("plot_id")
    )

    print(f"Number of rows in deposition data: {df_deposition_raw.height}")

    df_deposition_raw = df_deposition_raw.rename(
        {
            "n_total": "n_tot",
            "c_total": "c_tot",
            "s_total": "s_tot",
            "p_total": "p_tot",
            "conductivity": "cond",
            "alkalinity": "alk",
        }
    )

    deposition_names = [
        "ph",
        "cond",
        "k",
        "ca",
        "mg",
        "na",
        "n_nh4",
        "cl",
        "n_no3",
        "s_so4",
        "alk",
        "n_tot",
        "doc",
        "al",
        "mn",
        "fe",
        "p_po4",
        "cu",
        "zn",
        "hg",
        "pb",
        "co",
        "mo",
        "ni",
        "cd",
        "s_tot",
        "c_tot",
        "n_org",
        "p_tot",
        "cr",
        "n_no2",
        "hco3",
        "don",
        "n_no3_plus_n_no2",
    ]
    rename_dep_dict = {col: f"dep_{col}" for col in deposition_names}
    non_conc_dep_cols = ["dep_alk", "dep_ph", "dep_cond"]

    df_deposition_raw = df_deposition_raw.rename(rename_dep_dict)

    df_deposition = (
        df_deposition_raw.filter(
            pl.col("date_start").is_not_null()
            & pl.col("date_end").is_not_null()
            & pl.col("code_sampler").eq(1)
        )
        .with_columns([pl.col(col).cast(pl.Float64) for col in rename_dep_dict.values()])
        .filter(~pl.col("code_vsampling").is_in([2, 3, 4, 7, 9]))
        .filter(~pl.col("code_sampler").eq(8))
        .with_columns(
            pl.when(cs.starts_with("dep_").exclude(*non_conc_dep_cols).ne(-1.0))
            .then(cs.starts_with("dep_").exclude(*non_conc_dep_cols))
            .otherwise(None)
        )
        .with_columns(cs.starts_with("dep_").fill_nan(None))
        .with_columns(
            dep_n_tot=pl.when(pl.col("dep_n_tot").is_null())
            .then(pl.col("dep_n_nh4") + pl.col("dep_n_no3") + pl.col("dep_n_org").fill_null(0))
            .otherwise(pl.col("dep_n_tot"))
        )
        # Convert concentrations (mg/l) × quantity (l/m²) → fluxes (mg/m²) → kg/ha or g/ha
        .with_columns(cs.starts_with("dep_").exclude(*non_conc_dep_cols) * pl.col("quantity"))
        .with_columns(cs.starts_with("dep_").exclude(*non_conc_dep_cols) / 100)
        # Aggregate to annual level per plot
        .group_by("plot_id", "survey_year")
        .agg(
            cs.starts_with("dep_").exclude(*non_conc_dep_cols).sum(),
            cs.by_name(*non_conc_dep_cols).mean(),
            pl.col("quantity").sum().alias("yearly_precip"),
            pl.len().alias("num_deposition_obs"),
        )
    )

    print(f" `- after aggregating to annual level: {df_deposition.height}")

    # Integrate over ±5 year window around each census date (average annual deposition)
    df_deposition_with_period = (
        df_deposition.join(
            df_growth_plot.select("plot_id", "tree_id", "date"),
            on="plot_id",
            how="inner",
        )
        .with_columns(
            period_start_year=pl.col("date").dt.year() - 5,
            period_end_year=pl.col("date").dt.year() + 5,
        )
        .filter(
            pl.col("survey_year").is_between(
                pl.col("period_start_year"), pl.col("period_end_year")
            )
        )
        .group_by("tree_id", "date")
        .agg(
            cs.starts_with("dep_").exclude(*non_conc_dep_cols).mean(),
            cs.by_name(*non_conc_dep_cols).mean(),
            pl.mean("yearly_precip").alias("yearly_precip"),
            pl.sum("num_deposition_obs").alias("num_deposition_obs"),
        )
        .select("tree_id", "date", "num_deposition_obs", "yearly_precip", cs.starts_with("dep_"))
    )

    print(f" `- after integrating over census window: {df_deposition_with_period.height}")

    return (df_deposition_with_period,)


@app.cell
def _(data_folder, df_growth_plot, os, pl, species_df):
    df_crown_raw = pl.read_csv(
        os.path.join(data_folder, "raw/ICP/595_cc_20260309084712/cc_trc.csv"), separator=";"
    )

    df_crown_raw = df_crown_raw.with_columns(
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

    df_crown_raw = df_crown_raw.with_columns(pl.col("date_survey").str.to_datetime().alias("date"))
    df_crown_raw = df_crown_raw.rename({"code_social_class": "social_class_code"})

    df_crown_raw = df_crown_raw.join(
        species_df.select(["code_tree_species", "specie"]), on="code_tree_species"
    )

    # Prepare the crown condition data
    print(f"Number of rows in crown condition data: {df_crown_raw.height}")

    # Dropping all rows with null or negative defoliation
    df_crown = df_crown_raw.filter(
        pl.col("code_defoliation").cast(pl.Int64, strict=False).is_not_null()
        & pl.col("code_defoliation").cast(pl.Int64, strict=False).ge(0)
    ).with_columns(defoliation=pl.col("code_defoliation").cast(pl.Int32))
    print(f"Number of rows with valid defoliation: {df_crown.height}")

    df_crown = (
        df_crown.sort(by="date")
        .join_asof(
            df_growth_plot.select(
                "tree_id",
                "date",
                "specie",
                pl.col("date").alias("census_date"),
            ).sort(["tree_id", "specie", "date"]),
            by=["tree_id", "specie"],
            on="date",
            strategy="forward",
            suffix="_gp",
        )
        .drop_nulls(subset="census_date")
        .filter(
            pl.col("date").is_between(
                pl.col("census_date") - pl.duration(days=int(365.25 * 5)),
                pl.col("census_date"),
            )
        )
    )

    print(
        f"Number of rows after merging crown condition data: \
        {df_crown.height}"
    )

    df_crown = (
        df_crown.group_by("tree_id", "census_date", "specie")
        .agg(
            pl.len().alias("num_defoliation_obs"),
            # Defoliation statistics
            pl.mean("defoliation").alias("defoliation_mean"),
            pl.min("defoliation").alias("defoliation_min"),
            pl.max("defoliation").alias("defoliation_max"),
            pl.median("defoliation").alias("defoliation_median"),
            pl.last("defoliation").alias("defoliation_last"),
            # Social status statistics
            pl.min("social_class_code").alias("social_class_min"),
            pl.max("social_class_code").alias("social_class_max"),
            pl.col("social_class_code").mode().first().alias("social_class_mode"),
            pl.last("social_class_code").alias("social_class_last"),
            # Dominance indicators
            pl.col("social_class_code").eq(1).any().alias("was_dominant"),
            pl.col("social_class_code").eq(2).any().alias("was_codominant"),
            pl.col("social_class_code").eq(3).any().alias("was_subdominant"),
            pl.col("social_class_code").eq(4).any().alias("was_suppressed"),
            pl.col("social_class_code").eq(5).any().alias("was_dying"),
        )
        .rename({"census_date": "date"})
    )
    print(f"Number of rows after aggregating crown condition data: {df_crown.height}")

    # Drop rows where defoliation reached 100% (dead trees)
    df_crown = df_crown.filter(pl.col("defoliation_max").lt(100))
    print(f"Number of rows after dropping dead trees: {df_crown.height}")

    # Drop rows with less than two defoliation observations
    df_crown = df_crown.filter(pl.col("num_defoliation_obs").gt(1))
    print(
        f"Number of rows after dropping trees with less than two observations: \
            {df_crown.height}"
    )
    return


@app.cell
def _(df_deposition_with_period, df_growth_plot, df_soil_with_period):
    df_growth_all = df_growth_plot.join(
        df_soil_with_period, on=["tree_id", "date"], how="left"
    ).join(df_deposition_with_period, on=["tree_id", "date"], how="left")

    print("Number of rows after merging soil solutions data:", df_growth_all.height)
    return


if __name__ == "__main__":
    app.run()
