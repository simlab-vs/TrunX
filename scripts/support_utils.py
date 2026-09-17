import pandas as pd
import polars as pl
import streamlit as st

from trunx.gp3.create_data_inputs import dms_to_decimal

SPECIES_NAME_MAP = {
    "Picea abies": "Spruce",
    "Pinus sylvestris": "Pine",
    "Fagus sylvatica": "Beech",
    "Quercus robur": "Oak",
    "Quercus petraea": "Oak",
}


@st.cache_data
def load_prepare_data():
    tdf = pd.read_pickle("./data/raw/ICP/icpf/03_tidy/icpf-level2_growth-periods_with-cc.pkl.gzip")
    tdf = pl.DataFrame(pl.from_pandas(tdf))
    tdf = tdf.with_columns(
        pl.col("plot_latitude").map_elements(dms_to_decimal, return_dtype=pl.Float64).alias("Lat"),
        pl.col("plot_longitude")
        .map_elements(dms_to_decimal, return_dtype=pl.Float64)
        .alias("Lon"),
    )

    filtered_df = tdf.filter(pl.col("specie").is_in(list(SPECIES_NAME_MAP.keys())))
    filtered_df = filtered_df.with_columns(
        pl.col("specie").replace(SPECIES_NAME_MAP).alias("Species")
    )

    new_tdf = pl.read_parquet("./data/clean/icp_level2_cleaned.parquet")
    new_tdf = new_tdf.with_columns(
        pl.col("plot_latitude").map_elements(dms_to_decimal, return_dtype=pl.Float64).alias("Lat"),
        pl.col("plot_longitude")
        .map_elements(dms_to_decimal, return_dtype=pl.Float64)
        .alias("Lon"),
    )

    new_filtered_df = new_tdf.filter(pl.col("specie").is_in(list(SPECIES_NAME_MAP.keys())))
    new_filtered_df = new_filtered_df.with_columns(
        pl.col("specie").replace(SPECIES_NAME_MAP).alias("Species")
    )
    return filtered_df, new_filtered_df


def species_summary(title, df):
    sdf = df.filter(pl.col("Species") == title)
    st.subheader(title)
    col1, col2, col3 = st.columns([1, 1, 1])

    col1.metric("Growth periods", sdf.shape[0])
    col2.metric("Unique trees", sdf.select(pl.col("tree_id")).unique().shape[0])
    col3.metric("Unique plots", sdf.select(pl.col("plot_id")).unique().shape[0])
