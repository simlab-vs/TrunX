import pandas as pd
import polars as pl
import streamlit as st
from support_utils import (
    load_prepare_data,
)

from trunx.gp3.PG3_model_impl import run_threepg_main
from trunx.plot_utils import plot_combined_3pg_outputs_obv, plot_geographic_location_species

st.set_page_config(page_title="ICP Forests EDA", layout="wide")

# --- Page selection ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    [
        "About",
        "Explore locations",  # "EDA analysis",
        "3PG model",
    ],
)

old_df, new_df = load_prepare_data()


def get_DBH_data(tdf, plot_id):
    """Return DBH data for plotting."""
    filtered_df = tdf.filter(pl.col("specie").is_in(["Fagus sylvatica"]))
    df_growth = filtered_df.filter(pl.col("plot_id") == plot_id).to_pandas()

    avg_diameter = df_growth.groupby("period_end")["diameter_end"].mean().reset_index()
    return df_growth, avg_diameter


if page == "About":
    st.title("ICP Forests")
    st.subheader(
        "Old - Number of unique plots: " + str(old_df.select(pl.col("plot_id")).unique().shape[0])
    )

    species_list = old_df.select(pl.col("Species")).unique().to_series().to_list()
    data = {"Metric": ["Growth periods", "Unique trees", "Unique plots"]}

    for species in species_list:
        sdf = old_df.filter(pl.col("Species") == species)
        data[species] = [
            sdf.shape[0],
            sdf.select(pl.col("tree_id")).unique().shape[0],
            sdf.select(pl.col("plot_id")).unique().shape[0],
        ]

    summary_df = pl.DataFrame(data)
    st.dataframe(summary_df)

    st.subheader(
        "New - Number of unique plots: " + str(new_df.select(pl.col("plot_id")).unique().shape[0])
    )
    data = {"Metric": ["Growth periods", "Unique trees", "Unique plots"]}

    for species in species_list:
        sdf = new_df.filter(pl.col("Species") == species)
        data[species] = [
            sdf.shape[0],  # Growth periods
            sdf.select(pl.col("tree_id")).unique().shape[0],
            sdf.select(pl.col("plot_id")).unique().shape[0],
        ]

    summary_df = pl.DataFrame(data)
    st.dataframe(summary_df)


if page == "Explore locations":
    st.title("Geographic location of species")
    st.sidebar.header("Filters")

    species_list = new_df.select(pl.col("Species")).unique().to_series().to_list()
    selected_species = st.sidebar.multiselect("Select species", species_list, default=species_list)

    fig = plot_geographic_location_species(new_df, selected_species=selected_species)
    st.plotly_chart(fig, use_container_width=True)

if page == "3PG model":
    st.title("3PG model implementation")
    st.sidebar.header("Filters")

    file_choice = st.sidebar.selectbox(
        "Select dataset",
        [
            "Single species Trotsiuk data",
            "Trotsiuk data",
            "ICP data",
        ],
    )

    if file_choice == "Single species Trotsiuk data":
        file_path = "./data/data_sspecies_nothinning.xlsx"
        st.subheader(
            "Implementation using Trotsiuk eg. weather data \
            for beech (single species + no thinning)"
        )
    elif file_choice == "ICP data":
        file_path = "./data/data_semisynthetic.xlsx"
        st.subheader("Implementation using ICP weather data for beech (Plot id: 50.0013, CH)")
    else:
        st.subheader("Implementation using Trotsiuk eg. weather data for beech")
        file_path = "./data/data.input.xlsx"

    fig, outputs = run_threepg_main(file_path, plot_output=False, r_comparison=True)

    observed_data = None
    if file_choice == "ICP data":
        observed_data = get_DBH_data(
            new_df, plot_id="50.0013"
        )  # Your function that returns (df_growth, avg_diameter)

    comp_df = pd.read_csv("./data/r_python.comparison.csv")
    fig = plot_combined_3pg_outputs_obv(comp_df, observed_data=observed_data)

    st.pyplot(fig)

    if observed_data is not None:
        df_growth, avg_diameter = observed_data
        st.dataframe(avg_diameter.head())
