import polars as pl
import streamlit as st
from support_utils import (
    load_prepare_data,
)

from trunx.plot_utils import (
    plot_geographic_location_species,
)

from trunx.gp3.PG3_model_impl import run_threepg_main

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

st.set_page_config(page_title="ICP Forests EDA", layout="wide")

# --- Page selection ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["About", "Explore locations", # "EDA analysis",
                                           "3PG model"])

df = load_prepare_data()


def get_DBH_plot():
    tdf = pd.read_pickle("./data/raw/ICP/icpf/03_tidy/icpf-level2_growth-periods_with-cc.pkl.gzip")
    tdf = pl.DataFrame(pl.from_pandas(tdf))
    filtered_df = tdf.filter(pl.col("specie").is_in(["Fagus sylvatica"]))
    df_growth = filtered_df.filter(pl.col("plot_id")== "50.0013").to_pandas()

    avg_diameter = df_growth.groupby("period_end")["diameter_end"].mean().reset_index()

    fig = plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=df_growth,
        x="period_end",
        y="diameter_end",
        hue="tree_id",   # use this for multiple trees
        marker="o",
        palette="tab10",
        legend=False,
        alpha = 0.5
    )

    sns.lineplot(
        data=avg_diameter,
        x="period_end",
        y="diameter_end",
        color="black",
        marker="o",
        linewidth=2.5,
        label="Average"  # optional: shows legend only for the average
    )

    plt.title("")
    plt.xlabel("Year")
    plt.ylabel("DBH [cm]")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return fig

if page == "About":
    st.title("ICP Forests")
    # st.success(f"Loaded {len(df):,} rows")
    st.subheader("Number of unique plots: " + str(df.select(pl.col("plot_id")).unique().shape[0]))

    species_list = df.select(pl.col("Species")).unique().to_series().to_list()
    # Prepare data dictionary for the table
    data = {"Metric": ["Growth periods", "Unique trees", "Unique plots"]}

    # Loop over species and calculate metrics
    for species in species_list:
        sdf = df.filter(pl.col("Species") == species)
        data[species] = [
            sdf.shape[0],  # Growth periods
            sdf.select(pl.col("tree_id")).unique().shape[0],  # Unique trees
            sdf.select(pl.col("plot_id")).unique().shape[0],  # Unique plots
        ]

    # Convert to Polars DataFrame for display
    summary_df = pl.DataFrame(data)

    # Show as table in Streamlit
    st.dataframe(summary_df)

if page == "Explore locations":
    st.title("Geographic location of species")
    st.sidebar.header("Filters")

    species_list = df.select(pl.col("Species")).unique().to_series().to_list()
    selected_species = st.sidebar.multiselect("Select species", species_list, default=species_list)

    fig = plot_geographic_location_species(df, selected_species=selected_species)
    st.plotly_chart(fig, use_container_width=True)

if page == "3PG model":
    st.title("3PG model implementation")
    st.sidebar.header("Filters")

    file_choice = st.sidebar.selectbox(
    "Select dataset",
    ["ICP data", "Trotsiuk data"]
    )
    # Map choice to path
    if file_choice == "ICP data":
        file_path = "./data/data_semisynthetic.xlsx"
        st.subheader("Implementation using ICP weather data for beech (Plot id: 50.0013, CH)")
    else:
        st.subheader("Implementation using Trotsiuk eg. weather data for beech")
        file_path = "./data/data.input.xlsx"
    
    fig = run_threepg_main(file_path)

    st.pyplot(fig)

    if file_choice == "ICP data":
        fig = get_DBH_plot()
        st.pyplot(fig)

            




