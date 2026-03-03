import polars as pl
import streamlit as st
from support_utils import (
    load_prepare_data,
)

from trunx.plot_utils import (
    plot_geographic_location_species,
)

st.set_page_config(page_title="ICP Forests EDA", layout="wide")

# --- Page selection ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["About", "Explore locations", "EDA analysis"])

df = load_prepare_data()

if page == "About":
    st.title("ICP Forests - EDA")
    # st.success(f"Loaded {len(df):,} rows")
    st.write("Number of unique plots: " + str(df.select(pl.col("plot_id")).unique().shape[0]))

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


# filtered_df = df[
#     (df["Species"].isin(selected_species)) &
#     (df["Year"].between(*year_range))
# ]

# # -----------------------------
# # KPIs
# # -----------------------------
# col1, col2, col3 = st.columns(3)

# col1.metric("Plots", filtered_df["PlotID"].nunique())
# col2.metric("Trees", filtered_df["TreeID"].nunique())
# col3.metric("Records", len(filtered_df))

# # -----------------------------
# # Geographic map
# # -----------------------------
# st.subheader("Geographic Distribution of Species")

# map_df = (
#     filtered_df
#     .dropna(subset=["Lat", "Lon"])
#     .groupby(["PlotID", "Species", "Lat", "Lon"])
#     .size()
#     .reset_index(name="Count")
# )

# fig_map = px.scatter_mapbox(
#     map_df,
#     lat="Lat",
#     lon="Lon",
#     color="Species",
#     size="Count",
#     zoom=4,
#     height=500,
#     mapbox_style="carto-positron",
#     hover_data=["PlotID"]
# )

# st.plotly_chart(fig_map, use_container_width=True)

# # -----------------------------
# # Social class distribution
# # -----------------------------
# st.subheader("Social Class Distribution")

# if "SocialClass" in filtered_df.columns:
#     fig_social = px.histogram(
#         filtered_df,
#         x="SocialClass",
#         color="Species",
#         barmode="group"
#     )
#     st.plotly_chart(fig_social, use_container_width=True)
# else:
#     st.warning("Column 'SocialClass' not found.")

# # -----------------------------
# # Year-wise social class trend
# # -----------------------------
# st.subheader("Year-wise Social Class Trends")

# if {"Year", "SocialClass"}.issubset(filtered_df.columns):
#     trend_df = (
#         filtered_df
#         .groupby(["Year", "SocialClass"])
#         .size()
#         .reset_index(name="Count")
#     )

#     fig_trend = px.line(
#         trend_df,
#         x="Year",
#         y="Count",
#         color="SocialClass",
#         markers=True
#     )
#     st.plotly_chart(fig_trend, use_container_width=True)

# # -----------------------------
# # Variable histograms
# # -----------------------------
# st.subheader("Environmental Variable Distributions")

# numeric_cols = filtered_df.select_dtypes("number").columns.tolist()
# default_cols = [c for c in numeric_cols if c.startswith(("ss_", "dep_", "soph_"))]

# selected_vars = st.multiselect(
#     "Select variables",
#     numeric_cols,
#     default=default_cols[:4]
# )

# for var in selected_vars:
#     fig = px.histogram(
#         filtered_df,
#         x=var,
#         color="Species",
#         nbins=30,
#         marginal="box"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# # -----------------------------
# # Raw data preview
# # -----------------------------
# with st.expander("Show raw data"):
#     st.dataframe(filtered_df.head(1000))
