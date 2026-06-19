"""ERA5 data processing for 3PG models."""

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

    import polars as pl

    from trunx.config import clean_data_folder, era5_data_folder

    return clean_data_folder, era5_data_folder, os, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Precipitation
    """)
    return


@app.cell
def _(era5_data_folder, os, pl):
    precip = pl.read_parquet(os.path.join(era5_data_folder, "era5_total_precipitation.parquet"))

    # Drop unwanted columns
    precip = precip.drop(["step", "point", "number", "surface", "valid_time"])

    precip = (
        precip.with_columns(time=pl.col("time").str.slice(0, 10))
        .with_columns(time=pl.col("time").str.to_datetime("%Y-%m-%d"))
        .with_columns(year=pl.col("time").dt.year(), month=pl.col("time").dt.month())
    )
    return (precip,)


@app.cell
def _(precip):
    era5_locations = precip.select(["latitude", "longitude"]).unique()
    return (era5_locations,)


@app.cell
def _(era5_locations, pl):
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(
        go.Scattermap(
            lat=era5_locations["latitude"],
            lon=era5_locations["longitude"],
            mode="markers",
            marker=dict(size=5),
            name="Tree locations",
        )
    )

    fig.update_layout(
        map=dict(
            style="open-street-map",
            zoom=4,
            center=dict(
                lat=era5_locations.select(pl.mean("latitude")).item(),
                lon=era5_locations.select(pl.mean("longitude")).item(),
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
def _(clean_data_folder, os, pl):
    icp_locations = pl.read_csv(os.path.join(clean_data_folder, "icp_plot_locations.csv"))

    icp_locations = icp_locations.with_columns(
        plot_id=pl.col("plot_id").map_elements(lambda x: f"{float(x):02.4f}")
    )
    return (icp_locations,)


@app.cell
def _(era5_locations, icp_locations, pl):
    from haversine import haversine

    results = [
        {
            "plot_id": icp_locations["plot_id"][icp_idx],
            "icp_lat": icp_locations["Lat"][icp_idx],
            "icp_lon": icp_locations["Lon"][icp_idx],
            "era5_lat": era5_locations["latitude"][era5_idx],
            "era5_lon": era5_locations["longitude"][era5_idx],
            "distance_km": haversine(
                (icp_locations["Lat"][icp_idx], icp_locations["Lon"][icp_idx]),
                (era5_locations["latitude"][era5_idx], era5_locations["longitude"][era5_idx]),
            ),
        }
        for icp_idx in range(len(icp_locations))
        for era5_idx in range(len(era5_locations))
    ]

    df_distances = pl.DataFrame(results)
    return (df_distances,)


@app.cell
def _(df_distances, pl):
    icp_era5_mapping_df = df_distances.join(
        df_distances.group_by(["plot_id", "icp_lat", "icp_lon"]).agg(pl.col("distance_km").min()),
        on=["plot_id", "icp_lat", "icp_lon", "distance_km"],
        how="inner",
    )

    print(icp_era5_mapping_df.head())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
