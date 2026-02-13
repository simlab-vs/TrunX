"""Script to associate ICOS stations to ICP Forests plots based on distance.

To each ICP Forests plot, all ICOS staions within a given radius (1km by default)
are associated.

The resulting associations are plotted in an interactive map and saved
in the intermediate data folder under ICP_ICOS_association.csv.
"""

import plotly.graph_objects as go
import polars as pl
from haversine import haversine
from icoscp_core.icos import meta

from trunx.config import icp_raw_data_folder, intermediate_data_folder
from trunx.web_utils import download_from_url


def get_icos_stations() -> pl.DataFrame:
    """Fetch ICOS station data using the icoscp_core library.

    Returns
    -------
    pl.DataFrame
        ICOS station information with columns:
        ['Id', 'Label', 'Name', 'Country code', 'Lat', 'Lon']
    """
    icos_stations = meta.list_stations()
    station_data = [
        {
            "Id": station.type_uri,
            "Label": station.label,
            "Name": station.name,
            "Country code": station.country_code,
            "Lat": station.lat,
            "Lon": station.lon,
        }
        for station in icos_stations
    ]
    return pl.DataFrame(station_data)


def plot_icp_icos_map(
    df: pl.DataFrame,
    output_file: str | None = None,
) -> None:
    """Plot ICP-ICOS stations association on interactive map.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with columns ['ICP_Lat', 'ICP_Lon', 'ICP_gid',
        'ICOS_Lat', 'ICOS_Lon', 'ICOS_Name', 'Distance_km'].
    output_file : str | None
        Path to save the output HTML file.
    """
    fig = go.Figure()

    # draw links
    for row in df.iter_rows(named=True):
        fig.add_trace(
            go.Scattermap(
                lat=[row["ICP_Lat"], row["ICOS_Lat"]],
                lon=[row["ICP_Lon"], row["ICOS_Lon"]],
                mode="lines",
                line=dict(color="gray", width=1),
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scattermap(
            lat=df["ICP_Lat"].to_list(),
            lon=df["ICP_Lon"].to_list(),
            mode="markers",
            marker=dict(size=8, color="blue"),
            text=df["ICP_gid"].cast(pl.Utf8).to_list(),
            name="ICP Stations",
            hovertemplate="ICP GID: %{text}<br>Lat: %{lat}, Lon: %{lon}<extra></extra>",
        )
    )

    icos_text = df.select(
        (
            pl.lit("ICOS: ")
            + pl.col("ICOS_Name")
            + pl.lit("<br>Dist: ")
            + pl.col("Distance_km").round(2).cast(pl.Utf8)
            + pl.lit(" km")
        ).alias("text")
    )["text"].to_list()

    fig.add_trace(
        go.Scattermap(
            lat=df["ICOS_Lat"].to_list(),
            lon=df["ICOS_Lon"].to_list(),
            mode="markers",
            marker=dict(size=8, color="red"),
            text=icos_text,
            name="ICOS Stations",
            hovertemplate="%{text}<br>Lat: %{lat}, Lon: %{lon}<extra></extra>",
        )
    )

    fig.update_layout(
        map=dict(
            style="open-street-map",
            zoom=4,
            center=dict(
                lat=df.select(pl.mean("ICP_Lat")),
                lon=df.select(pl.mean("ICP_Lon")),
            ),
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(x=0, y=1),
    )

    fig.write_html(output_file)
    fig.show()


def find_nearby_stations(
    icp_stations: pl.DataFrame,
    icos_stations: pl.DataFrame,
    radius_km: float = 20.0,
) -> pl.DataFrame:
    """Find ICOS stations within a specified radius of ICP stations.

    Parameters
    ----------
    icp_stations : pl.DataFrame
        ICP stations with columns ['gid', 'lat_epsg4326', 'lon_epsg4326'].
    icos_stations : pl.DataFrame
        ICOS stations with columns ['Id', 'Name', 'Lat', 'Lon'].
    radius_km : float
        Search radius in kilometers.

    Returns
    -------
    pl.DataFrame
        Matched stations with columns ['ICP_gid', 'ICP_Lat', 'ICP_Lon',
        'ICOS_Id', 'ICOS_Name', 'ICOS_Lat', 'ICOS_Lon', 'Distance_km'].
    """
    icos_valid = icos_stations.filter(pl.col("Lat").is_not_null() & pl.col("Lon").is_not_null())

    results = []
    for icp_row in icp_stations.iter_rows(named=True):
        icp_lat, icp_lon = icp_row["lat_epsg4326"], icp_row["lon_epsg4326"]

        for icos_row in icos_valid.iter_rows(named=True):
            distance = haversine((icp_lat, icp_lon), (icos_row["Lat"], icos_row["Lon"]))
            if distance <= radius_km:
                results.append(
                    {
                        "ICP_gid": icp_row["gid"],
                        "ICP_Lat": icp_lat,
                        "ICP_Lon": icp_lon,
                        "ICOS_Id": icos_row["Id"],
                        "ICOS_Name": icos_row["Name"],
                        "ICOS_Lat": icos_row["Lat"],
                        "ICOS_Lon": icos_row["Lon"],
                        "Distance_km": distance,
                    }
                )

    return pl.DataFrame(results) if results else pl.DataFrame()


if __name__ == "__main__":
    # Download ICP stations data.
    url = "https://icp-forests.org/open_data/level_ii/gpd/gpd_level_ii.csv"
    output_file = icp_raw_data_folder / "gpd_level_ii.csv"
    download_from_url(url, str(output_file))
    icp_stations = pl.read_csv(output_file, separator=";")

    icos_stations = get_icos_stations()

    radius_km = 1.0
    result = find_nearby_stations(icp_stations, icos_stations, radius_km=radius_km)

    if result.is_empty():
        print("No nearby stations found.")
    else:
        output_path = intermediate_data_folder / "ICP_ICOS_association.csv"
        result.write_csv(output_path)

        plot_icp_icos_map(
            df=result,
            output_file=str(
                intermediate_data_folder / f"ICP_ICOS_assiciation_map_{int(radius_km)}.html"
            ),
        )
