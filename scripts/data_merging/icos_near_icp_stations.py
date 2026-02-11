"""
Script identifies ICOS stations that are geographically close to ICP Forests Level II stations.

It performs the following steps:
1. Downloads ICP station data from a specified URL.
2. Prepares the ICP station data for merging.
3. Fetches ICOS station data using the icoscp_core library.
4. Calculates the Haversine distance between ICP and ICOS stations to find nearby
    stations within a specified radius.
5. Plots the ICP and ICOS stations on an interactive map using Plotly, with options
    to show links between nearby stations and save the map as an HTML file.
6. Saves the merged data of nearby stations to a CSV file for further analysis.
"""

import polars as pl
import math
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
from icoscp_core.icos import meta


pio.renderers.default = "browser"


def download_from_url(url: str, output_file: str):
    """
    Download data from the given URL and save it to the specified output file.

    Parameters
    ----------
    url : str
        The URL to download the file from.
    output_file : str
        The local path where the downloaded file will be saved.

    Returns
    -------
    None
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {output_file}")
    else:
        print(f"Failed to download: HTTP {response.status_code}")


def prepare_icp_stations(csv_file: str) -> pl.DataFrame:
    """
    Load ICP station data from a CSV file and prepare it for merging.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing ICP station data.
        The CSV is expected to have columns:
        ["gid",  "code_country_iso", "lib_country",
            "lat_epsg4326", "lon_epsg4326", "tree_species"]

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing ICP station information with columns:
        ['gid', 'Lat', 'Lon', 'Country code', 'Country', 'Tree species']
    """
    df = pl.read_csv(csv_file, separator=";")

    df = df.select(
        ["gid", "code_country_iso", "lib_country", "lat_epsg4326", "lon_epsg4326", "tree_species"]
    )
    df = df.rename(
        {
            "code_country_iso": "Country code",
            "lib_country": "Country",
            "lat_epsg4326": "Lat",
            "lon_epsg4326": "Lon",
        }
    )

    df.write_csv("./data/intermediate/ICP_stations_locations.csv")

    return df


def get_icos_stations():
    """
    Get ICOS station data using the icoscp_core library.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing ICOS station information with columns:
        ['Label', 'Name', 'Country code', 'Lat', 'Lon']
    """
    # Fetch station data using the icoscp_core library
    icos_stations = meta.list_stations()
    print(icos_stations)
    # Convert to Polars DataFrame
    schema = ["Id", "Label", "Name", "Country code", "Lat", "Lon"]
    station_data = []
    for station in icos_stations:
        station_data.append(
            {
                "Id": station.type_uri,
                "Label": station.label,
                "Name": station.name,
                "Country code": station.country_code,
                "Lat": station.lat,
                "Lon": station.lon,
            }
        )
    station_data = pl.DataFrame(station_data, schema=schema)

    station_data.write_csv("./data/intermediate/ICOS_stations_locations.csv")

    return station_data


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the Haversine distance between two points on the Earth's surface.

    Ref: https://en.wikipedia.org/wiki/Haversine_formula
    Calculate the distance between two points on Earth using the Haversine formula.
    Given latitudes and longitudes in degrees.
    Returns distance in kilometers.

    Parameters
    ----------
    lat1 : float
        Latitude of the first point in degrees.
    lon1 : float
        Longitude of the first point in degrees.
    lat2 : float
        Latitude of the second point in degrees.
    lon2 : float
        Longitude of the second point in degrees.

    Returns
    -------
    float
        Distance between the two points in kilometers.
    """
    R = 6371  # Earth radius in kilometers

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c

    return distance


def plot_icp_icos_map(csv_file, save_map=True, html_file=None, map_plot=False, map_links=False):
    """
    Plot ICP and ICOS stations on a map and saves the interactive map as HTML.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing ICOS and ICP station data.
        Must include columns:
        ['ICP_Lat', 'ICP_Lon', 'ICP_gid', 'ICOS_Lat', 'ICOS_Lon', 'ICOS_Name']
    save_map : bool
        If True, saves the map as an HTML file.
    html_file : str
        Path to save the output HTML file (e.g., 'output_map.html').
    show_map : bool
        If True, displays the map inline (requires Jupyter notebook).

    Returns
    -------
    None
    """
    # Load data
    df = pd.read_csv(csv_file)

    fig = go.Figure()

    # Lines connecting ICP -> ICOS
    if map_links:
        for _, row in df.iterrows():
            fig.add_trace(
                go.Scattermapbox(
                    lat=[row["ICP_Lat"], row["ICOS_Lat"]],
                    lon=[row["ICP_Lon"], row["ICOS_Lon"]],
                    mode="lines",
                    line=dict(color="gray", width=1),
                    showlegend=False,
                )
            )

    # Plot ICP stations (blue)
    fig.add_trace(
        go.Scattermapbox(
            lat=df["ICP_Lat"],
            lon=df["ICP_Lon"],
            mode="markers",
            marker=dict(size=8, color="blue"),
            text=df["ICP_gid"].astype(str),
            name="ICP Stations",
            hovertemplate="ICP GID: %{text}<br>Lat: %{lat}, Lon: %{lon}<extra></extra>",
        )
    )

    # Plot ICOS stations (red) - drawn on top of ICP stations for better visibility
    fig.add_trace(
        go.Scattermapbox(
            lat=df["ICOS_Lat"],
            lon=df["ICOS_Lon"],
            mode="markers",
            marker=dict(size=8, color="red"),
            text="ICOS: "
            + df["ICOS_Name"]
            + "<br>Dist: "
            + df["Distance_km"].round(2).astype(str)
            + " km",
            name="ICOS Stations",
            hovertemplate="%{text}<br>Lat: %{lat}, Lon: %{lon}<extra></extra>",
        )
    )

    # Set map layout
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            zoom=4,
            center=dict(
                lat=df[["ICP_Lat", "ICOS_Lat"]].mean().mean(),
                lon=df[["ICP_Lon", "ICOS_Lon"]].mean().mean(),
            ),
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(x=0, y=1),
    )

    # Save map to HTML
    if save_map and html_file is not None:
        fig.write_html(html_file)
        print(f"Interactive map saved as {html_file}")

    # Show map inline if requested
    if map_plot:
        fig.show()


def find_nearby_stations(
    target_station: dict, icos_stations: pl.DataFrame, radius_km: float = 20.0
) -> pl.DataFrame:
    """
    Find ICOS stations within a specified radius of a target ICP station.

    Parameters
    ----------
    target_station (pl.Series): A Polars Series representing the target ICP station row,
                                expected to have 'Lat' and 'Lon' columns.
    icos_stations (pl.DataFrame): A Polars DataFrame containing all ICOS stations,
                                expected to have 'Id', 'Lat', and 'Lon' columns.
    radius_km (float): The radius in kilometers to search within.

    Returns
    -------
    pl.DataFrame: A DataFrame of ICOS stations within the specified radius,
                including 'Id', 'Lat', 'Lon', and 'Distance_km'.
    """
    nearby_icos_stations_data = []

    target_lat = target_station["Lat"]
    target_lon = target_station["Lon"]

    # Iterate through all ICOS stations
    for icos_station_row in icos_stations.iter_rows(named=True):
        icos_lat = icos_station_row["Lat"]
        icos_lon = icos_station_row["Lon"]

        # Skip rows with missing Lat/Lon for ICOS stations
        if icos_lat is None or icos_lon is None:
            continue

        distance = haversine_distance(target_lat, target_lon, icos_lat, icos_lon)

        if distance <= radius_km:
            nearby_icos_stations_data.append(
                {
                    "ICP_gid": target_station["gid"],
                    "ICP_Lon": target_lon,
                    "ICP_Lat": target_lat,
                    "ICOS_Id": icos_station_row["Id"],
                    "ICOS_Name": icos_station_row["Name"],
                    "ICOS_Lat": icos_lat,
                    "ICOS_Lon": icos_lon,
                    "Distance_km": distance,  # Add distance to the output
                }
            )

    # Convert the list of dicts back to a Polars DataFrame
    if nearby_icos_stations_data:
        return pl.DataFrame(nearby_icos_stations_data)
    else:
        return pl.DataFrame({})  # Return an empty DataFrame if no stations are found


def collect_nearby_stations(
    target_stations: pl.DataFrame,
    candidate_stations: pl.DataFrame,
    radius_km: float,
    output_path: str,
    map_plot: bool = True,
    map_links: bool = False,
) -> pl.DataFrame:
    """
    Collect nearby stations.

    Find nearby candidate stations for each target station within a radius
    and save the merged result to CSV.

    Parameters
    ----------
    target_stations : pl.DataFrame
        Stations to iterate over (e.g. ICP stations)
    candidate_stations : pl.DataFrame
        Stations to search within (e.g. ICOS stations)
    radius_km : float
        Search radius in kilometers
    output_path : str
        Path where the resulting CSV will be saved

    Returns
    -------
    pl.DataFrame
        Concatenated DataFrame of nearby stations
    """
    if target_stations.is_empty():
        raise ValueError("Target stations DataFrame is empty.")

    nearby_results = []

    for idx in range(len(target_stations)):
        target_row = target_stations.row(idx, named=True)
        print(f"Target Station: {target_row}")

        nearby = find_nearby_stations(
            target_station=target_row,
            icos_stations=candidate_stations,
            radius_km=radius_km,
        )

        print(
            f"Stations within {radius_km}km of target station "
            f"({target_row.get('Country code')} "
            f"{target_row.get('Lat')},{target_row.get('Lon')}):"
        )

        if not nearby.is_empty():
            nearby_results.append(nearby)
        else:
            print("No nearby stations found.")

    if not nearby_results:
        print("No nearby stations found for any target station.")
        return pl.DataFrame()

    result = pl.concat(nearby_results)
    result.write_csv(output_path)

    print("Shape:", result.shape)

    print("Result:")
    print(result.head())

    if map_plot:
        plot_icp_icos_map(
            csv_file=output_path,
            save_map=True,
            html_file=f"./data/intermediate/ICP_ICOS_map_{int(radius_km)}.html",
            map_plot=map_plot,
            map_links=map_links,
        )
    return result


if __name__ == "__main__":
    # ICP Forests Level II plots
    url = "https://icp-forests.org/open_data/level_ii/gpd/gpd_level_ii.csv"
    output_file = "./data/raw/ICP/gpd_level_ii.csv"
    download_from_url(url, output_file)

    # Prepare ICP station data for merging
    ICP_stations = prepare_icp_stations(output_file)

    # Fetch ICOS station data
    ICOS_stations = get_icos_stations()

    collect_nearby_stations(
        target_stations=ICP_stations,
        candidate_stations=ICOS_stations,
        radius_km=50.0,
        output_path="./data/intermediate/ICOS_near_ICP_stations.csv",
        map_plot=True,  # Set to true to generate the map, False to skip it.
        # Note that generating the map can be time-consuming if there are
        # many stations.
        map_links=True,  # In case you want to see which ICP station is linked to
        # which ICOS station on the map, set this to True.
        #  It will draw lines between them.
    )
