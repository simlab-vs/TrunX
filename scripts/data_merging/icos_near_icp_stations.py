import polars as pl
import math
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "notebook"


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

def plot_icp_icos_map(csv_file, save_map=True, html_file=None, show_map=False, map_links=False):
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

    # Plot ICOS stations (red)
    fig.add_trace(
        go.Scattermapbox(
            lat=df["ICOS_Lat"],
            lon=df["ICOS_Lon"],
            mode="markers",
            marker=dict(size=8, color="red"),
            text=df["ICOS_Name"],
            name="ICOS Stations",
            hovertemplate="ICOS: %{text}<br>Lat: %{lat}, Lon: %{lon}<extra></extra>",
        )
    )

    # Lines connecting ICP -> ICOS
    if map_links:
        for _, row in df.iterrows():
            fig.add_trace(
                go.Scattermapbox(
                    lat=[row["ICP_Lat"], row["ICOS_Lat"]],
                    lon=[row["ICP_Lon"], row["ICOS_Lon"]],
                    mode="lines",
                    line=dict(color="gray", width=1),
                    hoverinfo="none",
                    showlegend=False,
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
    if show_map:
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
            show_map=False,  # set True to display inline in Jupyter
        )
    return result


if __name__ == "__main__":
    # Load ICOS and ICP station data
    ICOS_stations = pl.read_csv("./data/intermediate/ICOS_stations_locations.csv")
    ICP_stations = pl.read_csv("./data/intermediate/ICP_stations_locations.csv")

    collect_nearby_stations(
        target_stations=ICP_stations,
        candidate_stations=ICOS_stations,
        radius_km=10.0,
        output_path="./data/intermediate/ICOS_near_ICP_stations.csv",
        map_plot=True,  # Set to true to generate the map, False to skip it. 
                        # Note that generating the map can be time-consuming if there are
                        # many stations.
        map_links=True,  # In case you want to see which ICP station is linked to 
                         # which ICOS station on the map, set this to True.
                         #  It will draw lines between them.
    
    )
