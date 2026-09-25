"""Interactive map comparing LWF, ICOS, ICP Forests, and FLUXNET sites."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook compares LWF, ICOS, ICP, and FLUXNET sites on a world map.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sources:
    - LWF https://lwf.wsl.ch/en/flaechen
    - ICP https://icp-forests.org/documentation/Introduction/index.html
    - ICOS https://www.icos-cp.eu/measurements/station-network
    - FluxNet https://data.fluxnet.org/data/
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import plotly.express as px
    import polars as pl

    return Path, pl, px


@app.cell
def _(Path):
    DATA_DIR = Path("data")


    NETWORKS = [

        "ICP Forests Level II",

        "LWF",

        "ICOS",

        "FLUXNET",

    ]
    return DATA_DIR, NETWORKS


@app.cell
def _():
    MAP_CENTER = {

        "lat": 50.0,

        "lon": 10.0,

    }


    MAP_ZOOM = 3.5
    return MAP_CENTER, MAP_ZOOM


@app.cell
def _(DATA_DIR):
    FLUXNET_FILE = DATA_DIR / "fluxnet_sites.csv"
    ICOS_FILE = DATA_DIR / "icos_sites.csv"
    ICP_FILE = DATA_DIR / "icp_sites.csv"
    LWF_FILE = DATA_DIR / "LWF_site_metadata.csv"
    return FLUXNET_FILE, ICOS_FILE, ICP_FILE, LWF_FILE


@app.cell
def _(FLUXNET_FILE, pl):
    fluxnet_raw = pl.read_csv(
        FLUXNET_FILE,
        infer_schema_length=10000,
    )
    return (fluxnet_raw,)


@app.cell
def _(fluxnet_raw, pl):
    fluxnet_sites = fluxnet_raw.select(

        pl.lit("FLUXNET").alias("network"),

        pl.col("site_id").cast(pl.String),

        pl.col("site_name").cast(pl.String),

        pl.lit(None).cast(pl.String).alias("country"),

        pl.col("location_lat").cast(pl.Float64).alias("latitude"),

        pl.col("location_long").cast(pl.Float64).alias("longitude"),

        pl.lit(None).cast(pl.Float64).alias("elevation_m"),

        pl.col("igbp").cast(pl.String).alias("site_type"),

        pl.lit(None).cast(pl.String).alias("status"),

        pl.col("first_year").cast(pl.Int64).alias("start_year"),

        pl.col("last_year").cast(pl.Int64).alias("end_year"),

        pl.lit("FLUXNET").alias("source"),

        pl.lit("https://fluxnet.org/").alias("source_url"),

    )
    return (fluxnet_sites,)


@app.cell
def _(fluxnet_sites):
    fluxnet_sites
    return


@app.cell
def _(ICOS_FILE, pl):
    icos_raw = pl.read_csv(

        ICOS_FILE,

        separator=",",

        infer_schema_length=10000,

    )
    return (icos_raw,)


@app.cell
def _(icos_raw, pl):
    icos_sites = (

        icos_raw

        .with_columns(

            pl.col("Position")

            .str.split(" ")

            .list.get(0)

            .cast(pl.Float64, strict=False)

            .alias("longitude"),


            pl.col("Position")

            .str.split(" ")

            .list.get(1)

            .cast(pl.Float64, strict=False)

            .alias("latitude"),

        )

        .select(

            pl.lit("ICOS").alias("network"),

            pl.col("Id").cast(pl.String).alias("site_id"),

            pl.col("Name").cast(pl.String).alias("site_name"),

            pl.col("Country").cast(pl.String).alias("country"),

            pl.col("latitude"),

            pl.col("longitude"),

            pl.col("Elevation above sea")

            .cast(pl.Float64, strict=False)

            .alias("elevation_m"),

            pl.col("Site type").cast(pl.String).alias("site_type"),

            pl.col("Station class").cast(pl.String).alias("status"),

            pl.lit(None).cast(pl.Int64).alias("start_year"),

            pl.lit(None).cast(pl.Int64).alias("end_year"),

            pl.lit("ICOS").alias("source"),

            pl.lit("https://www.icos-cp.eu/").alias("source_url"),

        )

    )
    return (icos_sites,)


@app.cell
def _(icos_raw, pl):
    pl.DataFrame(

        {

            "column": icos_raw.columns,

            "dtype": [str(dtype) for dtype in icos_raw.dtypes],

        }

    )
    return


@app.cell
def _(icos_sites):
    icos_sites
    return


@app.cell
def _(ICP_FILE, pl):
    icp_raw = pl.read_csv(

        ICP_FILE,

        separator=";",

        infer_schema_length=10000,

    )
    return (icp_raw,)


@app.cell
def _(icp_raw, pl):
    pl.DataFrame(

        {

            "column": icp_raw.columns,

            "dtype": [str(dtype) for dtype in icp_raw.dtypes],

        }

    )
    return


@app.cell
def _(icp_raw, pl):
    icp_sites = icp_raw.select(

        pl.lit("ICP Forests Level II").alias("network"),

        pl.col("code_plot").cast(pl.String).alias("site_id"),

        pl.col("code_plot").cast(pl.String).alias("site_name"),

        pl.col("lib_country").cast(pl.String).alias("country"),

        pl.col("lat_epsg4326").cast(pl.Float64).alias("latitude"),

        pl.col("lon_epsg4326").cast(pl.Float64).alias("longitude"),

        pl.col("altitude").cast(pl.Float64, strict=False).alias("elevation_m"),

        pl.lit(None).cast(pl.String).alias("site_type"),

        pl.col("plot_status").cast(pl.String).alias("status"),

        pl.col("date_install")

        .cast(pl.String)

        .str.slice(0, 4)

        .cast(pl.Int64, strict=False)

        .alias("start_year"),

        pl.lit(None).cast(pl.Int64).alias("end_year"),

        pl.lit("ICP Forests").alias("source"),

        pl.lit("https://www.icp-forests.net/").alias("source_url"),

    )
    return (icp_sites,)


@app.cell
def _(icp_sites):
    icp_sites
    return


@app.cell
def _(LWF_FILE, pl):
    lwf_raw = pl.read_csv(

        LWF_FILE,

        infer_schema_length=10000,

    )
    return (lwf_raw,)


@app.cell
def _(lwf_raw, pl):
    pl.DataFrame(

        {

            "column": lwf_raw.columns,

            "dtype": [str(dtype) for dtype in lwf_raw.dtypes],

        }

    )
    return


@app.cell
def _(lwf_raw, pl):
    lwf_sites = lwf_raw.select(
        pl.lit("LWF").alias("network"),
        pl.col("LWF_ID").cast(pl.String).alias("site_id"),
        pl.col("Site").cast(pl.String).alias("site_name"),
        pl.lit("Switzerland").alias("country"),
        pl.col("Latitude").cast(pl.Float64).alias("latitude"),
        pl.col("Longitude").cast(pl.Float64).alias("longitude"),
        pl.col("Altitude_mean_m")
        .cast(pl.Float64, strict=False)
        .alias("elevation_m"),
        pl.lit("Long-term forest ecosystem research site")
        .alias("site_type"),
        pl.lit("active").alias("status"),
        pl.col("Installation_date")
        .str.strptime(
            pl.Date,
            format="%d %B %Y",
            strict=False,
        )
        .dt.year()
        .cast(pl.Int64)
        .alias("start_year"),
        pl.lit(None).cast(pl.Int64).alias("end_year"),
        pl.lit("WSL LWF").alias("source"),
        pl.lit("https://lwf.wsl.ch/en/flaechen").alias("source_url"),
    )
    return (lwf_sites,)


@app.cell
def _(lwf_sites):
    lwf_sites
    return


@app.cell
def _(fluxnet_sites, icos_sites, icp_sites, lwf_sites, pl):
    network_sites = pl.concat(

        [

            fluxnet_sites,

            icos_sites,

            icp_sites,

            lwf_sites,

        ],

        how="vertical_relaxed",

    )
    return (network_sites,)


@app.cell
def _(network_sites, pl):
    sites_by_network = (

        network_sites

        .group_by("network")

        .agg(

            pl.len().alias("site_count"),

            pl.col("latitude").is_null().sum().alias("missing_latitude"),

            pl.col("longitude").is_null().sum().alias("missing_longitude"),

        )

        .sort("network")

    )


    sites_by_network
    return


@app.cell
def _(network_sites, pl):
    coordinate_check = network_sites.filter(

        pl.col("latitude").is_null()

        | pl.col("longitude").is_null()

        | (pl.col("latitude") < -90)

        | (pl.col("latitude") > 90)

        | (pl.col("longitude") < -180)

        | (pl.col("longitude") > 180)

    )


    coordinate_check
    return


@app.cell
def _(network_sites, pl):
    mapped_sites = network_sites.filter(

        pl.col("latitude").is_not_null()

        & pl.col("longitude").is_not_null()

        & pl.col("latitude").is_between(-90, 90)

        & pl.col("longitude").is_between(-180, 180)

    )
    return (mapped_sites,)


@app.cell
def _(network_sites, pl):
    duplicate_network_sites = (

        network_sites

        .group_by(["network", "site_id"])

        .agg(

            pl.len().alias("count"),

            pl.col("site_name").unique().alias("site_names"),

            pl.col("latitude").unique().alias("latitudes"),

            pl.col("longitude").unique().alias("longitudes"),

        )

        .filter(pl.col("count") > 1)

    )


    duplicate_network_sites
    return


@app.cell
def _(mapped_sites, pl):
    import math

    from sklearn.metrics.pairwise import haversine_distances

    MATCH_THRESHOLD_KM = 25.0

    network_a_sites = mapped_sites.filter(
        pl.col("network").is_in(
            ["LWF", "ICP Forests Level II"]
        )
    )

    network_b_sites = mapped_sites.filter(
        pl.col("network").is_in(
            ["ICOS", "FLUXNET"]
        )
    )

    site_records_a = network_a_sites.to_dicts()
    site_records_b = network_b_sites.to_dicts()

    # Convert coordinates from degrees to radians
    coords_a = [
        [
            math.radians(site["latitude"]),
            math.radians(site["longitude"]),
        ]
        for site in site_records_a
    ]

    coords_b = [
        [
            math.radians(site["latitude"]),
            math.radians(site["longitude"]),
        ]
        for site in site_records_b
    ]

    # Calculate all A-to-B Haversine distances at once.
    # haversine_distances returns angular distance in radians.
    distance_matrix_km = (
        haversine_distances(coords_a, coords_b) * 6371.0
    )

    candidate_matches = []

    for i, site_a in enumerate(site_records_a):
        closest_index = distance_matrix_km[i].argmin()
        closest_distance = distance_matrix_km[i, closest_index]
        closest_site = site_records_b[closest_index]

        if closest_distance <= MATCH_THRESHOLD_KM:
            candidate_matches.append(
                {
                    "network_a": site_a["network"],
                    "site_id_a": site_a["site_id"],
                    "site_name_a": site_a["site_name"],
                    "network_b": closest_site["network"],
                    "site_id_b": closest_site["site_id"],
                    "site_name_b": closest_site["site_name"],
                    "distance_km": closest_distance,
                }
            )

    candidate_matches = (
        pl.DataFrame(candidate_matches)
        .sort(["network_a", "site_name_a"])
    )
    return (candidate_matches,)


@app.cell
def _(candidate_matches):
    candidate_matches.sort("distance_km").unique(subset="site_name_b").sort(
        "distance_km"
    )
    return


@app.cell
def _(NETWORKS):
    import marimo as mo
    network_selector = mo.ui.multiselect(
        options=NETWORKS,
        value=NETWORKS,
        label="Networks",
    )
    return mo, network_selector


@app.cell
def _(mapped_sites, network_selector, pl):
    selected_sites = mapped_sites.filter(

        pl.col("network").is_in(network_selector.value)

    )
    return (selected_sites,)


@app.cell
def _(MAP_CENTER, MAP_ZOOM, px, selected_sites):
    fig = px.scatter_map(

        lat=selected_sites["latitude"].to_list(),

        lon=selected_sites["longitude"].to_list(),

        color=selected_sites["network"].to_list(),

        hover_name=selected_sites["site_name"].to_list(),

        hover_data={

            "network": selected_sites["network"].to_list(),

            "site_id": selected_sites["site_id"].to_list(),

            "country": selected_sites["country"].to_list(),

            "elevation_m": selected_sites["elevation_m"].to_list(),

        },

        zoom=MAP_ZOOM,

        center=MAP_CENTER,

        height=700,

    )


    fig.update_layout(

        map_style="carto-positron",

        margin={"r": 0, "t": 0, "l": 0, "b": 0},

        legend_title_text="Network",

    )


    fig
    return


if __name__ == "__main__":
    app.run()
