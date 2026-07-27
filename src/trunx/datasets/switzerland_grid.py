"""Build a regular ERA5-Land grid clipped to Switzerland's border."""

import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import requests
from shapely.geometry import Point

from trunx.config import clean_data_folder, raw_data_folder

NATURAL_EARTH_COUNTRIES_URL = (
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
)
ERA5_LAND_GRID_RESOLUTION = 0.1  # degrees


def download_natural_earth_countries(dest_dir: Path) -> Path:
    """Download and extract the Natural Earth admin-0 countries shapefile."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "ne_10m_admin_0_countries.zip"
    extract_dir = dest_dir / "ne_10m_admin_0_countries"
    shp_path = extract_dir / "ne_10m_admin_0_countries.shp"

    if not shp_path.exists():
        response = requests.get(NATURAL_EARTH_COUNTRIES_URL, timeout=60)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    return shp_path


def save_switzerland_boundary(countries_shp_path: Path, output_path: Path) -> gpd.GeoDataFrame:
    """Isolate Switzerland from the countries shapefile and save it."""
    countries = gpd.read_file(countries_shp_path)
    switzerland = countries[countries["ISO_A3"] == "CHE"][["ADMIN", "ISO_A3", "geometry"]]
    switzerland = switzerland.reset_index(drop=True)
    switzerland.to_file(output_path, driver="GPKG")
    return switzerland


def build_era5_grid_for_switzerland(
    boundary: gpd.GeoDataFrame, resolution: float = ERA5_LAND_GRID_RESOLUTION
) -> pl.DataFrame:
    """Build the regular ERA5-Land grid points that fall inside a boundary."""
    minx, miny, maxx, maxy = boundary.total_bounds

    lons = np.arange(np.floor(minx / resolution) * resolution, maxx, resolution)
    lats = np.arange(np.floor(miny / resolution) * resolution, maxy, resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    points = gpd.GeoDataFrame(
        {"Lat": lat_grid.ravel(), "Lon": lon_grid.ravel()},
        geometry=[Point(xy) for xy in zip(lon_grid.ravel(), lat_grid.ravel(), strict=True)],
        crs="EPSG:4326",
    )
    boundary_union = boundary.union_all()
    inside = points[points.within(boundary_union)].reset_index(drop=True)

    return pl.DataFrame(
        {
            "plot_id": [
                f"ERA5_CH_{lat:.1f}_{lon:.1f}"
                for lat, lon in zip(inside["Lat"], inside["Lon"], strict=True)
            ],
            "Lat": inside["Lat"].to_numpy(),
            "Lon": inside["Lon"].to_numpy(),
        }
    )


if __name__ == "__main__":
    shp_path = download_natural_earth_countries(raw_data_folder / "boundaries")
    switzerland_boundary = save_switzerland_boundary(
        shp_path, clean_data_folder / "switzerland_boundary.gpkg"
    )
    era5_points = build_era5_grid_for_switzerland(switzerland_boundary)
    era5_points.write_csv(clean_data_folder / "era5_switzerland_points.csv")
    print(f"Saved {era5_points.height} ERA5-Land grid points inside Switzerland")
