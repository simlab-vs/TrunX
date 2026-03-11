"""Read site, species and climate data from excel data file."""

import jax.numpy as jnp
import numpy as np
import polars as pl

from trunx.gp3.model_inputs import ClimateData, SiteData, SpeciesData


def read_site_data(filepath: str, sheet_name: str) -> SiteData:
    """Read site data from excel data file."""
    df = pl.read_excel(filepath, sheet_name=sheet_name)

    # Take the first row
    row = df.row(0, named=True)

    site_data = SiteData(
        latitude=float(row["latitude"]),
        altitude=float(row["altitude"]),
        social_class=int(row["soil_class"]),
        ASW=float(row["asw_i"]),
        ASW_max=float(row["asw_max"]),
        ASW_min=float(row["asw_min"]),
        site_start=np.datetime64(row["from"]),
        site_end=np.datetime64(row["to"]),
    )
    return site_data


def read_species_data(
    filepath: str, sheet_name: str
) -> SpeciesData:  # Replace with List[SpeciesData] for full adaptation of code
    """Read species data from excel data file."""
    df = pl.read_excel(filepath, sheet_name=sheet_name)

    species_data: list[SpeciesData] = []

    for row in df.iter_rows(named=True):
        year_p, month_p = map(int, row["planted"].split("-"))

        sp = SpeciesData(
            FR=float(row["fertility"]),
            WF=float(row["biom_foliage"]),
            WR=float(row["biom_root"]),
            WS=float(row["biom_stem"]),
            N=float(row["stems_n"]),
            planted=np.datetime64(row["planted"], "M"),
        )

        species_data.append(sp)
    print(species_data[0])
    return species_data[0]  # This need to be modified later to adapt for other species


def saturation_vapor_pressure(T: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate saturation vapour pressure.

    Parameters
    ----------
    T in °C

    Returns
    -------
    kPa
    """
    kPa = 6.108 * jnp.exp((17.27 * T) / (T + 237.3))
    return kPa


def vpd_from_tmin_tmax(tmp_min: jnp.ndarray, tmp_max: jnp.ndarray) -> jnp.ndarray:
    """
    Monthly VPD estimate for 3PG.

    Parameters
    ----------
    tmp_min
        floar array with minimum temperature.
    tmp_max
        float array with maximum temperature.
    """
    es_min = saturation_vapor_pressure(tmp_min)
    es_max = saturation_vapor_pressure(tmp_max)
    return (es_max - es_min) / 2 


def read_climate_data(file_path: str, sheet_name: str) -> ClimateData:
    """Read climate data from excel data file."""
    df = pl.read_excel(file_path, sheet_name=sheet_name)
    n_years = len(df.select(pl.col("year")).unique())
    tmp_min = jnp.array(df["tmp_min"])
    tmp_max = jnp.array(df["tmp_max"])

    start_month = np.datetime64(
        str(df.select(pl.col("year")).min().item())
        + "-"
        + str(df.select(pl.col("month")).min().item()).zfill(2),
        "M",
    )

    climate_data = ClimateData(
        T_avg=jnp.array(df["tmp_ave"].to_numpy()),
        precip=jnp.array(df["prcp"].to_numpy()),
        solar_rad=jnp.array(df["srad"].to_numpy()),
        frost_days=jnp.array(df["frost_days"].to_numpy()),
        n_days=jnp.tile(jnp.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]), n_years),
        VPD=vpd_from_tmin_tmax(tmp_min, tmp_max),
        start_month=start_month,
    )

    return climate_data


if __name__ == "__main__":
    # site_data = read_site_data("./data/data.input.xlsx", sheet_name="site")

    # species_data = read_species_data("./data/data.input.xlsx", sheet_name="species")

    file_path = "./data/data.input.xlsx"
    sheet_name = "climate"

    climate_data = read_climate_data(file_path, sheet_name)

    print(climate_data.VPD)
