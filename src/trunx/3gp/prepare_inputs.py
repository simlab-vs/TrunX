"""Prepare all inputs for 3PG model."""

import polars as pl
from prepare_climate import prepare_climate
from prepare_parameters import prepare_parameters
from prepare_site import prepare_site
from prepare_sizeDist import prepare_sizeDist
from prepare_species import prepare_species
from prepare_thinning import prepare_thinning


def prepare_input(
    site,
    species,
    climate,
    i_parameters: pl.DataFrame,
    i_sizeDist: pl.DataFrame,
    thinning: pl.DataFrame | None = None,
    parameters: pl.DataFrame | None = None,
    size_dist: pl.DataFrame | None = None,
    settings: dict[str, int] | None = None,
):
    """Check and prepare all input tables for running the 3-PG model."""
    # Site
    site = prepare_site(site)

    # Species
    species = prepare_species(species)

    # Settings
    default_settings = {
        "light_model": 1,
        "transp_model": 1,
        "phys_model": 1,
        "height_model": 1,
        "correct_bias": 0,
        "calculate_d13c": 0,
    }

    if settings is not None:
        default_settings.update(settings)

    settings_out = default_settings

    # Climate
    if settings_out["calculate_d13c"] == 1:
        required_cols = {"co2", "d13catm"}
        if not required_cols.issubset(set(climate.columns)):
            raise ValueError(
                "Please provide forcing data for co2 and d13catm in climate, if calculate_d13c = 1"
            )

    climate = prepare_climate(
        climate=climate,
        from_=site["from"].item(),
        to=site["to"].item(),
    )

    # Thinning
    thinning = prepare_thinning(
        thinning=thinning,
        sp_names=species["species"].to_list(),
    )

    # Parameters
    parameters = prepare_parameters(
        i_parameters=i_parameters,
        parameters=parameters,
        sp_names=species["species"].to_list(),
    )

    # Size distribution
    if settings_out["correct_bias"] == 1 and size_dist is None:
        raise ValueError("Please provide size_dist table or change the setting correct_bias = 0")

    size_dist = prepare_sizeDist(
        i_sizeDist=i_sizeDist,
        size_dist=size_dist,
        sp_names=species["species"].to_list(),
    )

    # Output
    return {
        "site": site,
        "species": species,
        "climate": climate,
        "thinning": thinning,
        "parameters": parameters,
        "size_dist": size_dist,
        "settings": settings_out,
    }


if __name__ == "__main__":
    file_path = "./data/data.input.xlsx"
    site_sheet_name = "site"
    species_sheet_name = "species"
    climate_sheet_name = "climate"
    thinning_sheet_name = "thinning"
    sizeDist_sheet_name = "sizeDist"

    site = pl.read_excel(file_path, sheet_name=site_sheet_name)

    thinning = pl.read_excel(file_path, sheet_name=thinning_sheet_name)

    species = pl.read_excel(file_path, sheet_name=species_sheet_name)

    climate = pl.read_excel(file_path, sheet_name=climate_sheet_name)

    i_parameters = pl.read_csv("./data/i_parameters.csv")
    i_sizeDist = pl.read_csv("./data/i_sizeDist.csv")

    df = prepare_input(
        site=site,
        species=species,
        climate=climate,
        i_parameters=i_parameters,
        parameters=None,
        i_sizeDist=i_sizeDist,
        thinning=thinning,
    )

    print(df)
