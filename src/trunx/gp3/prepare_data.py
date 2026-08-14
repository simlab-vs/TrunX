"""Prepare data for 3PG model."""

import logging

import jax.numpy as jnp
import polars as pl

from trunx.config import SPECIES_INDICES
from trunx.gp3.helper_function import is_dormant
from trunx.gp3.model_inputs import InputData, Params, State
from trunx.gp3.prepare_climate import prepare_climate
from trunx.gp3.prepare_deposition import prepare_deposition
from trunx.gp3.prepare_site import prepare_site
from trunx.gp3.prepare_species import prepare_species


def prepare_data(file_path: str, extended_params: dict | None = None) -> InputData:
    """Prepare data and initial state for 3PG model."""
    d_site = pl.read_excel(file_path, sheet_name="site")
    site_data, site_start, site_end = prepare_site(d_site)

    d_climate = pl.read_excel(file_path, sheet_name="climate")
    climate = prepare_climate(d_climate, str(site_start), str(site_end))

    deposition = prepare_deposition(d_climate, str(site_start), str(site_end))

    d_species = pl.read_excel(file_path, sheet_name="species")
    species_data = prepare_species(d_species)

    logging.info("Pre-processed species data for %d species", len(species_data.specie))

    params_df = pl.read_excel(file_path, sheet_name="parameters")

    param_names = params_df["parameter"].to_list()
    # species_names = [col for col in params_df.columns if col != "parameter"]
    species_indices = species_data.specie
    index_to_species = {index: name for name, index in SPECIES_INDICES.items()}
    species_names = [index_to_species[int(index)] for index in species_indices]
    values_matrix = params_df[species_names].to_numpy()

    params_dict = {}
    for i, param_name in enumerate(param_names):
        params_dict[param_name] = jnp.asarray(values_matrix[i, :])

    params = Params(**params_dict)

    # Check if start month is dormant
    start_month = site_data.month_i
    start_dormant = is_dormant(start_month, params.leafgrow, params.leaffall)
    initial_WF = jnp.where(start_dormant, jnp.asarray(0.0), species_data.WF)
    initial_WF_debt = jnp.where(start_dormant, species_data.WF, jnp.asarray(0.0))

    asw_min = jnp.where(
        site_data.ASW_min > site_data.ASW_max, site_data.ASW_max, site_data.ASW_min
    )
    asw_max = site_data.ASW_max

    # Clip ASW to [asw_min, asw_max] for each species
    initial_ASW = jnp.clip(site_data.ASW, asw_min, asw_max)

    n_species = len(species_data.specie)
    climate_year = int(site_data.year_i[0])
    climate_month = int(site_data.month_i[0])

    age_months = (climate_year - species_data.year_p) * 12 + (climate_month - species_data.month_p)

    initial_state = State(
        WF=initial_WF,
        WR=species_data.WR,
        WS=species_data.WS,
        N=species_data.N,
        ASW=jnp.full(n_species, initial_ASW, dtype=initial_ASW.dtype),
        age=age_months,
        WF_debt=initial_WF_debt,
        prev_month=jnp.full(
            n_species, 12 if start_month == 1 else start_month - 1, dtype=jnp.int32
        ),
    )
    input_data = InputData(
        initial_state=initial_state,
        climate=climate,
        params=params,
        site=site_data,
        species=species_data,
        n_species=n_species,
        species_names=species_names,
        extended_params=None,
        deposition=deposition,
    )
    return input_data
