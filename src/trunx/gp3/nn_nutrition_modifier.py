"""Learnable componenet of the 3PG model: a nutrition modifier optimized using gradient descent."""

import os

import jax
import jax.nn
import jax.numpy as jnp
import optax
import polars as pl

from trunx.config import threepg_data_folder
from trunx.gp3.model_inputs import (
    ExtendedParams,
    MonthlyData,
    Params,
    SiteData,
    SpeciesData,
    State,
)
from trunx.gp3.prepare_data import prepare_data
from trunx.gp3.run_3pg import run_3pg


def make_loss_function(
    initial_state: State,
    monthly_data: MonthlyData,
    params: Params,
    site_data: SiteData,
    species_data: SpeciesData,
    extended_params: ExtendedParams,
):
    """Create a loss function for the 3PG model with a nutrition modifier."""

    def loss_function(extended_params):
        _, outputs = run_3pg(
            initial_state, monthly_data, params, site_data, species_data, extended_params
        )


if __name__ == "__main__":
    file_path = os.path.join(threepg_data_folder, "S_weather_data.xlsx")

    input_data = prepare_data(file_path)

    _, outputs = run_3pg(
        input_data.initial_state,
        input_data.climate,
        input_data.params,
        input_data.site,
        input_data.species,
    )

    extended_params = ExtendedParams(poly_params=jnp.ones((2, 2)) * 5)  # Example weights
    _, outputs_extended = run_3pg(
        input_data.initial_state,
        input_data.climate,
        input_data.params,
        input_data.site,
        input_data.species,
        extended_params,
    )

    print("WS (no nutrition modifier):", outputs["alpha_c"][0:5])
    print("WS (with nutrition modifier):", outputs_extended["alpha_c"][0:5])
