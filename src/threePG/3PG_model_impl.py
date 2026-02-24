"""Base implementation of 3PG Model."""

import jax.numpy as jnp
import polars as pl
from helper_function import plot_outputs, run_3pg, ws_final
from jax import grad
from model_inputs import Params, State
from read_excel_data import read_climate_data, read_site_data, read_species_data

if __name__ == "__main__":
    climate = read_climate_data("./data/data.input.xlsx", sheet_name="climate")

    site_data = read_site_data("./data/data.input.xlsx", sheet_name="site")

    species_data = read_species_data("./data/data.input.xlsx", sheet_name="species")

    params_df = pl.read_excel("./data/data.input.xlsx", sheet_name="parameters")

    default_params = dict(
        zip(params_df["parameter"].to_list(), params_df["Fagus sylvatica"].to_list(), strict=True)
    )
    params = Params(**default_params)

    initial_state = State(
        WF=jnp.asarray(species_data.WF),
        WR=jnp.asarray(species_data.WR),
        WS=jnp.asarray(species_data.WS),
        N=jnp.asarray(species_data.N),
        ASW=jnp.asarray(site_data.ASW),
        age=jnp.asarray(int(climate.start_month - species_data.planted)),
    )

    final_state, outputs = run_3pg(
        initial_state=initial_state,
        climate=climate,
        params=params,
        site=site_data,
        species=species_data,
    )

    print("Final stem biomass (Mg/ha):", final_state.WS)
    print("Final LAI:", outputs["LAI"][-1])

    grad_alpha = grad(ws_final, argnums=0)(
        params.alphaCx,
        params.CoeffCond,
        params.Y,
        params,
        initial_state,
        climate,
        site_data,
        species_data,
    )

    grad_Kg = grad(ws_final, argnums=1)(
        params.alphaCx,
        params.CoeffCond,
        params.Y,
        params,
        initial_state,
        climate,
        site_data,
        species_data,
    )

    grad_Y = grad(ws_final, argnums=2)(
        params.alphaCx,
        params.CoeffCond,
        params.Y,
        params,
        initial_state,
        climate,
        site_data,
        species_data,
    )

    print("Final WS:", final_state.WS)
    print("∂WS/∂alphaCx:", grad_alpha)
    print("∂WS/∂CoeffCond:", grad_Kg)
    print("∂WS/∂Y:", grad_Y)

    plot_outputs(outputs)
