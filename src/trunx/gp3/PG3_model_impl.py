"""Base implementation of 3PG Model."""

# %%
import os

import jax.numpy as jnp
import polars as pl
from jax import grad

from trunx.config import project_root
from trunx.gp3.helper_function import is_dormant
from trunx.gp3.model_inputs import Params, State
from trunx.gp3.plot_function import (
    create_comparison_dataframe,
    plot_combined_3pg_outputs,
    plot_outputs,
)
from trunx.gp3.prepare_climate import prepare_climate
from trunx.gp3.prepare_site import prepare_site
from trunx.gp3.prepare_species import prepare_species
from trunx.gp3.run_3pg import run_3pg, ws_final
from trunx.gp3.run_r3pg import run_comparison_r

os.chdir(project_root)


def run_threepg_main(file_path, plot_output=True, r_comparison=False):
    """Run 3PG model."""
    if file_path == "./data/data_sspecies_nothinning.xlsx":
        fig_name = "r_3PG_trotsiuk_nothinning.png"
    elif file_path == "./data/data.input.xlsx":
        fig_name = "r_3PG_trotsiuk.png"
    elif file_path == "./data/data_semisynthetic.xlsx":
        fig_name = "r_3PG_ICPdata.png"
    else:
        fig_name = None

    d_site = pl.read_excel(file_path, sheet_name="site")
    site_data = prepare_site(d_site)

    d_climate = pl.read_excel(file_path, sheet_name="climate")
    climate = prepare_climate(d_climate, str(site_data.site_start), str(site_data.site_end))

    d_species = pl.read_excel(file_path, sheet_name="species")
    species_data = prepare_species(d_species)

    params_df = pl.read_excel(file_path, sheet_name="parameters")

    default_params = dict(
        zip(params_df["parameter"].to_list(), params_df["Fagus sylvatica"].to_list(), strict=True)
    )
    params = Params(**default_params)

    # Check if start month is dormant
    start_month = 1  # January
    start_dormant = is_dormant(start_month, params.leafgrow, params.leaffall)

    if start_dormant:
        initial_WF = jnp.asarray(0.0)
        initial_WF_debt = species_data.WF
    else:
        initial_WF = species_data.WF
        initial_WF_debt = jnp.asarray(0.0)

    asw_min = jnp.where(
        site_data.ASW_min > site_data.ASW_max, site_data.ASW_max, site_data.ASW_min
    )
    asw_max = site_data.ASW_max
    initial_ASW = jnp.clip(site_data.ASW, asw_min, asw_max)

    initial_state = State(
        WF=jnp.asarray(initial_WF),
        WR=jnp.asarray(species_data.WR),
        WS=jnp.asarray(species_data.WS),
        N=jnp.asarray(species_data.N),
        ASW=jnp.asarray(initial_ASW),
        age=jnp.asarray(int(climate.start_month - species_data.planted)),
        WF_debt=jnp.asarray(initial_WF_debt),
        prev_month=jnp.asarray(12 if start_month == 1 else start_month - 1),
        water_runoff_polled=jnp.asarray(0.0)
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

    if r_comparison and plot_output:
        r_outputs = run_comparison_r(file_path)
        fig = plot_combined_3pg_outputs(r_outputs, outputs, climate.start_month, fig_name)
        create_comparison_dataframe(r_outputs, outputs, climate.start_month)
    elif r_comparison:
        r_outputs = run_comparison_r(file_path)
        create_comparison_dataframe(r_outputs, outputs, climate.start_month)
        fig = None
    elif plot_output:
        fig = plot_outputs(outputs, climate.start_month, fig_name)
    else:
        fig = None

    return fig, outputs


if __name__ == "__main__":
    # file_path = "./data/data_semisynthetic.xlsx"
    # file_path = "./data/data.input.xlsx"
    file_path = "./data/data_sspecies_nothinning.xlsx"

    fig, outputs = run_threepg_main(file_path, plot_output=True, r_comparison=True)

# %%
