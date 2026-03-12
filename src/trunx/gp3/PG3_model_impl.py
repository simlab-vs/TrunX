"""Base implementation of 3PG Model."""

import jax.numpy as jnp
import polars as pl
from jax import grad

from trunx.gp3.helper_function import plot_outputs, run_3pg, ws_final
from trunx.gp3.model_inputs import Params, State
from trunx.gp3.plot_function import plot_combined_3pg_outputs
from trunx.gp3.prepare_climate import prepare_climate
from trunx.gp3.prepare_site import prepare_site
from trunx.gp3.prepare_species import prepare_species
from trunx.gp3.run_r3pg import run_comparison_r


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

    if r_comparison and plot_output:
        r_outputs = run_comparison_r(file_path)
        fig = plot_combined_3pg_outputs(r_outputs, outputs, climate.start_month, fig_name)
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
