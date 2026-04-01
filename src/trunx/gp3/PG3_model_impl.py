"""Base implementation of 3PG Model."""

# %%
import os

import jax
import jax.numpy as jnp
import pandas as pd
import polars as pl
from jax import config, grad

from trunx.config import clean_data_folder, icp_raw_data_folder, project_root

# config.update("jax_debug_nans", True)  # Enable NaN debugging
from trunx.datasets.ICP_weather_data import prepare_icp_weather_data
from trunx.gp3.create_data_inputs import create_input_data, create_weather_input
from trunx.gp3.helper_function import is_dormant
from trunx.gp3.model_inputs import Params, State
from trunx.gp3.plot_function import (
    create_comparison_dataframe,
    plot_combined_3pg_outputs,
    plot_combined_3pg_outputs_obv,
    plot_combined_3pg_outputs_per_species,
    plot_outputs,
)
from trunx.gp3.prepare_climate import prepare_climate
from trunx.gp3.prepare_site import prepare_site
from trunx.gp3.prepare_species import prepare_species
from trunx.gp3.run_3pg import run_3pg, ws_final, ws_final_vector
from trunx.gp3.run_r3pg import run_comparison_r

os.chdir(project_root)


def prepare_data(file_path):
    """Prepare data and initial state for 3PG model."""
    d_site = pl.read_excel(file_path, sheet_name="site")
    site_data = prepare_site(d_site)

    d_climate = pl.read_excel(file_path, sheet_name="climate")
    climate = prepare_climate(d_climate, str(site_data.site_start), str(site_data.site_end))

    d_species = pl.read_excel(file_path, sheet_name="species")
    species_data = prepare_species(d_species)

    params_df = pl.read_excel(file_path, sheet_name="parameters")

    param_names = params_df["parameter"].to_list()
    species_names = [col for col in params_df.columns if col != "parameter"]
    values_matrix = params_df[species_names].to_numpy()

    params_dict = {}
    for i, param_name in enumerate(param_names):
        params_dict[param_name] = values_matrix[i, :]

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
    climate_year, climate_month = (
        pd.to_datetime(climate.start_month).year,
        pd.to_datetime(climate.start_month).month,
    )

    age_months = (climate_year - species_data.year_p) * 12 + (climate_month - species_data.month_p)

    initial_state = State(
        WF=initial_WF,
        WR=species_data.WR,
        WS=species_data.WS,
        N=species_data.N,
        ASW=jnp.full(n_species, initial_ASW, dtype=jnp.float32),
        age=age_months,
        WF_debt=initial_WF_debt,
        prev_month=jnp.full(n_species, 12 if start_month == 1 else start_month - 1),
    )

    return initial_state, climate, params, site_data, species_data, n_species


def run_threepg_main(file_path, observed_data=None, plot_output=True, r_comparison=False):
    """Run 3PG model."""
    if file_path == "./data/data_sspecies_nothinning.xlsx":
        fig_name = "r_3PG_trotsiuk_nothinning.png"
    elif file_path == "./data/data.input.xlsx":
        fig_name = "r_3PG_trotsiuk.png"
    elif file_path == "./data/data_semisynthetic.xlsx":
        fig_name = "r_3PG_ICPdata.png"
    elif file_path == "./data/data_nothinning.xlsx":
        fig_name = "r_3PG_trotsiuk_mult_nothinning.png"
    else:
        fig_name = None

    initial_state, climate, params, site_data, species_data, n_species = prepare_data(file_path)

    final_state, outputs = run_3pg(
        initial_state=initial_state,
        climate=climate,
        params=params,
        site=site_data,
        species=species_data,
        n_species=n_species,
    )

    print("Final stem biomass (Mg/ha):", final_state.WS)
    print("Final LAI:", outputs["LAI"][-1])
    print("Final WS:", final_state.WS)

    params_vec = jnp.array(
        [
            params.alphaCx,
            params.CoeffCond,
            params.Y,
        ]
    )

    # Get Jacobian matrix (n_species × 3)
    jacobian = jax.jacobian(ws_final_vector)(
        params_vec, params, initial_state, climate, site_data, species_data, n_species
    )

    # jacobian has shape (n_species, 3)
    for idx, specie in enumerate(species_data.specie):
        print(f"{specie}: [∂WS/∂alphaCx, ∂WS/∂CoeffCond, ∂WS/∂Y] = {jacobian[idx]}")

    if r_comparison and plot_output:
        r_outputs = run_comparison_r(file_path)
        # fig = plot_combined_3pg_outputs(
        #     r_outputs, outputs, climate.start_month, species_data.specie, fig_name
        # )
        df_comp = create_comparison_dataframe(r_outputs, outputs, climate.start_month)
        fig = plot_combined_3pg_outputs_obv(df_comp, observed_data=observed_data)
    elif r_comparison:
        r_outputs = run_comparison_r(file_path)
        create_comparison_dataframe(r_outputs, outputs, climate.start_month)
        fig = None
    elif plot_output:
        fig = plot_outputs(outputs, climate.start_month, fig_name)
    else:
        fig = None

    return fig, outputs


def run_threepg_with_icp(plot_id=None, plot_output=True, r_comparison=True):
    """Run 3PG model with ICP weather data."""
    file_path = os.path.join("./data/", "S_weather_data.xlsx")
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    miss_months, observed_data = create_input_data(file_path, plot_id)
    if len(miss_months) == 0:
        fig, outputs = run_threepg_main(
            file_path, observed_data, plot_output=plot_output, r_comparison=r_comparison
        )
        return fig, outputs
    else:
        print(
            "The weather data is not complete and need pre-processing before 3PG implementation."
        )
        return None, None


if __name__ == "__main__":
    # # file_path = "./data/data_semisynthetic.xlsx"
    # # file_path = "./data/data.input.xlsx"
    # file_path = "./data/data_sspecies_nothinning.xlsx"
    # file_path = "./data/data_nothinning.xlsx"
    # fig, outputs = run_threepg_main(file_path,
    #                                   observed_data = None,
    #                                   plot_output=True,
    #                                   r_comparison=True
    # )

    plot_id = "50.0013"
    fig, outputs = run_threepg_with_icp(plot_id=plot_id, plot_output=True, r_comparison=True)
