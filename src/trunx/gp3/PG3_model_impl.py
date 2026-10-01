"""Base implementation of 3PG Model."""

# %%
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl
from jax import config, grad

from trunx.config import SPECIES_INDICES, project_root, threepg_data_folder

# config.update("jax_debug_nans", True)  # Enable NaN debugging
from trunx.gp3.create_combined_inputs import create_plot_input_file, load_plot_tables
from trunx.gp3.model_inputs import Params, State
from trunx.gp3.plot_function import (
    create_comparison_dataframe,
    plot_combined_3pg_outputs,
    plot_combined_3pg_outputs_obv,
    plot_dbh_distribution,
    plot_outputs,
)
from trunx.gp3.prepare_data import prepare_data
from trunx.gp3.run_3pg import run_3pg, ws_final, ws_final_vector

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

os.chdir(project_root)


def run_threepg_main(
    file_path,
    observed_data=None,
    plot_output=True,
    r_comparison=False,
    plot_id="",
    show_plots: bool = True,
):
    """Run 3PG model."""
    if file_path == "./data/data_sspecies_nothinning.xlsx":
        fig_name = "r_3PG_trotsiuk_nothinning"
    elif file_path == "./data/data.input.xlsx":
        fig_name = "r_3PG_trotsiuk"
    elif file_path == "./data/data_semisynthetic.xlsx":
        fig_name = "r_3PG_ICPdata"
    elif file_path == "./data/data_nothinning.xlsx":
        fig_name = "r_3PG_trotsiuk_mult_nothinning"
    else:
        fig_name = "ICP"

    # Plot from `all_observed` (surveys + monthly GPP) when the file has it
    try:
        sheet_names = pd.ExcelFile(file_path).sheet_names
        observed_sheet = "all_observed" if "all_observed" in sheet_names else "observed"
        observed_data = pd.read_excel(file_path, sheet_name=observed_sheet)
    except Exception as e:
        print(f"Could not read observed data from {file_path}: {e}")
        observed_data = None

    input_data = prepare_data(file_path)

    simulation_start_month = np.datetime64(
        f"{int(input_data.site.year_i[0]):04d}-{int(input_data.site.month_i[0]):02d}", "M"
    )

    final_state, outputs = run_3pg(
        initial_state=input_data.initial_state,
        climate=input_data.climate,
        params=input_data.params,
        site=input_data.site,
        species=input_data.species,
    )

    print("Final stem biomass (Mg/ha):", final_state.WS)
    print("Final LAI:", outputs["LAI"][-1])
    print("Final WS:", final_state.WS)

    params_vec = jnp.array(
        [
            input_data.params.alphaCx,
            input_data.params.CoeffCond,
            input_data.params.Y,
        ]
    )

    # Get Jacobian matrix (n_species × 3)
    jacobian = jax.jacobian(ws_final_vector)(
        params_vec,
        input_data.params,
        input_data.initial_state,
        input_data.climate,
        input_data.site,
        input_data.species,
    )

    # jacobian has shape (n_species, 3)
    for idx, specie in enumerate(input_data.species.specie):
        print(f"{specie}: [∂WS/∂alphaCx, ∂WS/∂CoeffCond, ∂WS/∂Y] = {jacobian[idx]}")

    if r_comparison and plot_output:
        from trunx.gp3.run_r3pg import run_comparison_r

        r_outputs = run_comparison_r(file_path)
        df_comp = create_comparison_dataframe(
            r_outputs, outputs, simulation_start_month, input_data.species_names
        )
        fig = plot_combined_3pg_outputs_obv(
            df_comp,
            observed_data=observed_data,
            fig_name=fig_name,
            plot_id=plot_id,
            show=show_plots,
            plot_metrics=["BA", "DBH", "Height", "WF", "WS", "WR", "GPP"],
            per_species=False,
        )
    elif r_comparison:
        from trunx.gp3.run_r3pg import run_comparison_r

        r_outputs = run_comparison_r(file_path)
        create_comparison_dataframe(
            r_outputs, outputs, simulation_start_month, input_data.species_names
        )
        fig = None
    elif plot_output:
        fig = plot_outputs(outputs, simulation_start_month, fig_name, show=show_plots)
    else:
        fig = None

    return fig, outputs


def run_threepg_for_plot(
    plot_id: str,
    source: str = "ICP",
    literature_source: str = "Trotsiuk",
    plot_output: bool = True,
    r_comparison: bool = True,
):
    """Run 3PG model on one plot, built from the combined plot and weather tables.

    Parameters
    ----------
    plot_id : str
        Plot identifier within `source`.
    source : str
        Dataset of the plot: "NFI", "EFM", "LWF" or "ICP".
    literature_source : str
        Literature table the parameter sheets are built from.
    plot_output : bool
        Whether to plot the model outputs.
    r_comparison : bool
        Whether to also run r3PG and compare.
    """
    file_path = os.path.join(threepg_data_folder, "S_weather_data.xlsx")
    create_plot_input_file(
        plot_id,
        source,
        file_path,
        *load_plot_tables(plot_id, source),
        literature_source=literature_source,
    )
    return run_threepg_main(
        file_path, plot_output=plot_output, r_comparison=r_comparison, plot_id=plot_id
    )


if __name__ == "__main__":
    # file_path = os.path.join(threepg_data_folder, "data_semisynthetic.xlsx")
    # file_path = os.path.join(threepg_data_folder, "data.input.xlsx")
    # file_path = os.path.join(threepg_data_folder, "data_sspecies_nothinning.xlsx")
    # file_path = os.path.join(threepg_data_folder, "data_nothinning.xlsx")
    # file_path = os.path.join(threepg_data_folder, "solling_data.xlsx")
    # file_path = os.path.join(threepg_data_folder, "davos_data.xlsx")
    # file_path = os.path.join(threepg_data_folder, "Davos_data_GPP.xlsx")

    # fig, outputs = run_threepg_main(
    #     file_path, observed_data=None, plot_output=True, r_comparison=True
    # )

    # file_path = os.path.join(threepg_data_folder, "S_weather_data.xlsx")
    # fig, outputs = run_threepg_main(
    #     file_path, observed_data=None, plot_output=True, r_comparison=True
    # )

    # species_plot_ids = {
    #     "Pinus sylvestris": [
    #         "01.0082",
    #         "04.1303",
    #         "51.0015",
    #         "53.0109",
    #         "53.0112",
    #         "53.0114",
    #         "53.0302",
    #         "53.0306",
    #         "53.0311",
    #         "53.0312",
    #         "53.0313",
    #         "53.0316",
    #         "53.0407",
    #         "53.0501",
    #         "53.0513",
    #         "53.0603",
    #         "53.0617",
    #         "53.0618",
    #         "53.0623",
    #         "59.0001",
    #         "59.0003",
    #     ],
    #     "Fagus sylvatica": ["04.0101", "04.0704", "08.0034", "53.0107"],
    #     "Picea abies": [
    #         "04.0302",
    #         "04.1402",
    #         "04.1403",
    #         "14.0017",
    #         "52.0010",
    #         "53.0701",
    #         "59.0008",
    #     ],
    # }

    # plot_ids = species_plot_ids["Pinus sylvestris"]

    plot_ids = ["50.0018"]
    for plot_id in plot_ids:
        # plot_dbh_distribution(
        #     plot_id=plot_id,
        #     file_path=os.path.join(clean_data_folder, "icp_tree_data.parquet"),
        #     kind="box",
        #     fig_name=f"ICP_{plot_id}_dbh_distribution",
        #     show=True,
        # )
        fig, outputs = run_threepg_for_plot(
            plot_id=plot_id, source="ICP", plot_output=True, r_comparison=True
        )
