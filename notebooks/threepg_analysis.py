"""3PG analysis notebook."""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    import os

    import polars as pl

    from scripts.support_utils import load_prepare_data
    from trunx.config import data_folder
    from trunx.gp3.PG3_model_impl import run_threepg_main, run_threepg_with_icp

    return (
        data_folder,
        load_prepare_data,
        os,
        pl,
        run_threepg_main,
        run_threepg_with_icp,
    )


@app.cell
def _(data_folder, mo, os):
    preprocessed_data_files = [
        os.path.join(data_folder, "data.input.xlsx"),  # Data from r3PG vignette - multi species
        os.path.join(
            data_folder, "data_nothinning.xlsx"
        ),  # Data from r3PG vigneteer, with thinning data removed
        os.path.join(
            data_folder, "data_sspecies_nothinning.xlsx"
        ),  # Data from r3PG with single species and no thinning
        os.path.join(data_folder, "solling_data.xlsx"),  # Data from r3PG vignette of solling data
        os.path.join(data_folder, "Davos_data_GPP.xlsx"),  # Data from Mirko
    ]

    preprocessed_data_ui = mo.ui.dropdown(
        options=preprocessed_data_files,
        label="Select preprocessed data",
        value=os.path.join(data_folder, "solling_data.xlsx"),
    )

    mo.vstack([preprocessed_data_ui])
    return (preprocessed_data_ui,)


@app.cell
def _(preprocessed_data_ui, run_threepg_main):
    fig, outputs = run_threepg_main(
        preprocessed_data_ui.value, observed_data=None, plot_output=True, r_comparison=True
    )

    return


@app.cell
def _(load_prepare_data, mo, pl):
    _, icp_df = load_prepare_data()

    # Getting single species plots
    icp_df = icp_df.join(
        icp_df.group_by(["Lat", "Lon"]).agg(species_count=pl.col("Species").unique().count()),
        on=["Lat", "Lon"],
    ).filter(pl.col("species_count") == 1)

    plot_ids = icp_df["plot_id"].unique().to_list()

    plot_id_ui = mo.ui.dropdown(
        options=plot_ids,
        label="Select plot id",
        value="50.0018",  # Davos plot id
    )

    mo.vstack([plot_id_ui])
    return (plot_id_ui,)


@app.cell
def _(plot_id_ui, run_threepg_with_icp):
    icp_fig, icp_outputs = run_threepg_with_icp(
        plot_id=plot_id_ui.value, plot_output=True, r_comparison=True
    )

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
