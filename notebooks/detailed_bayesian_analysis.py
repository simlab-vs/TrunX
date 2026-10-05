"""Detailed analysis of the Bayesian calibrations of 3PG, per plot and across plots."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayesian Calibration Overview

    This section provides an overview of how Bayesian calibrations are initiated for the 3PG model.

    ## Input Data

    We start with field‑measured values of DBH and tree height.
    Using allometric equations from
    [Forrester et al.](https://www.sciencedirect.com/science/article/pii/S0378112717301238),
    these DBH values are used to compute derived variables, including:

    - Stem biomass
    - Root biomass
    - Foliage biomass
    - Basal area

    This yields a total of **six observed and derived variables** that serve as the basis for
    Bayesian calibration. This method is commonly used for biomass determination
    (see [Trotsiuk et al.](https://onlinelibrary.wiley.com/doi/epdf/10.1111/gcb.15011)).

    ## Parameter Priors

    For the calibration, parameter priors are adopted from [Forrester et al.](https://link.springer.com/article/10.1007/s10342-021-01370-3):

    - Priors for the **18 calibrated parameters** are taken directly from their
      species‑specific values.
    - All other non‑calibrated parameters are also sourced from this paper.
    - The **posterior estimates** reported by Forrester et al. are used as **initial values**
      for the Markov Chain Monte Carlo (MCMC) chains.

    ## DBH Calculation in the 3PG Model

    In the 3PG model, DBH is calculated using the following equation:

    $$
    DBH = \left( \frac{\text{biomass per tree}}{aWS} \right)^{1/nWS}
    $$

    where

    $$
    \text{stem biomass per tree} =
    \frac{\text{stem biomass} \times 1000}{\text{num trees per hectare}}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # For each plot

    Figures saved by `bayesian_comparison_plots.plot_and_save`:

    - Predictions of the default, gradient descent, DEz (PyMC) and NUTS (HMC) runs vs observations
    - Convergence diagnostics
    - Posterior distributions of both samplers, with the gradient descent fit
    - Calibrated parameter values across methods
    - Trace and posterior per sampler
    """)
    return


@app.cell
def _(mo):
    import os

    from trunx.gp3.bayesiancalibrations.bayesian_comparison_plots import (
        literature_plot_ids,
        plot_posterior_across_plots,
        run_dir,
    )
    from trunx.gp3.bayesiancalibrations.bayesian_config import ERROR_MODES, species_plot_ids

    def show_image(path: str) -> object:
        """Display a saved figure, or a warning if it hasn't been produced yet.

        Parameters
        ----------
        path : str
            Path of the PNG figure.

        Returns
        -------
        object
            The marimo image or warning callout.
        """
        if not os.path.exists(path):
            return mo.callout(mo.md(f"Missing `{path}`"), kind="warn")
        return mo.image(src=path)

    return (
        ERROR_MODES,
        literature_plot_ids,
        os,
        plot_posterior_across_plots,
        run_dir,
        show_image,
        species_plot_ids,
    )


@app.cell(hide_code=True)
def _(ERROR_MODES, mo):
    ui_process_error = mo.ui.switch(label="Latent initial biomass (process error)")
    ui_literature_source = mo.ui.dropdown(
        options=["Forrester", "Trotsiuk"], value="Forrester", label="Literature source"
    )
    ui_error_terms = mo.ui.dropdown(
        options=list(ERROR_MODES), value="all_error_terms", label="Error terms"
    )
    mo.hstack([ui_process_error, ui_literature_source, ui_error_terms], justify="start")
    return ui_error_terms, ui_literature_source, ui_process_error


@app.cell(hide_code=True)
def _(literature_plot_ids, mo, species_plot_ids, ui_literature_source):
    plot_ids = literature_plot_ids(ui_literature_source.value)
    # Plots of each species calibrated with the selected literature source; Solling is a
    # Picea abies stand calibrated independently of the literature source
    plots_by_species = {
        species: [plot_id for plot_id in ids if plot_id in plot_ids]
        for species, ids in species_plot_ids.items()
    }
    plots_by_species["Picea abies"] = ["solling", *plots_by_species.get("Picea abies", [])]
    plots_by_species = {species: ids for species, ids in plots_by_species.items() if ids}
    ui_species = mo.ui.tabs({species: "" for species in plots_by_species}, label="Species")
    mo.output.replace(ui_species)
    return plot_ids, plots_by_species, ui_species


@app.cell(hide_code=True)
def _(mo, plots_by_species, ui_species):
    species_plots = plots_by_species[ui_species.value]
    ui_plot_id = mo.ui.dropdown(options=species_plots, value=species_plots[0], label="Plot ID")
    mo.output.replace(ui_plot_id)
    return (ui_plot_id,)


@app.cell(hide_code=True)
def _(
    run_dir,
    ui_error_terms,
    ui_literature_source,
    ui_plot_id,
    ui_process_error,
):
    combo_dir = run_dir(
        ui_plot_id.value,
        ui_literature_source.value,
        ui_error_terms.value,
        ui_process_error.value,
    )
    return (combo_dir,)


@app.cell(hide_code=True)
def _(combo_dir, mo, os, show_image, ui_plot_id):
    mo.vstack(
        [
            mo.md(f"## Plot {ui_plot_id.value}\n\n`{combo_dir}`"),
            mo.md("### Predictions"),
            show_image(os.path.join(combo_dir, "plots/prediction_comparison.png")),
            mo.md("### Convergence"),
            show_image(os.path.join(combo_dir, "plots/convergence_comparison.png")),
            mo.md("### Posterior comparison"),
            show_image(os.path.join(combo_dir, "plots/posterior_comparison.png")),
            mo.md("### Parameter values"),
            show_image(os.path.join(combo_dir, "plots/parameter_value_comparison.png")),
        ]
    )
    return


@app.cell(hide_code=True)
def _(combo_dir, mo, os, show_image):
    mo.vstack(
        [
            mo.md("### Trace and posterior per sampler"),
            mo.ui.tabs(
                {
                    label: mo.vstack(
                        [
                            show_image(os.path.join(combo_dir, method, "trace.png")),
                            show_image(os.path.join(combo_dir, method, "posterior.png")),
                        ]
                    )
                    for label, method in [("DEz (PyMC)", "demetropolisz"), ("NUTS", "nuts")]
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Posteriors across plots

    Overlays each parameter's posterior over the ICP plots of the selected species and
    literature source (`bayesian_comparison_plots.plot_posterior_across_plots`); Solling is
    left out.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    ui_method = mo.ui.dropdown(
        options={"DEz (PyMC)": "demetropolisz", "NUTS": "nuts"}, value="NUTS", label="Sampler"
    )
    ui_run_across = mo.ui.run_button(label="Load posteriors")
    mo.hstack([ui_method, ui_run_across], justify="start")
    return ui_method, ui_run_across


@app.cell(hide_code=True)
def _(
    mo,
    plot_posterior_across_plots,
    plots_by_species,
    ui_error_terms,
    ui_literature_source,
    ui_method,
    ui_process_error,
    ui_run_across,
    ui_species,
):
    mo.stop(not ui_run_across.value)
    plot_posterior_across_plots(
        [plot_id for plot_id in plots_by_species[ui_species.value] if plot_id != "solling"],
        ui_literature_source.value,
        ui_error_terms.value,
        ui_method.value,
        ui_process_error.value,
    )
    return


if __name__ == "__main__":
    app.run()
