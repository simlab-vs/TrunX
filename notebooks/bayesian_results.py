"""Bayesian results analysis notebook."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os

    import arviz as az
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import polars as pl

    from trunx.config import data_folder, results_data_folder, threepg_data_folder

    return (
        az,
        data_folder,
        go,
        os,
        pl,
        plt,
        results_data_folder,
        threepg_data_folder,
    )


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian Results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Time Comparison
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    |Language | Num chains|Execution time per chain| Total time|
    |---|---|---|---|
    |R | 3|21 hours | 63 hours |
    |Python|3|13.86 hours | 13.86 hours|
    """)
    return


@app.cell
def _(os, results_data_folder):
    output_dir = os.path.join(results_data_folder, "results/pymc_inference_results")

    # idata = load_inference_data(os.path.join(output_dir, "inference_data.nc"))
    # predictions = load_predictions(
    #     os.path.join(output_dir, "predictions.npz")
    # )
    return (output_dir,)


@app.cell
def _(az, idata):
    physiological_names = [
        var for var in list(idata.posterior.data_vars) if not var.startswith("err_")
    ]

    summary = az.summary(idata, var_names=physiological_names)
    return physiological_names, summary


@app.cell
def _(idata):
    print(idata)
    return


@app.cell
def _():
    # for num_chain in range(3):
    #     values = idata.warmup_posterior["pFS20"][num_chain]
    #     ind = np.arange(0, len(values))
    #     plt.plot(ind, values, label= num_chain)
    # plt.show()
    return


@app.cell
def _(summary):
    print(summary)
    return


@app.cell
def _(mo, physiological_names):
    phys_names = mo.ui.multiselect(
        options=physiological_names, value=["pFS20", "aWS"], label="Select physiological names"
    )

    mo.hstack([phys_names])
    return (phys_names,)


@app.cell
def _(az, idata, phys_names, plt):
    az.plot_trace(idata, var_names=phys_names.value)
    az.plot_posterior(idata, var_names=phys_names.value)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3PG using inference data
    """)
    return


@app.cell
def _(data_folder, os, output_dir, plt, results_data_folder):
    from trunx.gp3.bayesiancalibrations.bayesian_comparison_plots import (
        plot_comparison,
    )

    _file_path = os.path.join(output_dir, "solling_data.xlsx")

    _hmc_output_dir = os.path.join(data_folder, "hmc_results")
    _plot_output_dir = os.path.join(results_data_folder, "bayesian_test_plot")
    os.makedirs(_plot_output_dir, exist_ok=True)

    _include_gradient_descent = False
    _include_bayesian = True
    _include_hmc = False

    _fig, _metrics_df = plot_comparison(
        _file_path,
        output_dir,
        _hmc_output_dir,
        include_gradient_descent=_include_gradient_descent,
        include_bayesian=_include_bayesian,
        include_hmc=_include_hmc,
    )

    _fig.savefig(
        os.path.join(_plot_output_dir, "prediction_comparison.png"), dpi=200, bbox_inches="tight"
    )

    # _conv_fig, _conv_df = plot_convergence_comparison(
    #     pymc_inference_path=os.path.join(output_dir, "inference_data.nc"),
    #     hmc_inference_path=os.path.join(_hmc_output_dir, "numpyro_inference_data.nc"),
    #     include_bayesian=_include_bayesian,
    #     include_hmc = _include_hmc
    # )

    # _conv_fig.savefig(
    #     os.path.join(_plot_output_dir, "convergence_comparison.png"),
    #     dpi=200,
    #     bbox_inches="tight"
    # )

    plt.show()
    return (plot_comparison,)


@app.cell
def _():
    return


@app.cell
def _(data_folder, os, plot_comparison, plt, results_data_folder):
    plot_id = "14.0003"

    _bayesian_dir = os.path.join(results_data_folder, f"results/pymc_inference_results_{plot_id}")
    _file_path = os.path.join(_bayesian_dir, f"{plot_id}_data.xlsx")

    _hmc_output_dir = os.path.join(data_folder, "hmc_results")
    _plot_output_dir = os.path.join(results_data_folder, "bayesian_test_plot")
    os.makedirs(_plot_output_dir, exist_ok=True)

    _include_gradient_descent = False
    _include_bayesian = True
    _include_hmc = False

    _fig, _metrics_df = plot_comparison(
        _file_path,
        _bayesian_dir,
        _hmc_output_dir,
        include_gradient_descent=_include_gradient_descent,
        include_bayesian=_include_bayesian,
        include_hmc=_include_hmc,
    )

    plt.show()
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grid Spatial Simulations
    """)
    return


@app.cell
def _(grid_input_path, pl):
    pl.read_parquet(grid_input_path).select("grid_id", "x", "y")
    return


@app.cell
def _(os, pl, results_data_folder, threepg_data_folder):
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

    grid_input_path = os.path.join(threepg_data_folder, "grid_input.parquet")
    grid_ids = pl.read_parquet(grid_input_path).select("grid_id", "x", "y")

    x_coords = grid_ids["x"].to_numpy()
    y_coords = grid_ids["y"].to_numpy()

    lon, lat = transformer.transform(x_coords, y_coords)

    # Add to DataFrame
    grid_ids = grid_ids.with_columns([pl.Series("lon", lon), pl.Series("lat", lat)])

    grid_ids = grid_ids.drop(["x", "y"])

    grid_outputs = pl.read_parquet(os.path.join(results_data_folder, "grid_3pg_outputs.parquet"))

    grid_outputs = (
        grid_outputs.select("grid_id", "param_idx", "WS", "WR", "WF")
        .group_by("grid_id")
        .agg(mean_WS=pl.col("WS").mean(), mean_WR=pl.col("WR").mean(), mean_WF=pl.col("WF").mean())
    )

    grid_outputs = grid_outputs.join(grid_ids, on="grid_id", how="inner")
    return grid_input_path, grid_outputs


@app.cell
def _(go, grid_outputs, pl):
    plot_locations_fig = go.Figure()

    plot_locations_fig.add_trace(
        go.Scattermap(
            lat=grid_outputs["lat"],
            lon=grid_outputs["lon"],
            mode="markers",
            marker=dict(
                size=3, color=grid_outputs["mean_WS"], colorscale="Plasma", showscale=True
            ),
        )
    )

    plot_locations_fig.update_layout(
        map=dict(
            style="open-street-map",
            zoom=7,
            center=dict(
                lat=grid_outputs.select(pl.mean("lat")).item(),
                lon=grid_outputs.select(pl.mean("lon")).item(),
            ),
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        legend=dict(x=0, y=1),
        title=dict(
            text="Tree locations",
            x=0.5,
            xanchor="center",
        ),
    )
    return


if __name__ == "__main__":
    app.run()
