"""Paper figures: R vs Python runtime, Morris sensitivity, Bayesian calibration.

Solling site, biomass_only error terms (predictions vs observations included
there), plus spatial simulation — greenish color theme throughout.
"""

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

    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from cycler import cycler
    from plotly.subplots import make_subplots
    from pyproj import Transformer

    from trunx.config import (
        clean_data_folder,
        images_folder,
        results_data_folder,
        threepg_data_folder,
    )

    return (
        Transformer,
        clean_data_folder,
        cycler,
        go,
        gpd,
        images_folder,
        make_subplots,
        np,
        os,
        pl,
        plt,
        results_data_folder,
        threepg_data_folder,
    )


@app.cell
def _():
    from trunx.gp3.bayesiancalibrations.bayesian_comparison_plots import plot_comparison
    from trunx.gp3.bayesiancalibrations.bayesian_config import FIT_PARAMS
    from trunx.gp3.plot_function import create_comparison_dataframe, plot_combined_3pg_outputs_obv
    from trunx.gp3.prepare_data import prepare_data
    from trunx.gp3.run_3pg import run_3pg
    from trunx.gp3.run_r3pg import run_comparison_r

    return (
        FIT_PARAMS,
        create_comparison_dataframe,
        plot_combined_3pg_outputs_obv,
        plot_comparison,
        prepare_data,
        run_3pg,
        run_comparison_r,
    )


@app.cell
def _(cycler, plt):
    # Greenish (ColorBrewer Greens-6) theme: any plot/artist that doesn't set
    # an explicit color pulls from this cycle instead of matplotlib's default
    # blue/orange palette. Light green -> dark green.
    GREEN_SHADES = ["#edf8e9", "#c7e9c0", "#a1d99b", "#74c476", "#31a354", "#006d2c"]
    plt.rcParams["axes.prop_cycle"] = cycler(color=GREEN_SHADES)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    return (GREEN_SHADES,)


@app.cell
def _(images_folder, os):
    paper_figs_dir = os.path.join(images_folder, "paper_figs")
    os.makedirs(paper_figs_dir, exist_ok=True)
    return (paper_figs_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paper Figures
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### R vs Python 3PG Outputs (Solling)
    """)
    return


@app.cell
def _(os, threepg_data_folder):
    solling_file_path = os.path.join(threepg_data_folder, "solling_data.xlsx")
    return (solling_file_path,)


@app.cell
def _(np, prepare_data, run_3pg, solling_file_path):
    _input_data = prepare_data(solling_file_path)
    _, r_py_outputs = run_3pg(
        _input_data.initial_state,
        _input_data.climate,
        _input_data.params,
        _input_data.site,
        _input_data.species,
    )
    r_py_start_month = np.datetime64(
        f"{int(_input_data.site.year_i[0]):04d}-{int(_input_data.site.month_i[0]):02d}", "M"
    )
    r_py_species_names = _input_data.species_names
    return r_py_outputs, r_py_species_names, r_py_start_month


@app.cell
def _(
    create_comparison_dataframe,
    r_py_outputs,
    r_py_species_names,
    r_py_start_month,
    run_comparison_r,
    solling_file_path,
):
    _r_outputs = run_comparison_r(solling_file_path)
    r_python_df = create_comparison_dataframe(
        _r_outputs, r_py_outputs, r_py_start_month, r_py_species_names
    )
    return (r_python_df,)


@app.cell
def _(
    GREEN_SHADES,
    os,
    paper_figs_dir,
    plot_combined_3pg_outputs_obv,
    plt,
    r_python_df,
):
    # show=False: rotate the x-tick labels before displaying/saving, not after
    # (plot_combined_3pg_outputs_obv would otherwise call plt.show() itself
    # first, so the inline preview would miss the rotation).
    _figs = plot_combined_3pg_outputs_obv(
        r_python_df,
        plot_metrics=["WS", "WF", "WR"],
        fig_name="r_python_solling",
        plot_id="solling",
        series_colors={"python": GREEN_SHADES[5], "r": GREEN_SHADES[1]},
        show=False,
    )
    for _ax in _figs[0].axes:
        _ax.tick_params(axis="x", rotation=45)
    _figs[0].savefig(os.path.join(paper_figs_dir, "image1.png"))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Morris Sensitivity Analysis
    """)
    return


@app.cell
def _(GREEN_SHADES, np, os, paper_figs_dir, pl, plt, results_data_folder):
    morris_dir = os.path.join(results_data_folder, "morris_analysis_results_jax")
    morris_df = pl.read_csv(os.path.join(morris_dir, "morris_all_components.csv"))

    morris_total_df = morris_df.filter(pl.col("Component") == "total").sort(by="sigma")

    _x = np.arange(len(morris_total_df["Parameter"]))
    _width = 0.35

    morris_fig, morris_ax = plt.subplots(figsize=(20, 10))
    morris_ax.bar(
        _x - _width / 2, morris_total_df["sigma"], _width, label="sigma", color=GREEN_SHADES[2]
    )
    morris_ax.bar(
        _x + _width / 2,
        morris_total_df["mu_star"],
        _width,
        label="mu_star",
        color=GREEN_SHADES[5],
    )
    morris_ax.set_xticks(_x)
    morris_ax.set_xticklabels(morris_total_df["Parameter"], rotation=90)
    morris_ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(paper_figs_dir, "image2.png"))

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bayesian Calibration Results — Solling (biomass_only)
    """)
    return


@app.cell
def _(os, results_data_folder):
    calibration_dir = os.path.join(
        results_data_folder, "calibration_sweep", "solling", "biomass_only"
    )
    bayesian_output_dir = os.path.join(calibration_dir, "demetropolisz")
    hmc_output_dir = os.path.join(calibration_dir, "nuts")
    return bayesian_output_dir, hmc_output_dir


@app.cell
def _(
    FIT_PARAMS,
    GREEN_SHADES,
    bayesian_output_dir,
    hmc_output_dir,
    os,
    paper_figs_dir,
    plot_comparison,
    plt,
    solling_file_path,
):
    prediction_fig, prediction_metrics_df = plot_comparison(
        solling_file_path,
        FIT_PARAMS,
        bayesian_output_dir=bayesian_output_dir,
        hmc_output_dir=hmc_output_dir,
        plot_variables=["WS", "WF", "WR"],
        include_bayesian=True,
        include_hmc=True,
        bayesian_label="PyMC (DEz)",
        hmc_label="HMC (NUTS)",
        series_colors={
            "default": GREEN_SHADES[3],
            "bayesian": GREEN_SHADES[5],
            "hmc": "orange",
        },
    )
    # No titles/subtitles: drop the per-subplot RMSE annotation (each subplot's
    # y-axis already labels its variable) and skip the figure suptitle.
    for _ax in prediction_fig.axes:
        _ax.set_title("")
    prediction_fig.savefig(os.path.join(paper_figs_dir, "image3.png"))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Spatial Simulation
    """)
    return


@app.cell
def _(Transformer, os, pl, results_data_folder, threepg_data_folder):
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

    grid_input_path = os.path.join(threepg_data_folder, "grid_input.parquet")
    grid_ids = pl.read_parquet(grid_input_path).select("grid_id", "x", "y")

    lon, lat = transformer.transform(grid_ids["x"].to_numpy(), grid_ids["y"].to_numpy())
    grid_ids = grid_ids.with_columns([pl.Series("lon", lon), pl.Series("lat", lat)]).drop(
        ["x", "y"]
    )

    grid_outputs = pl.read_parquet(os.path.join(results_data_folder, "grid_3pg_outputs.parquet"))
    grid_outputs = (
        grid_outputs.select("grid_id", "param_idx", "WS")
        .group_by("grid_id")
        .agg(
            mean_WS=pl.col("WS").mean(),
            lower_post=pl.col("WS").quantile(0.025),
            upper_post=pl.col("WS").quantile(0.975),
        )
    )
    grid_outputs = grid_outputs.join(grid_ids, on="grid_id", how="inner").with_columns(
        conf_interval=pl.col("upper_post") - pl.col("lower_post")
    )
    return (grid_outputs,)


@app.cell
def _(
    GREEN_SHADES,
    clean_data_folder,
    go,
    gpd,
    grid_outputs,
    make_subplots,
    os,
    paper_figs_dir,
    results_data_folder,
):
    # Plain cartesian axes with a hand-drawn border instead of a tile-based map,
    # so the "Greens" coloraxis reads consistently against a plain white ground.
    boundary = gpd.read_file(os.path.join(clean_data_folder, "switzerland_boundary.gpkg"))
    border_lon, border_lat = boundary.geometry.iloc[0].exterior.xy

    _mean = grid_outputs["mean_WS"]
    _ci = grid_outputs["conf_interval"]
    _cmin = min(_mean.min(), _ci.min())
    _cmax = max(_mean.max(), _ci.max())

    spatial_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Posterior predictive mean",
            "Width of 95% credible interval",
        ),
        horizontal_spacing=0.0001,
    )

    for _col, _values in ((1, _mean), (2, _ci)):
        spatial_fig.add_trace(
            go.Scatter(
                x=list(border_lon),
                y=list(border_lat),
                mode="lines",
                line=dict(width=1, color=GREEN_SHADES[5]),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=_col,
        )
        spatial_fig.add_trace(
            go.Scatter(
                x=grid_outputs["lon"],
                y=grid_outputs["lat"],
                mode="markers",
                marker=dict(size=2, color=_values, coloraxis="coloraxis"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=_col,
        )

    spatial_fig.update_xaxes(showticklabels=False, ticks="")
    spatial_fig.update_yaxes(showticklabels=False, ticks="")

    spatial_fig.update_layout(
        coloraxis=dict(
            colorscale="Greens",
            cmin=_cmin,
            cmax=_cmax,
            colorbar=dict(title=dict(text="Stand biomass (t/ha)", side="right")),
        ),
        plot_bgcolor="white",
        width=1000,
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    _output_dir = os.path.join(results_data_folder, "fig_paper")
    os.makedirs(_output_dir, exist_ok=True)
    spatial_fig.write_image(os.path.join(_output_dir, "stand_biomass_grid.png"), scale=2)
    spatial_fig.write_image(os.path.join(paper_figs_dir, "image4.png"))

    spatial_fig.show()
    return


if __name__ == "__main__":
    app.run()
