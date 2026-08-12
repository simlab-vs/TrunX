"""Sensitivity analysis for 3PG model parameters using Morris method."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from trunx.config import images_folder, results_data_folder

    return images_folder, np, os, pl, plt, results_data_folder


@app.cell
def _(images_folder, np, os, pl, plt, results_data_folder):
    morris_dir = os.path.join(results_data_folder, "morris_analysis_results_jax")

    sensitivity_df = pl.read_csv(os.path.join(morris_dir, "morris_all_components.csv"))

    df = sensitivity_df.filter(pl.col("Component") == "total").sort(by="sigma")

    x = np.arange(len(df["Parameter"]))
    width = 0.3

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.bar(x + width / 2, df["mu_star"], width, label="$\mu_{*}$", color="red")
    ax.bar(x - width / 2, df["sigma"], width, label="$\sigma$", color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Parameter"], rotation=90)
    ax.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(os.path.join(images_folder, "morris_sensitivity_analysis.png"))

    plt.show()
    return (df,)


@app.cell
def _(df, pl):
    params = (
        df.filter(pl.col("Component") == "total")
        .sort(by="sigma", descending=True)["Parameter"]
        .to_list()
    )

    first_20 = [param for param in params if not param.startswith("err_")][0:20]

    print(first_20)
    return (first_20,)


@app.cell
def _(first_20):
    from trunx.gp3.bayesiancalibrations.bayesian_config import FIT_PARAMS

    set(FIT_PARAMS) - set(first_20)
    return (FIT_PARAMS,)


@app.cell
def _(FIT_PARAMS, first_20):
    set(first_20) - set(FIT_PARAMS)
    return


if __name__ == "__main__":
    app.run()
