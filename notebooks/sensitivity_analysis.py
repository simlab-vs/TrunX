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

    from trunx.config import results_data_folder

    return np, os, pl, plt, results_data_folder


@app.cell
def _(np, os, pl, plt, results_data_folder):
    morris_dir = os.path.join(results_data_folder, "morris_analysis_results_jax")

    sensitivity_df = pl.read_csv(os.path.join(morris_dir, "morris_all_components.csv"))

    df = sensitivity_df.filter(pl.col("Component") == "total").sort(by="sigma", descending=True)[
        0:25
    ]

    x = np.arange(len(df["Parameter"]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(x - width / 2, df["sigma"], width, label="sigma")
    ax.bar(x + width / 2, df["mu_star"], width, label="mu_star")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Parameter"], rotation=45)
    ax.legend()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
