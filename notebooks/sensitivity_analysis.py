"""Sensitivity analysis for 3PG model parameters using Morris method."""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():

    return


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    return np, os, pl, plt


@app.cell
def _(np, os, pl, plt):
    morris_dir = os.path.join("./data", "morris_analysis_results")

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
