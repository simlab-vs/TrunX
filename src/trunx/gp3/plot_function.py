"""Plot function for comparison of 3PG models implement in r3Pg and Python."""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_combined_3pg_outputs(r_df, outputs, start_month, fig_name: str | None = None):
    """
    Visualize both R 3-PG outputs and python implementation in the same plot.

    Parameters
    ----------
    r_df: pl.DataFrame
        polars DataFrame from R with columns: date, variable, value, species
    outputs: Dict
        dict of original outputs like {"WS": array, "DBH": array, ...}
    start_month: datetime
        numpy datetime64 for start (e.g., np.datetime64('2000-01-01'))
    fig_name: str
        name to save figure
    """
    if fig_name is None:
        fig_name = "3PG_combined_comparison.png"

    # Define variables to plot (matching your R code)
    i_var = ["dbh", "lai", "gpp", "biom_stem", "biom_foliage", "biom_root"]
    i_lab = [
        "DBH (cm)",
        "LAI",
        r"GPP (mol C m$^{-2}$)",
        r"Stem biomass (kg ha$^{-1}$)",
        r"Foliage biomass (kg ha$^{-1}$)",
        r"Root biomass (kg ha$^{-1}$)",
    ]

    # Map R variable names to original output keys
    var_mapping = {
        "dbh": "DBH",
        "lai": "LAI",
        "gpp": "GPP",
        "biom_stem": "WS",
        "biom_foliage": "WF",
        "biom_root": "WR",
    }

    # Filter R data for variables of interest
    plot_data = r_df.filter(pl.col("variable").is_in(i_var))

    # Get unique species from R data
    species_list = plot_data["species"].unique().to_list()

    # Get dates from R data
    dates = plot_data["date"].unique().sort().to_numpy()
    num_months = len(dates)
    months = np.arange(num_months)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    cmap = plt.cm.get_cmap("Set2")
    r_colors = cmap(np.linspace(0, 1, len(species_list)))

    for idx, (var, label) in enumerate(zip(i_var, i_lab, strict=True)):
        ax = axes.flat[idx]
        var_data = plot_data.filter(pl.col("variable") == var)

        for species, color in zip(species_list, r_colors, strict=True):
            species_data = var_data.filter(pl.col("species") == species)
            if species_data.height > 0:
                values = []
                for date in dates:
                    val = species_data.filter(pl.col("date") == date)["value"]
                    values.append(val[0] if len(val) > 0 else np.nan)

                ax.plot(
                    months,
                    values,
                    "--",
                    label=f"R - {species}",
                    color=color,
                    linewidth=1.5,
                    alpha=0.7,
                )

        orig_key = var_mapping[var]
        if orig_key in outputs:
            orig_values = outputs[orig_key]
            if len(orig_values) >= num_months:
                ax.plot(
                    months,
                    orig_values[:num_months],
                    "-",
                    label="Python",
                    color="black",
                    linewidth=2,
                    alpha=0.8,
                )

        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3)

        ax.legend(loc="upper left", fontsize="small", ncol=2)

    num_months = outputs["WS"].shape[0]

    all_months = [start_month + np.timedelta64(i, "M") for i in range(num_months)]
    years = [str(m)[:4] for m in all_months]

    months = np.arange(num_months)

    tick_indices = [
        i for i, m in enumerate(all_months) if m.astype("datetime64[M]").astype(int) % 12 == 0
    ]
    tick_labels = [years[i] for i in tick_indices]
    last_year = int(years[-1])
    if int(tick_labels[-1]) < last_year + 1:
        tick_indices.append(num_months - 1)
        tick_labels.append(str(last_year + 1))

    for ax in axes.flat:
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Year")

    plt.suptitle("3-PG Model Outputs: R3PG vs Python3PG", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join("./images/", fig_name))
    plt.show()

    return fig
