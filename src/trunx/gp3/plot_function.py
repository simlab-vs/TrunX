"""Plot functions to visualize outputs and its comparison with r3PG."""

import os
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


def plot_outputs(outputs, start_month, fig_name: str | None = None):
    """Visualize key 3-PG state variables over time."""
    if fig_name is None:
        fig_name = "3PG.png"

    num_months = outputs["WS"].shape[0]

    all_months = [start_month + np.timedelta64(i, "M") for i in range(num_months)]
    years = [str(m)[:4] for m in all_months]

    months = jnp.arange(num_months)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True)

    # Top row
    axes[0, 0].plot(months, outputs["DBH"])
    axes[0, 0].set_ylabel(r"DBH (cm)")

    axes[0, 1].plot(months, outputs["LAI"])
    axes[0, 1].set_ylabel("LAI")

    axes[0, 2].plot(months, outputs["GPP"])
    axes[0, 2].set_ylabel(r"GPP ($\mathrm{mol\ C\ m^{-2}}$)")

    # Bottom row
    axes[1, 0].plot(months, outputs["WS"])
    axes[1, 0].set_ylabel(r"Stem biomass ($\mathrm{kg\ ha^{-1}}$)")

    axes[1, 1].plot(months, outputs["WF"])
    axes[1, 1].set_ylabel(r"Foliage biomass ($\mathrm{kg\ ha^{-1}}$)")

    axes[1, 2].plot(months, outputs["WR"])
    axes[1, 2].set_ylabel(r"Root biomass ($\mathrm{kg\ ha^{-1}}$)")

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

    plt.tight_layout()
    plt.savefig(os.path.join("./images/", fig_name))
    plt.show()

    return fig


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


def create_comparison_dataframe(r_df, outputs, start_month):
    """
    Create a polars DataFrame combining R 3-PG outputs and python implementation results.

    Parameters
    ----------
    r_df: pl.DataFrame
        polars DataFrame from R with columns: date, variable, value, species
    outputs: Dict
        dict of original outputs like {"WS": array, "DBH": array, ...}
    start_month: datetime
        numpy datetime64 for start (e.g., np.datetime64('2000-01'))

    Returns
    -------
    pl.DataFrame
        Combined DataFrame of R and Python outputs.
    """
    # Generate dates
    num_months = outputs["WS"].shape[0]
    dates = [start_month + np.timedelta64(i, "M") for i in range(num_months)]
    dates = pd.to_datetime(dates).to_list()
    dates = [d.date() for d in dates]

    r_outputs = r_df.filter(
        pl.col("variable").is_in(
            [
                "dbh",
                "lai",
                "gpp",
                "biom_stem",
                "biom_foliage",
                "biom_root",
                "f_vpd",
                "f_age",
                "f_tmp",
                "f_frost",
                "f_sw",
                "f_nutr",
                "f_phys",
                "pFS",
                "apar",
                "asw",
                "sla",
                "alpha_c",
                "f_calpha",
                "npp_fract_foliage",
                "npp_fract_stem",
                "npp_fract_root",
            ]
        )
    )

    r_outputs = r_outputs.pivot(index="date", columns="variable", values="value").sort("date")
    rename_dict = {
        "dbh": "r_DBH",
        "lai": "r_LAI",
        "gpp": "r_GPP",
        "biom_stem": "r_WS",
        "biom_foliage": "r_WF",
        "biom_root": "r_WR",
        "f_vpd": "r_fD",
        "f_age": "r_fAge",
        "f_tmp": "r_fT",
        "f_frost": "r_fF",
        "f_sw": "r_fSW",
        "f_nutr": "r_fN",
        "f_phys": "r_phi",
        "apar": "r_APAR",
        "asw": "r_ASW",
        "pFS": "r_pFS",
        "sla": "r_SLA",
        "alpha_c": "r_alpha_c",
        "f_calpha": "r_fcalpha",
        "npp_fract_foliage": "r_eta_F",
        "npp_fract_stem": "r_eta_S",
        "npp_fract_root": "r_eta_R",
    }
    r_outputs = r_outputs.rename(rename_dict)
    r_outputs = r_outputs.with_columns(pl.Series("Dates", dates, dtype=pl.Date))
    r_outputs = r_outputs.drop("date")

    p_outputs = pl.from_pandas(pd.DataFrame(outputs)).select(
        [
            "DBH",
            "LAI",
            "GPP",
            "WS",
            "WF",
            "WR",
            "fD",
            "fSW",
            "fAge",
            "fN",
            "fF",
            "fT",
            "phi",
            "APAR",
            "ASW",
            "pFS",
            "SLA",
            "alpha_c",
            "fcalpha",
            "eta_R",
            "eta_S",
            "eta_F",
        ]
    )

    p_outputs = p_outputs.with_columns(pl.Series("Dates", dates, dtype=pl.Date))

    df = p_outputs.join(r_outputs, on="Dates", how="inner")

    df.write_csv("./data/r_python.comparison.csv")

    return df
