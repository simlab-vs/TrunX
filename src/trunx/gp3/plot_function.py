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
    axes[1, 0].set_ylabel(r"Stem biomass ($\mathrm{t DM\ ha^{-1}}$)")

    axes[1, 1].plot(months, outputs["WF"])
    axes[1, 1].set_ylabel(r"Foliage biomass ($\mathrm{t DM\ ha^{-1}}$)")

    axes[1, 2].plot(months, outputs["WR"])
    axes[1, 2].set_ylabel(r"Root biomass ($\mathrm{t DM\ ha^{-1}}$)")

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


def plot_combined_3pg_outputs(
    r_df, outputs, start_month, species_list, fig_name: str | None = None
):
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
        fig_name = ""

    # Define variables to plot
    i_var = ["dbh", "lai", "gpp", "biom_stem", "biom_foliage", "biom_root"]
    i_lab = [
        "DBH (cm)",
        "LAI",
        r"GPP (mol C m$^{-2}$)",
        r"Stem biomass (t DM ha$^{-1}$)",
        r"Foliage biomass (t DM ha$^{-1}$)",
        r"Root biomass (t DM ha$^{-1}$)",
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
        for idx, (species, color) in enumerate(zip(species_list, r_colors, strict=True)):
            orig_key = var_mapping[var]
            if orig_key in outputs:
                orig_values = outputs[orig_key][:, idx]
                if len(orig_values) >= num_months:
                    ax.plot(
                        months,
                        orig_values[:num_months],
                        "-",
                        label=f"P - {species}",
                        color=color,
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
    species_list = r_df.select("species").unique().to_series().to_list()
    n_species = len(species_list)
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
                "gammaF",
                "f_transp_scale",
            ]
        )
    )

    r_outputs = r_outputs.pivot(
        index=["date", "species"], columns="variable", values="value"
    ).sort(["date", "species"])

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
        "gammaF": "r_gammaF",
        "f_transp_scale": "r_f_transp_scale",
    }
    r_outputs = r_outputs.rename(rename_dict)

    dates_expanded = []
    for date in dates:
        for _ in range(n_species):
            dates_expanded.append(date)

    # r_outputs = r_outputs.with_columns(pl.Series("Dates", dates, dtype=pl.Date))
    r_outputs = r_outputs.with_columns(pl.Series("Dates", dates_expanded, dtype=pl.Date))
    r_outputs = r_outputs.drop("date")

    p_records = []
    for var in outputs:
        for t in range(num_months):
            for s, specie in enumerate(species_list):
                p_records.append(
                    {
                        "Dates": dates[t],
                        "species": f"{specie}",
                        "variable": var,
                        "p_value": outputs[var][t, s]
                        if outputs[var].ndim > 1
                        else outputs[var][t],
                    }
                )

    p_outputs = pl.DataFrame(p_records)
    p_outputs = p_outputs.pivot(
        index=["Dates", "species"],
        on="variable",
        values="p_value",
    )

    p_outputs = p_outputs.select(
        [
            "Dates",
            "species",
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
            "gammaF",
            "f_transp_scale",
        ]
    )

    df = p_outputs.join(r_outputs, on=["Dates", "species"], how="inner")

    df.write_csv("./data/r_python.comparison.csv")

    return df.to_pandas()


def plot_combined_3pg_outputs_per_species(
    r_df, outputs, start_month, species_list, fig_name: str | None = None
):
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
        r"Stem biomass (t DM ha$^{-1}$)",
        r"Foliage biomass (t DM ha$^{-1}$)",
        r"Root biomass (t DM ha$^{-1}$)",
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

    # Get dates from R data
    dates = plot_data["date"].unique().sort().to_numpy()
    num_months = len(dates)
    months = np.arange(num_months)

    cmap = plt.cm.get_cmap("Set2")
    r_colors = cmap(np.linspace(0, 1, len(species_list)))
    figures = []
    for sp_idx, (species, color) in enumerate(zip(species_list, r_colors, strict=True)):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        species_data = plot_data.filter(pl.col("species") == species)
        for idx, (var, label) in enumerate(zip(i_var, i_lab, strict=True)):
            ax = axes.flat[idx]
            var_data = species_data.filter(pl.col("variable") == var)

            if species_data.height > 0:
                values = []
                for date in dates:
                    val = var_data.filter(pl.col("date") == date)["value"]
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
                orig_values = outputs[orig_key][:, sp_idx]
                if len(orig_values) >= num_months:
                    ax.plot(
                        months,
                        orig_values[:num_months],
                        "-",
                        label=f"P - {species}",
                        color=color,
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
        figures.append(fig)

    # plt.show()
    return figures if figures else []


def plot_combined_3pg_outputs_obv(
    df, metrics_to_plot=None, observed_data=None, show_r: bool = True, show_python: bool = True
):
    """
    Visualize R, and python 3PG implementation in the same plot, with observed data.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with columns: 'Dates', 'species', and both Python and R metrics
        Python columns: 'DBH', 'LAI', 'GPP', 'WS', 'WF', 'WR', etc.
        R columns: 'r_DBH', 'r_LAI', 'r_GPP', 'r_WS', 'r_WF', 'r_WR', etc.
    metrics_to_plot: dict, optional
        Dictionary specifying which metrics to plot
    observed_data:
        DataFrame with observed data for certain metrics (e.g., DBH)
    show_r: bool
        whether to show R outputs
    show_python: bool
        whether to show Python outputs
    """
    # Convert dates and get basic info
    df["Dates"] = pd.to_datetime(df["Dates"])
    species_list = df["species"].unique()
    start_year = df["Dates"].min().year

    # Default metrics if not specified
    if metrics_to_plot is None:
        metrics_to_plot = {
            "DBH": {
                "label": "DBH (cm)",
                "python_col": "DBH",
                "r_col": "r_DBH",
                "has_observed": True,
            },
            "LAI": {"label": "LAI", "python_col": "LAI", "r_col": "r_LAI", "has_observed": False},
            "GPP": {
                "label": "GPP (mol C m⁻²)",
                "python_col": "GPP",
                "r_col": "r_GPP",
                "has_observed": False,
            },
            "WS": {
                "label": "Stem Biomass (t DM ha⁻¹)",
                "python_col": "WS",
                "r_col": "r_WS",
                "has_observed": False,
            },
            "WF": {
                "label": "Foliage Biomass (t DM ha⁻¹)",
                "python_col": "WF",
                "r_col": "r_WF",
                "has_observed": False,
            },
            "WR": {
                "label": "Root Biomass (t DM ha⁻¹)",
                "python_col": "WR",
                "r_col": "r_WR",
                "has_observed": False,
            },
        }

    n_metrics = len(metrics_to_plot)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    dates_sorted = sorted(df["Dates"].unique())
    num_months = len(dates_sorted)
    months = np.arange(num_months)

    all_months = [
        pd.Timestamp(f"{start_year}-01-01") + pd.DateOffset(months=i) for i in range(num_months)
    ]
    tick_indices = [i for i, m in enumerate(all_months) if m.month == 1]
    tick_labels = [str(all_months[i].year) for i in tick_indices]

    cmap = plt.cm.get_cmap("Set2")
    species_colors = {
        species: cmap(i / max(1, len(species_list) - 1)) for i, species in enumerate(species_list)
    }

    figures = []
    for species in species_list:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(min(15, 5 * n_cols), min(10, 4 * n_rows))
        )
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        species_data = df[df["species"] == species].sort_values("Dates")
        for idx, (_metric_name, config) in enumerate(metrics_to_plot.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]

            if show_python and config["python_col"] in df.columns:
                values = [
                    species_data[species_data["Dates"] == date][config["python_col"]].iloc[0]
                    if len(species_data[species_data["Dates"] == date]) > 0
                    else np.nan
                    for date in dates_sorted
                ]

                if not all(np.isnan(v) for v in values):
                    ax.plot(
                        months,
                        values,
                        "-",
                        label=f"Python - {species}",
                        color=species_colors[species],
                        linewidth=2,
                        alpha=0.8,
                        marker="o",
                        markersize=3,
                        markevery=max(1, len(months) // 20),
                    )

            if show_r and config["r_col"] in df.columns:
                values = [
                    species_data[species_data["Dates"] == date][config["r_col"]].iloc[0]
                    if len(species_data[species_data["Dates"] == date]) > 0
                    else np.nan
                    for date in dates_sorted
                ]

                if not all(np.isnan(v) for v in values):
                    ax.plot(
                        months,
                        values,
                        "--",
                        label=f"R - {species}",
                        color=species_colors[species],
                        linewidth=1.5,
                        alpha=0.7,
                        marker="s",
                        markersize=3,
                        markevery=max(1, len(months) // 20),
                    )

            if config.get("has_observed", False) and observed_data is not None:
                avg_diameter = observed_data[observed_data["specie"] == species]
                if avg_diameter.empty:
                    continue

                model_years = np.array([start_year + i // 12 for i in range(num_months)])

                obs_indices = []
                obs_values = []
                for year, dbh in zip(avg_diameter["period_end"], avg_diameter["DBH"], strict=True):
                    year_val = year.year if hasattr(year, "year") else int(year)
                    diff = np.abs(model_years - year_val)
                    closest_idx = np.argmin(diff)
                    if diff[closest_idx] < 0.5:
                        obs_indices.append(closest_idx)
                        obs_values.append(dbh)

                if obs_indices:
                    ax.scatter(
                        obs_indices,
                        obs_values,
                        color=species_colors[species],
                        s=50,
                        marker="s",
                        zorder=5,
                        label="Observed",
                        alpha=0.9,
                    )
                    ax.plot(
                        obs_indices,
                        obs_values,
                        "-",
                        color=species_colors[species],
                        linewidth=2,
                        alpha=0.6,
                    )

            ax.set_ylabel(config["label"], fontsize=11)
            ax.set_title(config["label"].split("(")[0].strip(), fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

            if idx == 0:
                ax.legend(fontsize="small")

        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        for ax in axes[:n_metrics]:
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
            ax.set_xlabel("Year", fontsize=10)
            # ax.set_xlim(0, num_months - 1)

        plt.suptitle("3-PG Model Outputs: R3PG vs Python3PG", fontsize=14, fontweight="bold")
        plt.tight_layout()
        figures.append(fig)

    plt.show()

    return figures
