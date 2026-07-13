"""Plot functions to visualize outputs and its comparison with r3PG."""

import datetime as dt
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from trunx.gp3.weather_processing import create_weather_input


def plot_outputs(outputs, start_month, fig_name: str | None = None, show: bool = True):
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
    if show:
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


def create_comparison_dataframe(r_df, outputs, start_month, species_list):
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
    dates = [d + pd.offsets.MonthEnd(0) for d in dates]
    # dates = [d.date() for d in dates]

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
                "stems_n",
                "mort_stress",
                "mort_thinn",
                "basal_area",
                "height",
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
        "stems_n": "r_stems_n",
        "mort_stress": "r_mort_stress",
        "mort_thinn": "r_mort_thinn",
        "basal_area": "r_BA",
        "height": "r_Height",
    }
    r_outputs = r_outputs.rename(rename_dict)
    r_outputs = r_outputs.with_columns(
        pl.col("date")
        .map_elements(
            lambda x: dt.datetime(1970, 1, 1) + dt.timedelta(days=x), return_dtype=pl.Datetime
        )
        .alias("Dates")
    ).with_columns(
        pl.col("Dates").dt.year().alias("year"), pl.col("Dates").dt.month().alias("month")
    )

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
            "stems_n",
            "mort_stress",
            "mort_thinn",
            "BA",
            "Height",
        ]
    )

    df = p_outputs.join(r_outputs, on=["Dates", "species"], how="inner")
    df = df.with_columns(pl.col("Dates").dt.strftime("%Y-%m-%d").alias("Dates"))
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
    df,
    metrics_to_plot=None,
    observed_data=None,
    fig_name=None,
    plot_id="",
    show: bool = True,
):
    """Visualize R and Python 3PG implementations with observed data."""
    # Prepare data
    df["Dates"] = pd.to_datetime(df["Dates"])
    if observed_data is not None and "period_end" in observed_data.columns:
        observed_data["period_end"] = pd.to_datetime(observed_data["period_end"])

    if observed_data is not None and "date" in observed_data.columns:
        observed_data["date"] = pd.to_datetime(observed_data["date"])

    species_list = df["species"].unique()

    metrics_to_plot = {
        "DBH": {"label": "DBH (cm)", "python_col": "DBH", "r_col": "r_DBH"},
        "LAI": {"label": "LAI", "python_col": "LAI", "r_col": "r_LAI"},
        "GPP": {"label": "GPP (mol C m⁻²)", "python_col": "GPP", "r_col": "r_GPP"},
        "WS": {"label": "Stem Biomass (t DM ha⁻¹)", "python_col": "WS", "r_col": "r_WS"},
        "WF": {"label": "Foliage Biomass (t DM ha⁻¹)", "python_col": "WF", "r_col": "r_WF"},
        "WR": {"label": "Root Biomass (t DM ha⁻¹)", "python_col": "WR", "r_col": "r_WR"},
    }

    # Setup subplots
    n_metrics = len(metrics_to_plot)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    figures = []
    for species in species_list:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        axes = axes.flatten() if n_metrics > 1 else [axes]

        species_data = df[df["species"] == species].sort_values("Dates")

        for idx, (_metric, config) in enumerate(metrics_to_plot.items()):
            if idx >= len(axes):
                break

            # Plot Python
            if config["python_col"] in df.columns:
                axes[idx].plot(
                    species_data["Dates"], species_data[config["python_col"]], "-", label="Python"
                )

            # Plot R
            if config["r_col"] in df.columns:
                axes[idx].plot(
                    species_data["Dates"],
                    species_data[config["r_col"]],
                    "--",
                    label="R",
                    alpha=0.7,
                )

            # Plot observed data
            if (
                observed_data is not None
                and config["python_col"] in observed_data.columns
                and "specie" in observed_data.columns
            ):
                obs = observed_data[observed_data["specie"] == species].dropna(
                    subset=[config["python_col"]]
                )
                axes[idx].scatter(
                    obs["Date"],
                    obs[config["python_col"]],
                    s=20,
                    marker="s",
                    color="red",
                    label="Observed",
                )

                axes[idx].plot(obs["Date"], obs[config["python_col"]], alpha=0.6)

            if (
                observed_data is not None
                and config["python_col"] in observed_data.columns
                and "date" in observed_data.columns
            ):
                axes[idx].scatter(
                    observed_data["date"],
                    observed_data[config["python_col"]],
                    s=20,
                    marker="s",
                    color="red",
                    label="Observed",
                )

            axes[idx].set_ylabel(config["label"])
            axes[idx].set_title(config["label"].split("(")[0].strip())
            axes[idx].grid(True, alpha=0.3)
            if idx == 0:
                axes[idx].legend()

        plt.suptitle(f"3-PG Model Outputs: {species}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            os.path.join("./images/", f"{fig_name}_{plot_id}_{species}.png") if fig_name else None
        )
        figures.append(fig)

    if show:
        plt.show()

    return figures


def plot_weather_data(clean_wdf, plot_id):
    """Visualize weather data with missing value periods highlighted."""
    _miss_months, weather_df = create_weather_input(clean_wdf, plot_id=plot_id)

    weather_pl = weather_df.with_columns(pl.date(pl.col("year"), pl.col("month"), 1).alias("date"))
    min_date = weather_pl.select(pl.col("date").min()).item()
    max_date = weather_pl.select(pl.col("date").max()).item()
    all_months = pl.date_range(start=min_date, end=max_date, interval="1mo", eager=True)

    weather_labels = {
        "tmp_min": "Minimum temperature (°C)",
        "tmp_max": "Maximum temperature(°C)",
        "tmp_ave": "Average temperature (°C)",
        "prcp": "Precipitation (mm)",
        "srad": "Solar Radiation (MJ/m²)",
        "frost_days": "Days/Month",
    }

    weather_all_months = pl.DataFrame({"date": pl.Series(all_months)}).join(
        weather_pl, on="date", how="left"
    )

    weather_pd = weather_all_months.to_pandas()
    weather_pd["date"] = pd.to_datetime(weather_pd["date"])
    weather_metrics = [col for col in weather_pd.columns if col not in ["date", "year", "month"]]

    _fig, axes = plt.subplots(len(weather_metrics), 1, figsize=(14, len(weather_metrics) * 6))

    for idx, metric in enumerate(weather_metrics):
        ax = axes[idx]
        # Identify missing value periods
        weather_pd["is_missing"] = weather_pd[metric].isna()

        # Find contiguous missing periods
        missing_periods = []
        in_missing = False
        start_idx = None

        for i, missing in enumerate(weather_pd["is_missing"]):
            if missing and not in_missing:
                start_idx = i
                in_missing = True
            elif not missing and in_missing:
                missing_periods.append((start_idx, i - 1))
                in_missing = False
        if in_missing:
            missing_periods.append((start_idx, len(weather_pd) - 1))

        # Create plot

        # Plot the line
        ax.plot(
            weather_pd["date"], weather_pd[metric], "b-", linewidth=2, label=weather_labels[metric]
        )

        # Highlight missing periods in red
        for start, end in missing_periods:
            ax.axvspan(
                weather_pd["date"].iloc[start],
                weather_pd["date"].iloc[end],
                alpha=0.3,
                color="red",
                label="Missing Data" if start == missing_periods[0][0] else "",
            )

        # Also mark missing points as red circles
        missing_data = weather_pd[weather_pd["is_missing"]]
        ax.scatter(
            missing_data["date"],
            [ax.get_ylim()[0]] * len(missing_data),
            color="red",
            s=30,
            marker="v",
            label="Missing Points",
            zorder=5,
        )

        ax.set_xlabel("Date")
        ax.set_ylabel(weather_labels[metric])
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
