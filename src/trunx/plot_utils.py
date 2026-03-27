"""Helper functions for plots and EDA visualizations."""

import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
import seaborn as sns

pio.renderers.default = "notebook"


def plot_social_class_per_plot(
    df: pl.DataFrame,
    n_cols: int = 4,
    x="year",
    y="norm count",
    hue="social_class_mode",
    normalized=True,
    plot_type: str = "bar",
):
    """
    Plot bar charts of social_class_mode counts per year for each plot_id.

    Parameters
    ----------
        df (pl.DataFrame): Polars DataFrame containing the columns:
            - plot_id
            - period_end (datetime)
            - social_class_mode
        n_cols (int): number of columns in subplot grid
    """
    y = "norm count" if normalized else "count"
    year_counts = (
        df.select(["plot_id", "period_end", "social_class_mode"])
        .drop_nulls()
        .with_columns(pl.col("period_end").dt.year().alias("year"))
        .group_by(["plot_id", "year", "social_class_mode"])
        .agg(pl.count().alias("count"))
        .sort("year")
    )

    plots_with_years = (
        year_counts.group_by("plot_id")
        .agg(pl.col("year").n_unique().alias("num_years"))
        .filter(pl.col("num_years") == pl.col("num_years").max())
        .select("plot_id")
    )

    year_counts_filtered = year_counts.join(plots_with_years, on="plot_id", how="inner")

    if normalized:
        year_counts_filtered = year_counts_filtered.join(
            year_counts_filtered.group_by(["plot_id", "year"]).agg(
                pl.sum("count").alias("total_count")
            ),
            on=["plot_id", "year"],
        ).with_columns((pl.col("count") / pl.col("total_count")).alias("norm count"))

    plot_ids = year_counts_filtered.select("plot_id").unique().to_series().to_list()
    num_classes = year_counts_filtered.select("social_class_mode").n_unique()
    unique_classes = (
        year_counts_filtered.select("social_class_mode").unique().to_series().sort().to_list()
    )
    num_years = year_counts_filtered.select("year").n_unique()
    unique_years = year_counts_filtered.select("year").unique().to_series().sort().to_list()

    if hue == "social_class_mode":
        palette = sns.cubehelix_palette(n_colors=num_classes, as_cmap=False)
        color_dict = dict(zip(unique_classes, palette, strict=True))
    elif hue == "year":
        palette = sns.cubehelix_palette(n_colors=num_years, as_cmap=False)
        color_dict = dict(zip(unique_years, palette, strict=True))
    else:
        raise ValueError("hue must be 'social_class_mode' or 'year'")

    n_rows = math.ceil(len(plot_ids) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
    axes = axes.flatten()

    for i, plot_id in enumerate(plot_ids):
        df_plot = (
            year_counts_filtered.filter(pl.col("plot_id") == plot_id).sort("year").to_pandas()
        )

        if plot_type == "bar":
            sns.barplot(
                data=df_plot,
                x=x,
                y=y,
                hue=hue,
                palette=color_dict,
                errorbar=None,
                ax=axes[i],
            )
            axes[i].set_ylabel("Proportion" if normalized else "Count")

        elif plot_type == "kde":
            unique_hues = df_plot[hue].unique()
            for cls in unique_hues:
                subset = df_plot[df_plot[hue] == cls]
                if subset.empty:
                    continue
                sns.kdeplot(
                    data=subset,
                    x=x,
                    weights=subset[y] if normalized else None,
                    fill=True,
                    alpha=0.5,
                    label=cls,
                    color=color_dict[cls],
                    ax=axes[i],
                )
            axes[i].set_ylabel("Density" if normalized else "Count")

        else:
            raise ValueError("plot_type must be 'bar' or 'kde'")

        axes[i].set_title(f"Plot {plot_id}")
        if axes[i].get_legend() is not None:
            axes[i].get_legend().remove()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", title="Social Class Mode")

    plt.tight_layout()
    plt.show()


def metric_change_per_plot_tree(
    df: pl.DataFrame,
    Species: str | None = None,
    metric: str = "social_class_mode",
    req_years: int | None = None,
    plot_id: str = "All",
):
    """
    Plot the variability of a specified metric per tree within plots over years.

    Parameters
    ----------
    df: pl.DataFrame
        DataFrame with columns 'tree_id', 'plot_id', metric, 'period_end'
    specie: str
        The species we need to examine
    metric: str
        Name of the column to analyze for changes (default: 'social_class_mode')
    max_years: int or None
        if set, filters plots to those with exactly max_years of data
    plot_id: str
        ID of the specific plot to plot ("All" for all plots)
    """
    if Species is not None:
        df = df.filter(pl.col("Species") == Species)

    df = df.with_columns(pl.col("period_end").dt.year().alias("year"))

    # Filter plots by number of years
    plots_years = df.group_by("plot_id").agg(pl.col("year").n_unique().alias("n_years"))
    max_n = req_years if req_years is not None else plots_years["n_years"].max()
    plots_filtered = plots_years.filter(pl.col("n_years") == max_n).select("plot_id")
    df_filtered = df.join(plots_filtered, on="plot_id", how="inner")

    plot_ids = plots_filtered.to_series().to_list()
    if plot_id != "All":
        if plot_id not in plot_ids:
            print(f"Plot ID '{plot_id}' not found.")
            return
        plot_ids = [plot_id]

    # Plotting
    if plot_id == "All":
        n_cols, n_rows = 2, math.ceil(len(plot_ids) / 2)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(35, 5 * n_rows))
        axes = axes.flatten()
        for i, pid in enumerate(plot_ids):
            d = df_filtered.filter(pl.col("plot_id") == pid)
            trees = (
                d.group_by("tree_id")
                .agg(pl.col(metric).n_unique().alias("n_unique"))
                .filter(pl.col("n_unique") > 1)["tree_id"]
                .to_list()
            )
            d_plot = d.filter(pl.col("tree_id").is_in(trees))
            ax = axes[i]
            if d_plot.height > 0:
                sns.barplot(
                    data=d_plot.to_pandas(),
                    x="tree_id",
                    y=metric,
                    hue="year",
                    palette="deep",
                    ax=ax,
                )
            else:
                ax.text(0.5, 0.5, f"No variability in {metric}", ha="center", va="center")
            ax.set_title(f"Plot {pid}")
            ax.tick_params(axis="x", rotation=90)
            if ax.legend_:
                ax.legend_.remove()
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        plt.tight_layout()
        plt.show()
    else:
        pid = plot_ids[0]
        d = df_filtered.filter(pl.col("plot_id") == pid)
        trees = (
            d.group_by("tree_id")
            .agg(pl.col(metric).n_unique().alias("n_unique"))
            .filter(pl.col("n_unique") > 1)["tree_id"]
            .to_list()
        )
        d_plot = d.filter(pl.col("tree_id").is_in(trees))
        if d_plot.height > 0:
            fig, ax = plt.subplots(1, 1, figsize=(35, 5))
            sns.barplot(
                data=d_plot.to_pandas(), x="tree_id", y=metric, hue="year", palette="deep", ax=ax
            )
            ax.set_title(f"Plot {pid}")
            ax.tick_params(axis="x", rotation=90)
            plt.show()
        else:
            print(f"No trees have variability in {metric} for plot {pid}.")


def plot_yearwise_social_class(tdf, ax, title=None, height=4, rotate_xticks=45):
    """Plot year wise social class from ICP data."""
    year_counts = (
        tdf.select("period_end", "social_class_mode")
        .drop_nulls()
        .with_columns(pl.col("period_end").dt.year().alias("year"))
        .group_by(["year", "social_class_mode"])
        .len(name="count")
        .sort("year")
    )

    sns.barplot(
        data=year_counts, x="year", y="count", hue="social_class_mode", errorbar=None, ax=ax
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.legend(title="social_class_mode")

    if rotate_xticks:
        ax.tick_params(axis="x", rotation=rotate_xticks)
    if title:
        ax.set_title(title)

    return ax


def plot_geographic_location_species(species_df, selected_species=None):
    """Plot geographic location of species."""
    if selected_species is not None:
        species_df = species_df.filter(pl.col("Species").is_in(selected_species))

    spruce_df = species_df.filter(pl.col("Species") == "Spruce")
    pine_df = species_df.filter(pl.col("Species") == "Pine")
    beech_df = species_df.filter(pl.col("Species") == "Beech")
    oak_df = species_df.filter(pl.col("Species") == "Oak")

    # Overlaps
    overlaps = (
        species_df.group_by(["plot_id", "Lat", "Lon"])
        .agg(
            pl.col("Species").n_unique().alias("n_species"),
            pl.col("Species").unique().alias("species_list"),
        )
        .filter(pl.col("n_species") > 1)
    )

    # Plot
    HOVER_TMPL = (
        "<b>Plot id:</b> %{text[1]}<br>"
        "<b>Latitude:</b> %{lat}<br>"
        "<b>Longitude:</b> %{lon}<br>"
        "<b>Species:</b> %{text[0]}"
        "<extra></extra>"
    )

    fig = go.Figure()

    def add_scattermap(fig, df, name, color, text):
        fig.add_trace(
            go.Scattermap(
                lat=df["Lat"].to_list(),
                lon=df["Lon"].to_list(),
                mode="markers",
                marker=dict(size=7, color=color),
                text=text,
                hovertemplate=HOVER_TMPL,
                name=name,
            )
        )

    species_layers = [
        (
            "Spruce",
            spruce_df,
            "green",
            np.column_stack(
                [["Spruce"] * len(spruce_df), [str(s) for s in spruce_df["plot_id"].to_list()]]
            ),
        ),
        (
            "Pine",
            pine_df,
            "red",
            np.column_stack(
                [["Pine"] * len(pine_df), [str(s) for s in pine_df["plot_id"].to_list()]]
            ),
        ),
        (
            "Beech",
            beech_df,
            "orange",
            np.column_stack(
                [["Beech"] * len(beech_df), [str(s) for s in beech_df["plot_id"].to_list()]]
            ),
        ),
        (
            "Oak",
            oak_df,
            "purple",
            np.column_stack(
                [["Oak"] * len(oak_df), [str(s) for s in oak_df["plot_id"].to_list()]]
            ),
        ),
    ]

    for name, df, color, text in species_layers:
        if len(df) > 0:
            add_scattermap(fig, df, name, color, text)

    # overlaps layer
    if overlaps.height > 0:
        add_scattermap(
            fig,
            overlaps,
            "Overlapping plots",
            "black",
            np.column_stack(
                [
                    [", ".join(s) for s in overlaps["species_list"].to_list()],
                    [str(s) for s in overlaps["plot_id"].to_list()],
                ]
            ),
        )

    fig.update_layout(
        map=dict(
            style="open-street-map",
            zoom=4,
            center=dict(
                lat=species_df.select(pl.mean("Lat")).item(),
                lon=species_df.select(pl.mean("Lon")).item(),
            ),
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        legend=dict(x=0, y=1),
    )

    return fig


def plot_histograms_grid(
    df,
    columns,
    hue,
    n_cols=4,
    bins=20,
    stat="count",
    multiple="dodge",
    figsize_per_col=5,
    figsize_per_row=4,
    edgecolor="black",
    share_legend=True,
):
    """
    Plot adjacent histograms for multiple columns in a grid.

    Parameters
    ----------
    df : DataFrame
        Input dataframe.
    columns : list[str]
        Columns to plot on x-axis (one per subplot).
    hue : str
        Column name for hue (e.g., species).
    n_cols : int, default=4
        Number of subplots per row.
    bins : int, default=20
        Number of histogram bins.
    stat : {"count", "density", "probability"}, default="count"
        Histogram statistic.
    multiple : {"dodge", "stack", "fill"}, default="dodge"
        Histogram style.
    figsize_per_col : int, default=5
        Width per column (in inches).
    figsize_per_row : int, default=4
        Height per row (in inches).
    edgecolor : str, default="black"
        Edge color for histogram bars.
    share_legend : bool, default=True
        Whether to show a single shared legend.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list[matplotlib.axes.Axes]
    """
    n_plots = len(columns)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(figsize_per_col * n_cols, figsize_per_row * n_rows), squeeze=False
    )

    axes = axes.flatten()

    for ax, col in zip(axes, columns, strict=False):
        plot_df = df.select([col, hue]).drop_nulls()

        if plot_df.height == 0:
            print(col, "has no data after dropping nulls. Skipping plot.")
            # ax.axis("off")
            # continue
            ax.set_title(f"{col}\n(all values NaN)")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="gray",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        sns.histplot(
            data=plot_df,
            x=col,
            hue=hue,
            bins=bins,
            stat=stat,
            multiple=multiple,
            edgecolor=edgecolor,
            ax=ax,
        )

        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel(stat.capitalize())

    # Remove unused axes
    for ax in axes[len(columns) :]:
        ax.remove()

    plt.tight_layout()

    return fig, axes[: len(columns)]


def plot_station_map(
    df,
    lat_col="Lat",
    lon_col="Lon",
    name="Stations",
    title="",
    hovertemplate=None,
    marker_color="blue",
    marker_size=8,
    zoom=8,
):
    """
    Plot station locations on a map using Plotly Scattermapbox.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing latitude and longitude columns
    lat_col : str
        Name of latitude column
    lon_col : str
        Name of longitude column
    name : str
        Name of the trace (legend label)
    hovertemplate : str or None
        Plotly hovertemplate string
    marker_color : str
        Marker color
    marker_size : int
        Marker size
    zoom : int
        Initial map zoom level

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scattermapbox(
            lat=df[lat_col],
            lon=df[lon_col],
            mode="markers",
            marker=dict(size=marker_size, color=marker_color),
            name=name,
            hovertemplate=hovertemplate,
        )
    )

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            zoom=zoom,
            center=dict(
                lat=df[lat_col].mean(),
                lon=df[lon_col].mean(),
            ),
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(x=0, y=1),
    )

    return fig


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
    observed_data: tuple, optional
        (df_growth, avg_diameter) from get_DBH_data() for DBH overlay
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
            "dbh": {
                "label": "DBH (cm)",
                "python_col": "DBH",
                "r_col": "r_DBH",
                "has_observed": True,
            },
            "lai": {"label": "LAI", "python_col": "LAI", "r_col": "r_LAI", "has_observed": False},
            "gpp": {
                "label": "GPP (mol C m⁻²)",
                "python_col": "GPP",
                "r_col": "r_GPP",
                "has_observed": False,
            },
            "stem_biomass": {
                "label": "Stem Biomass (kg ha⁻¹)",
                "python_col": "WS",
                "r_col": "r_WS",
                "has_observed": False,
            },
            "foliage_biomass": {
                "label": "Foliage Biomass (kg ha⁻¹)",
                "python_col": "WF",
                "r_col": "r_WF",
                "has_observed": False,
            },
            "root_biomass": {
                "label": "Root Biomass (kg ha⁻¹)",
                "python_col": "WR",
                "r_col": "r_WR",
                "has_observed": False,
            },
        }

    n_metrics = len(metrics_to_plot)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(min(15, 5 * n_cols), min(10, 4 * n_rows)))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

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

    for idx, (_metric_name, config) in enumerate(metrics_to_plot.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]

        if show_python and config["python_col"] in df.columns:
            for species in species_list:
                species_data = df[df["species"] == species].sort_values("Dates")
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
            for species in species_list:
                species_data = df[df["species"] == species].sort_values("Dates")
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
            df_growth, avg_diameter = observed_data

            model_years = np.array([start_year + i // 12 for i in range(num_months)])

            obs_indices = []
            obs_values = []
            for year, dbh in zip(
                avg_diameter["period_end"], avg_diameter["diameter_end"], strict=True
            ):
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
                    color="orange",
                    s=80,
                    marker="s",
                    zorder=5,
                    label="Observed",
                    alpha=0.9,
                    edgecolors="darkorange",
                )
                ax.plot(obs_indices, obs_values, "-", color="orange", linewidth=2, alpha=0.6)

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

    return fig
