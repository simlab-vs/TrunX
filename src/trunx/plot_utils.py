"""Helper functions for plots and EDA visualizations."""

import math

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
import seaborn as sns

pio.renderers.default = "notebook"


def plot_social_class_per_plot(df: pl.DataFrame, n_cols: int = 4):
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

    plot_ids = year_counts_filtered.select("plot_id").unique().to_series().to_list()
    num_classes = year_counts_filtered.select("social_class_mode").n_unique()
    unique_classes = (
        year_counts_filtered.select("social_class_mode").unique().to_series().sort().to_list()
    )

    palette = sns.cubehelix_palette(n_colors=num_classes, as_cmap=False)
    color_dict = dict(zip(unique_classes, palette, strict=True))

    n_rows = math.ceil(len(plot_ids) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
    axes = axes.flatten()

    for i, plot_id in enumerate(plot_ids):
        df_plot = year_counts_filtered.filter(pl.col("plot_id") == plot_id).sort("year")

        sns.barplot(
            data=df_plot,
            x="year",
            y="count",
            hue="social_class_mode",
            palette=color_dict,
            errorbar=None,
            ax=axes[i],
        )
        axes[i].set_title(f"Plot {plot_id}")
        axes[i].legend_.remove()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", title="Social Class Mode")

    plt.tight_layout()
    plt.show()


def metric_change_per_plot_tree(
    df: pl.DataFrame,
    Species: str = None,
    metric: str = "social_class_mode",
    req_years: int = None,
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


def plot_geographic_location_species(species_df):
    """Plot geographic location of species."""
    spruce_df = species_df.filter(pl.col("specie") == "Picea abies")
    pine_df = species_df.filter(pl.col("specie") == "Pinus sylvestris")
    beech_df = species_df.filter(pl.col("specie") == "Fagus sylvatica")
    oak_df = species_df.filter(pl.col("specie").is_in(["Quercus robur", "Quercus petraea"]))

    # Overlaps
    overlaps = (
        species_df.group_by(["Lat", "Lon"])
        .agg(
            pl.col("Species").n_unique().alias("n_species"),
            pl.col("Species").unique().alias("species_list"),
        )
        .filter(pl.col("n_species") > 1)
    )

    # Plot
    HOVER_TMPL = (
        "<b>Latitude:</b> %{lat}<br>"
        "<b>Longitude:</b> %{lon}<br>"
        "<b>Species:</b> %{text}"
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
        ("Spruce Plots", spruce_df, "green", ["Spruce"] * len(spruce_df)),
        ("Pine Plots", pine_df, "red", ["Pine"] * len(pine_df)),
        ("Beech Plots", beech_df, "orange", ["Beech"] * len(beech_df)),
        ("Oak Plots", oak_df, "purple", ["Oak"] * len(oak_df)),
    ]

    for name, df, color, text in species_layers:
        add_scattermap(fig, df, name, color, text)

    # overlaps layer
    add_scattermap(
        fig,
        overlaps,
        "Overlapping plots",
        "black",
        [", ".join(s) for s in overlaps["species_list"].to_list()],
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

    fig.show()


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
