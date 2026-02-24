"""Helper functions for plots and EDA visualizations."""

import math

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
import seaborn as sns

pio.renderers.default = "notebook"

def plot_yearwise_social_class(
    tdf,
    ax,
    title = None,
    height=4,
    rotate_xticks=45
):
    """Plot year wise social class from ICP data."""
    year_counts = (
        tdf
        .select("period_end", "social_class_mode")
        .drop_nulls()
        .with_columns(
            pl.col("period_end").dt.year().alias("year")
        )
        .group_by(["year", "social_class_mode"])
        .len(name="count")
        .sort("year")
    )

    sns.barplot(
        data=year_counts,
        x="year",
        y="count",
        hue="social_class_mode",
        errorbar=None,
        ax=ax
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
        species_df
        .group_by(["Lat", "Lon"])
        .agg(
            pl.col("specie").n_unique().alias("n_species"),
            pl.col("specie").unique().alias("species_list"),
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
        ("Spruce Plots", spruce_df, "green",  ["Spruce"] * len(spruce_df)),
        ("Pine Plots",   pine_df,   "red",    ["Pine"]   * len(pine_df)),
        ("Beech Plots",  beech_df,  "orange", ["Beech"] * len(beech_df)),
        ("Oak Plots",    oak_df,    "purple", ["Oak"]   * len(oak_df)),
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
        n_rows,
        n_cols,
        figsize=(figsize_per_col * n_cols, figsize_per_row * n_rows),
        squeeze=False
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
                0.5, 0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="gray"
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
    for ax in axes[len(columns):]:
        ax.remove()

    plt.tight_layout()
    
    return fig, axes[:len(columns)]