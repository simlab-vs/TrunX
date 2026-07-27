"""Comparison of ICP and ERA5 monthly weather data."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    sys.path.append(str(Path(__file__).parent.parent))
    from trunx.config import clean_data_folder
    from trunx.datasets.era5_icp_weather import get_plot_weather
    from trunx.gp3.weather_processing import aggregate_icp_monthly

    return (
        aggregate_icp_monthly,
        clean_data_folder,
        get_plot_weather,
        np,
        os,
        pl,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ICP vs ERA5 monthly weather comparison
    """)
    return


@app.cell
def _(clean_data_folder, os, pl):
    icp_wdf = pl.read_parquet(os.path.join(clean_data_folder, "ICP_weather_data.parquet"))
    era5_wdf = pl.read_parquet(os.path.join(clean_data_folder, "era5_weather_icp_plots.parquet"))
    return era5_wdf, icp_wdf


@app.cell
def _(era5_wdf, icp_wdf, pl):
    # Plots for which both ICP and ERA5 weather are available
    common_plot_ids = sorted(
        set(icp_wdf["plot_id"].unique().to_list()) & set(era5_wdf["plot_id"].unique().to_list())
    )

    icp_wdf_common = icp_wdf.filter(pl.col("plot_id").is_in(common_plot_ids))
    return common_plot_ids, icp_wdf_common


@app.cell
def _(common_plot_ids, mo):
    plot_selector_ui = mo.ui.dropdown(
        options=common_plot_ids, label="🌲 Select ICP Forest Plot", value=common_plot_ids[-1]
    )
    mo.vstack([plot_selector_ui])
    return (plot_selector_ui,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Monthly weather: ICP vs ERA5, same plot
    """)
    return


@app.cell
def _(pl):
    WEATHER_LABELS = {
        "tmp_ave": "Average temperature (°C)",
        "tmp_min": "Minimum temperature (°C)",
        "tmp_max": "Maximum temperature (°C)",
        "prcp": "Precipitation (mm)",
        "srad": "Solar radiation (MJ/m²)",
        "frost_days": "Frost days (days/month)",
    }

    def to_long(monthly, source):
        """Reshape a monthly weather DataFrame to long format tagged with its source."""
        return (
            monthly.with_columns(pl.date(pl.col("year"), pl.col("month"), 1).alias("date"))
            .unpivot(
                index="date", on=list(WEATHER_LABELS), variable_name="metric", value_name="value"
            )
            .with_columns(pl.lit(source).alias("source"))
            .drop_nulls("value")
        )

    def build_comparison_table(icp_monthly, era5_monthly):
        """Combine ICP and ERA5 monthly weather into one long-format table."""
        return pl.concat([to_long(icp_monthly, "ICP"), to_long(era5_monthly, "ERA5")])

    return WEATHER_LABELS, build_comparison_table


@app.cell
def _(
    aggregate_icp_monthly,
    build_comparison_table,
    get_plot_weather,
    icp_wdf_common,
):
    def comparison_for_plot(era5_wdf, plot_id):
        """Build the ICP vs ERA5 long-format comparison table for one plot."""
        icp_monthly = aggregate_icp_monthly(icp_wdf_common, plot_id)
        _, era5_monthly = get_plot_weather(plot_id, era5_wdf)
        return build_comparison_table(icp_monthly, era5_monthly)

    return (comparison_for_plot,)


@app.cell
def _(comparison_for_plot, era5_wdf, plot_selector_ui):
    comparison_df = comparison_for_plot(era5_wdf, plot_selector_ui.value)
    return (comparison_df,)


@app.cell
def _(WEATHER_LABELS, pl, plt):
    def plot_timeseries_comparison(comparison_df, plot_id):
        """Plot ICP and ERA5 monthly weather on the same axes for one plot."""
        colors = {"ICP": "tab:blue", "ERA5": "tab:orange"}
        metrics = [m for m in WEATHER_LABELS if comparison_df.filter(pl.col("metric") == m).height]

        fig, axes = plt.subplots(len(metrics), 1, figsize=(14, len(metrics) * 3.5), sharex=True)
        axes = [axes] if len(metrics) == 1 else axes

        for ax, metric in zip(axes, metrics, strict=True):
            metric_df = comparison_df.filter(pl.col("metric") == metric)
            for source, color in colors.items():
                source_df = metric_df.filter(pl.col("source") == source).sort("date")
                if source_df.height == 0:
                    continue
                ax.plot(
                    source_df["date"],
                    source_df["value"],
                    label=source,
                    color=color,
                    linewidth=1.5,
                    marker="o",
                    markersize=3,
                    alpha=0.8,
                )
            ax.set_ylabel(WEATHER_LABELS[metric])
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel("Date")
        fig.suptitle(
            f"ICP vs ERA5 monthly weather — plot {plot_id}", fontsize=14, fontweight="bold"
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return (plot_timeseries_comparison,)


@app.cell
def _(comparison_df, plot_selector_ui, plot_timeseries_comparison):
    plot_timeseries_comparison(comparison_df, plot_selector_ui.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Agreement between ICP and ERA5 (scatter)
    """)
    return


@app.cell
def _(WEATHER_LABELS, np, pl, plt):
    def plot_scatter_comparison(comparison_df, plot_id):
        """Scatter ICP against ERA5 monthly values for months present in both sources."""
        wide = comparison_df.pivot(
            index=["date", "metric"], on="source", values="value"
        ).drop_nulls()
        if "ICP" not in wide.columns or "ERA5" not in wide.columns:
            print(f"No overlapping ICP/ERA5 months for plot {plot_id}.")
            return
        metrics = [m for m in WEATHER_LABELS if wide.filter(pl.col("metric") == m).height]

        n_cols = min(3, len(metrics))
        n_rows = (len(metrics) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        for ax, metric in zip(axes, metrics, strict=False):
            metric_df = wide.filter(pl.col("metric") == metric)
            icp_vals = metric_df["ICP"].to_numpy()
            era5_vals = metric_df["ERA5"].to_numpy()
            ax.scatter(icp_vals, era5_vals, s=15, alpha=0.6)
            lims = [min(icp_vals.min(), era5_vals.min()), max(icp_vals.max(), era5_vals.max())]
            ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
            ax.set_xlabel(f"ICP — {WEATHER_LABELS[metric]}")
            ax.set_ylabel(f"ERA5 — {WEATHER_LABELS[metric]}")
            ax.set_title(WEATHER_LABELS[metric])
            ax.grid(True, alpha=0.3)

        for ax in axes[len(metrics) :]:
            ax.axis("off")

        fig.suptitle(f"ICP vs ERA5 agreement — plot {plot_id}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    return (plot_scatter_comparison,)


@app.cell
def _(comparison_df, plot_scatter_comparison, plot_selector_ui):
    plot_scatter_comparison(comparison_df, plot_selector_ui.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Difference statistics for this plot
    """)
    return


@app.cell
def _(pl):
    def diff_stats(comparison_df):
        """Compute bias, RMSE, and correlation between ICP and ERA5 for overlapping months."""
        wide = comparison_df.pivot(
            index=["date", "metric"], on="source", values="value"
        ).drop_nulls()
        if "ICP" not in wide.columns or "ERA5" not in wide.columns:
            return pl.DataFrame(
                schema={
                    "metric": pl.String,
                    "n_months": pl.UInt32,
                    "bias": pl.Float64,
                    "rmse": pl.Float64,
                    "correlation": pl.Float64,
                }
            )
        return (
            wide.group_by("metric")
            .agg(
                pl.len().alias("n_months"),
                (pl.col("ERA5") - pl.col("ICP")).mean().alias("bias"),
                (pl.col("ERA5") - pl.col("ICP")).pow(2).mean().sqrt().alias("rmse"),
                pl.corr("ICP", "ERA5").alias("correlation"),
            )
            .sort("metric")
        )

    return (diff_stats,)


@app.cell
def _(comparison_df, diff_stats):
    diff_stats(comparison_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Network-wide agreement across all common ICP/ERA5 plots
    """)
    return


@app.cell
def _(common_plot_ids, comparison_for_plot, diff_stats, era5_wdf, pl):
    all_stats = [
        stats.with_columns(pl.lit(plot_id).alias("plot_id"))
        for plot_id in common_plot_ids
        for stats in [diff_stats(comparison_for_plot(era5_wdf, plot_id))]
        if stats.height
    ]
    all_stats_df = pl.concat(all_stats)
    return (all_stats_df,)


@app.cell
def _(pl, plt):
    def plot_diff_summary(all_stats_df):
        """Summarize ICP vs ERA5 agreement across all plots via bias, RMSE, and correlation."""
        metrics = sorted(all_stats_df["metric"].unique().to_list())
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        panels = [("bias", "Bias (ERA5 − ICP)"), ("rmse", "RMSE"), ("correlation", "Correlation")]
        for ax, (column, title) in zip(axes, panels, strict=True):
            data = [all_stats_df.filter(pl.col("metric") == m)[column].to_list() for m in metrics]
            ax.boxplot(data, tick_labels=metrics)
            if column == "bias":
                ax.axhline(0, color="red", linestyle="--", alpha=0.5)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3)

        fig.suptitle("ICP vs ERA5 agreement across all plots", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    return (plot_diff_summary,)


@app.cell
def _(all_stats_df, plot_diff_summary):
    plot_diff_summary(all_stats_df)
    return


if __name__ == "__main__":
    app.run()
