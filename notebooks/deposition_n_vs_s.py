"""Monthly N against S deposition of ICP plots, one panel per plot."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import math
    import os

    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from trunx.config import clean_data_folder, results_data_folder
    from trunx.gp3.extended_helper import DEPOSITION_LIMITS

    return (
        DEPOSITION_LIMITS,
        clean_data_folder,
        math,
        mcolors,
        np,
        os,
        pl,
        plt,
        results_data_folder,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Monthly deposition

    Monthly deposition per plot (kg ha⁻¹ month⁻¹, the scale of the nutrition modifier
    inputs), keeping the months with both total N (measured values only) and SO4
    (gap-filled in the monthly table).
    """)
    return


@app.cell
def _(clean_data_folder, mo, os, pl):
    monthly = pl.read_parquet(os.path.join(clean_data_folder, "icp_monthly_deposition.parquet"))
    months = (
        monthly.select(
            "plot_id",
            "year",
            "month",
            pl.col("dep_n_tot").alias("N"),
            pl.col("dep_s_so4").alias("S"),
        )
        .drop_nulls(["N", "S"])
        .sort("plot_id", "year", "month")
    )
    mo.output.replace(months)
    return monthly, months


@app.cell
def _():
    # Plots of the nutrition modifier, per dominant species: `select_icp_plots` and
    # enough measured deposition (slow, as it builds each plot's 3PG input)
    from trunx.gp3.deposition_coverage import selected_plot_ids

    modifier_plots = selected_plot_ids()
    return (modifier_plots,)


@app.cell
def _(mo, modifier_plots):
    species = mo.ui.dropdown(
        options=sorted(modifier_plots),
        value="Picea abies" if "Picea abies" in modifier_plots else sorted(modifier_plots)[0],
        label="Species",
    )
    mo.output.replace(species)
    return (species,)


@app.cell
def _(mo, modifier_plots, months, species):
    plot_options = sorted(months["plot_id"].unique().to_list())
    plot_selector = mo.ui.multiselect(
        options=plot_options,
        value=[plot for plot in modifier_plots[species.value] if plot in plot_options],
        label=f"Plots ({species.value}, editable)",
    )
    mo.output.replace(plot_selector)
    return (plot_selector,)


@app.cell
def _(math, mcolors, mo, months, pl, plot_selector, plt):
    selected = months.filter(pl.col("plot_id").is_in(plot_selector.value))
    plot_ids = sorted(selected["plot_id"].unique().to_list())
    unit = "kg ha⁻¹ month⁻¹"

    n_cols = min(4, max(len(plot_ids), 1))
    n_rows = max(math.ceil(len(plot_ids) / n_cols), 1)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 3.2 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    # One light-to-dark blue scale for the years, shared by all panels
    cmap = mcolors.LinearSegmentedColormap.from_list("years", ["#cfe0f5", "#08306b"])
    norm = mcolors.Normalize(selected["year"].min(), selected["year"].max())

    for ax, plot_id in zip(axes.flat, plot_ids, strict=False):
        plot_months = selected.filter(pl.col("plot_id") == plot_id)
        points = ax.scatter(
            plot_months["N"],
            plot_months["S"],
            c=plot_months["year"],
            cmap=cmap,
            norm=norm,
            s=12,
            alpha=0.8,
            linewidths=0,
        )
        ax.set_title(f"{plot_id} ({plot_months.height} months)", fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(plot_ids) :]:
        ax.set_visible(False)
    # x-axis labels on the lowest panel of each column, as the last row can be partly empty
    for column in axes.T:
        lowest = [ax for ax in column if ax.get_visible()][-1]
        lowest.tick_params(labelbottom=True)
        lowest.set_xlabel(f"Total N ({unit})")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"SO₄ ({unit})")

    if plot_ids:
        colorbar = fig.colorbar(points, ax=axes, shrink=0.6)
        colorbar.set_label("Year")
        colorbar.outline.set_visible(False)
    fig.suptitle("Monthly N against S deposition")
    mo.output.replace(fig)
    return (unit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Trajectories across the (N, S) space

    Annual mean deposition of each selected plot (years with all 12 months), joined year
    to year. One color per plot, points fade from transparent (first year) to opaque
    (last year).
    """)
    return


@app.cell
def _(mcolors, mo, months, pl, plot_selector, plt, unit):
    trajectories = (
        months.filter(pl.col("plot_id").is_in(plot_selector.value))
        .group_by("plot_id", "year")
        .agg(pl.len().alias("n_months"), pl.col("N", "S").mean())
        .filter(pl.col("n_months") == 12)
        .sort("plot_id", "year")
    )

    trajectory_fig, trajectory_ax = plt.subplots(figsize=(9, 7), layout="constrained")
    year_norm = mcolors.Normalize(trajectories["year"].min(), trajectories["year"].max())
    plot_colors = plt.get_cmap("tab20").colors
    legend_handles = []
    for _i, ((_plot_id,), _path) in enumerate(
        trajectories.group_by("plot_id", maintain_order=True)
    ):
        _color = plot_colors[_i % len(plot_colors)]
        # Alpha from 0.15 (first year) to 1 (last year), shared by all plots
        _alphas = 0.15 + 0.85 * year_norm(_path["year"].to_numpy())
        trajectory_ax.plot(_path["N"], _path["S"], color=_color, alpha=0.4, linewidth=1, zorder=1)
        trajectory_ax.scatter(
            _path["N"],
            _path["S"],
            color=[mcolors.to_rgba(_color, _alpha) for _alpha in _alphas],
            s=24,
            linewidths=0,
            zorder=2,
        )
        legend_handles.append(
            plt.Line2D([], [], color=_color, marker="o", linestyle="", label=_plot_id)
        )
    trajectory_ax.set_xlabel(f"Total N ({unit})")
    trajectory_ax.set_ylabel(f"SO₄ ({unit})")
    trajectory_ax.grid(alpha=0.25, linewidth=0.6)
    trajectory_ax.spines[["top", "right"]].set_visible(False)

    if legend_handles:
        trajectory_ax.legend(
            handles=legend_handles,
            title="Plot",
            fontsize=8,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
        )
    trajectory_ax.set_title(
        f"Annual N and S deposition trajectories "
        f"({trajectories['year'].min()}–{trajectories['year'].max()}, fainter = earlier)"
    )
    mo.output.replace(trajectory_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Is the N optimum of the modifier a N–S coupling?

    Fitted modifier $f = 1 + \tanh(w_{00} + w_{10} N + w_{01} S + w_{11} N S)$ of the
    leave-one-plot-out run (weights averaged over folds, standardised N and S). At fixed
    S it is monotonic in N, so an optimum along N can only come from S changing with N.

    - **Left**: modifier over the (N, S) space, the months of the run's plots and the
      median S of each N decile (the coupled path).
    - **Right**: modifier along N on the coupled path, and at fixed S (10th, 50th and
      90th percentile). An optimum on the coupled path only is the coupling.

    The input standardisation is approximated by the clipped months of the run's plots
    in the monthly table.
    """)
    return


@app.cell
def _(mo, pl, results_data_folder):
    loocv_dir = results_data_folder / "nn_nutrition_modifier"
    loocv_summary = pl.read_csv(loocv_dir / "loocv_summary.csv")
    loocv_plots = (
        pl.read_csv(loocv_dir / "loocv.csv")["plot"].str.strip_suffix("_data").unique().to_list()
    )
    l2_values = sorted(loocv_summary["modifier_l2"].unique().to_list())
    l2_selector = mo.ui.dropdown(
        options={str(l2): l2 for l2 in l2_values}, value=str(l2_values[0]), label="modifier_l2"
    )
    mo.output.replace(l2_selector)
    return l2_selector, loocv_plots, loocv_summary


@app.cell
def _(
    DEPOSITION_LIMITS,
    l2_selector,
    loocv_plots,
    loocv_summary,
    mcolors,
    mo,
    months,
    np,
    pl,
    plt,
    unit,
):
    # Months of the run's plots, clipped as the modifier inputs
    modifier_months = months.filter(pl.col("plot_id").is_in(loocv_plots)).with_columns(
        pl.col(name).clip(*DEPOSITION_LIMITS[name]) for name in ("N", "S")
    )
    input_mean = modifier_months.select("N", "S").mean().row(0)
    input_std = modifier_months.select("N", "S").std(ddof=0).row(0)
    weights = loocv_summary.filter(pl.col("modifier_l2") == l2_selector.value).row(0, named=True)

    def _modifier(n: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Fitted modifier at N `n` and S `s` (kg ha⁻¹ month⁻¹)."""
        n_std = (n - input_mean[0]) / input_std[0]
        s_std = (s - input_mean[1]) / input_std[1]
        return 1 + np.tanh(
            weights["w00_mean"]
            + weights["w10_mean"] * n_std
            + weights["w01_mean"] * s_std
            + weights["w11_mean"] * n_std * s_std
        )

    # Median S of each N decile: S along the N gradient
    coupled_path = (
        modifier_months.with_columns(pl.col("N").qcut(10).alias("n_bin"))
        .group_by("n_bin")
        .agg(pl.col("N", "S").median())
        .sort("N")
    )

    coupling_fig, (surface_ax, along_n_ax) = plt.subplots(
        1, 2, figsize=(13, 5.2), layout="constrained"
    )
    grid_n, grid_s = np.meshgrid(
        np.linspace(0, modifier_months["N"].max(), 200),
        np.linspace(0, modifier_months["S"].max(), 200),
    )
    grid_f = _modifier(grid_n, grid_s)
    # Diverging scale centred on 1 (no effect)
    surface = surface_ax.contourf(
        grid_n,
        grid_s,
        grid_f,
        levels=21,
        cmap="RdBu",
        norm=mcolors.TwoSlopeNorm(
            vcenter=1, vmin=min(grid_f.min(), 0.99), vmax=max(grid_f.max(), 1.01)
        ),
    )
    surface_ax.contour(grid_n, grid_s, grid_f, levels=[1.0], colors="black", linewidths=0.8)
    surface_ax.scatter(
        modifier_months["N"], modifier_months["S"], s=4, color="#333333", alpha=0.15, linewidths=0
    )
    surface_ax.plot(
        coupled_path["N"],
        coupled_path["S"],
        "o-",
        color="black",
        linewidth=2,
        label="Coupled path",
    )
    surface_ax.set_xlabel(f"Total N ({unit})")
    surface_ax.set_ylabel(f"SO₄ ({unit})")
    surface_ax.legend(frameon=False, loc="upper left")
    surface_cbar = coupling_fig.colorbar(surface, ax=surface_ax, shrink=0.8)
    surface_cbar.set_label("Modifier (1 = no effect)")
    surface_cbar.outline.set_visible(False)
    surface_ax.set_title("Modifier over (N, S)")

    n_range = np.linspace(coupled_path["N"].min(), coupled_path["N"].max(), 200)
    s_levels = (
        modifier_months["S"].quantile(0.1),
        modifier_months["S"].median(),
        modifier_months["S"].quantile(0.9),
    )
    for _label, _s, _color in zip(
        ("S 10th pct", "S median", "S 90th pct"),
        s_levels,
        ("#9ecae1", "#4292c6", "#08306b"),
        strict=True,
    ):
        along_n_ax.plot(
            n_range,
            _modifier(n_range, np.full_like(n_range, _s)),
            color=_color,
            linewidth=1.5,
            label=f"{_label} ({_s:.2f})",
        )
    along_n_ax.plot(
        coupled_path["N"],
        _modifier(coupled_path["N"].to_numpy(), coupled_path["S"].to_numpy()),
        "o-",
        color="black",
        linewidth=2,
        label="Coupled path",
    )
    along_n_ax.axhline(1, color="#9a9a9a", linewidth=0.8, linestyle="--")
    along_n_ax.set_xlabel(f"Total N ({unit})")
    along_n_ax.set_ylabel("Modifier")
    along_n_ax.grid(alpha=0.25, linewidth=0.6)
    along_n_ax.spines[["top", "right"]].set_visible(False)
    along_n_ax.legend(frameon=False)
    along_n_ax.set_title("Modifier along N")
    coupling_fig.suptitle(
        f"Fitted modifier, {len(loocv_plots)} plots, modifier_l2 = {l2_selector.value}"
    )
    mo.output.replace(coupling_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlation between depositions

    - **Monthly, pooled**: every plot-month together.
    - **Between plots**: the mean of each plot, how plots differ from each other.
    - **Within plot**: each month minus its plot's mean, how a plot varies over time.
    - **Annual**: the mean of each plot-year, only years with all 12 months.
    - **Year to year**: the mean over plots of each year, the shared time trend.

    Fluxes are in kg ha⁻¹ month⁻¹; alkalinity, pH and conductivity are concentrations.
    """)
    return


@app.cell
def _(mo):
    # Deposition variables of the paper and their column in the monthly table
    deposition_variables = {
        "N": "dep_n_tot",
        "SO4": "dep_s_so4",
        "Alkalinity": "dep_alk",
        "pH": "dep_ph",
        "Conductivity": "dep_cond",
        "K": "dep_k",
        "Co": "dep_co",
        "Pb": "dep_pb",
        "DOC": "dep_doc",
    }
    plot_scope = mo.ui.radio(
        options=["All ICP plots", "Selected plots"], value="Selected plots", label="Plots"
    )
    method = mo.ui.radio(options=["pearson", "spearman"], value="pearson", label="Correlation")
    mo.output.replace(mo.hstack([plot_scope, method], justify="start", gap=2))
    return deposition_variables, method, plot_scope


@app.cell
def _(deposition_variables, monthly, pl, plot_scope, plot_selector):
    names = list(deposition_variables)
    deposition = monthly.select(
        "plot_id",
        "year",
        "month",
        *(pl.col(column).alias(name) for name, column in deposition_variables.items()),
    )
    if plot_scope.value == "Selected plots":
        deposition = deposition.filter(pl.col("plot_id").is_in(plot_selector.value))

    # Annual means, kept only for years with all 12 months of the variable
    annual = deposition.group_by("plot_id", "year").agg(
        pl.when(pl.col(name).count() == 12).then(pl.col(name).mean()).alias(name) for name in names
    )
    levels = {
        "Monthly, pooled": deposition,
        "Between plots": deposition.group_by("plot_id").agg(pl.col(names).mean()),
        "Within plot": deposition.with_columns(
            (pl.col(name) - pl.col(name).mean().over("plot_id")).alias(name) for name in names
        ),
        "Annual": annual,
        "Year to year": annual.group_by("year").agg(pl.col(names).mean()),
    }
    return levels, names


@app.cell
def _(pl):
    def correlation_matrix(
        df: pl.DataFrame, names: list[str], method: str, min_pairs: int = 10
    ) -> pl.DataFrame:
        """Correlation of each pair of `names`, over the rows where both have a value.

        Parameters
        ----------
        df : pl.DataFrame
            Table with one column per item of `names`.
        names : list[str]
            Columns to correlate.
        method : str
            "pearson" or "spearman".
        min_pairs : int
            Fewest rows with both values for a correlation; null below.

        Returns
        -------
        pl.DataFrame
            Long table with `x`, `y`, `r` and `n` (rows used) per pair.
        """
        rows = []
        for x in names:
            for y in names:
                if x == y:
                    rows.append({"x": x, "y": y, "r": 1.0, "n": df[x].count()})
                    continue
                pairs = df.select(x, y).drop_nulls()
                r = (
                    pairs.select(pl.corr(x, y, method=method)).item()
                    if pairs.height >= min_pairs
                    else None
                )
                rows.append({"x": x, "y": y, "r": r, "n": pairs.height})
        return pl.DataFrame(
            rows, schema={"x": pl.String, "y": pl.String, "r": pl.Float64, "n": pl.Int64}
        )

    return (correlation_matrix,)


@app.cell
def _(correlation_matrix, levels, method, mo, names, pl):
    correlations = pl.concat(
        correlation_matrix(df, names, method.value).with_columns(pl.lit(level).alias("level"))
        for level, df in levels.items()
    )
    # Each variable against N and SO4 at every level
    against_n_s = (
        correlations.filter(pl.col("y").is_in(["N", "SO4"]), pl.col("x") != pl.col("y"))
        .with_columns(pl.col("r").round(2))
        .pivot("level", index=["y", "x"], values="r")
        .rename({"y": "against", "x": "variable"})
        .sort("against", "variable")
    )
    level_selector = mo.ui.dropdown(
        options=list(levels), value="Monthly, pooled", label="Level of the heatmap"
    )
    mo.output.replace(
        mo.vstack(
            [
                mo.md(f"**Correlation ({method.value}) with N and SO4 at every level**"),
                against_n_s,
                level_selector,
            ]
        )
    )
    return correlations, level_selector


@app.cell
def _(correlations, level_selector, method, mo, names, pl, plt):
    level_correlations = correlations.filter(pl.col("level") == level_selector.value)
    matrix = level_correlations.pivot("y", index="x", values="r").select(names).to_numpy()
    n_rows_used = level_correlations.filter(pl.col("x") == pl.col("y"))["n"].to_list()

    heatmap_fig, heatmap_ax = plt.subplots(figsize=(7.5, 6.2), layout="constrained")
    # Diverging blue-red scale with a neutral midpoint at r = 0
    image = heatmap_ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    heatmap_ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    heatmap_ax.set_yticks(
        range(len(names)),
        [f"{name} (n={n})" for name, n in zip(names, n_rows_used, strict=True)],
    )
    for row in range(len(names)):
        for col in range(len(names)):
            value = matrix[row, col]
            if value == value:  # NaN when too few pairs
                heatmap_ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > 0.6 else "#222222",
                )
    heatmap_cbar = heatmap_fig.colorbar(image, ax=heatmap_ax, shrink=0.8)
    heatmap_cbar.set_label(f"{method.value} r")
    heatmap_cbar.outline.set_visible(False)
    heatmap_ax.set_title(f"Deposition correlations: {level_selector.value}")
    mo.output.replace(heatmap_fig)
    return


if __name__ == "__main__":
    app.run()
