"""Results of the leave-one-plot-out cross-validation of the nutrition modifier."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import polars as pl

    from trunx.config import results_data_folder

    return pl, plt, results_data_folder


@app.cell
def _(mo, results_data_folder):
    # One option per sweep, literature source and species, see scripts/run_loocv_sweep.py
    species_dirs = sorted(
        {
            str(path.parent.relative_to(results_data_folder))
            for path in results_data_folder.glob("loocv_nutrition_modifier*/*/*/l2=*")
        }
    )
    run = mo.ui.dropdown(species_dirs, value=species_dirs[0], label="Sweep / source / species")
    mo.output.replace(run)
    return (run,)


@app.cell
def _(pl, results_data_folder, run):
    # Fold-averaged RMSE of each L2 penalty, see loocv_nutrition_modifier.summarize_folds
    summary = pl.concat(
        [
            pl.read_csv(path)
            for path in (results_data_folder / run.value).glob("l2=*/loocv_summary.csv")
        ],
        how="diagonal_relaxed",
    ).with_columns(pl.col("split").str.split(" ").list.first().alias("set"))
    # RMSE of each plot and fold, see loocv_nutrition_modifier.leave_one_plot_out
    folds = pl.concat(
        [pl.read_csv(path) for path in (results_data_folder / run.value).glob("l2=*/loocv.csv")],
        how="diagonal_relaxed",
    ).with_columns(pl.col("split").str.split(" ").list.first().alias("set"))
    return folds, summary


@app.cell
def _(folds, mo, summary):
    variables = summary["variable"].unique(maintain_order=True).to_list()
    variable = mo.ui.dropdown(variables, value="DBH", label="Variable")
    sets = mo.ui.multiselect(["train", "test"], value=["test"], label="Set")
    rmse_view = mo.ui.radio(
        ["Mean ± std", "Each fold"], value="Each fold", label="RMSE", inline=True
    )
    _held_out_plots = sorted(folds["held_out"].unique())
    held_out = mo.ui.multiselect(
        {plot.removesuffix("_data"): plot for plot in _held_out_plots},
        value=[plot.removesuffix("_data") for plot in _held_out_plots],
        label="Held-out plots",
    )
    mo.output.replace(
        mo.vstack([mo.hstack([variable, sets, rmse_view], justify="start"), held_out])
    )
    return held_out, rmse_view, sets, variable


@app.cell
def _(folds, held_out, pl, variable):
    # Folds of the selected held-out plots, and their mean and std as in
    # `loocv_nutrition_modifier.summarize_folds`
    selected_folds = folds.filter(
        pl.col("variable") == variable.value, pl.col("held_out").is_in(held_out.value)
    )
    selected_summary = selected_folds.group_by("set", "modifier_l2").agg(
        pl.col("rmse_fitted", "rmse_phys").mean().name.suffix("_mean"),
        pl.col("rmse_fitted", "rmse_phys").std().name.suffix("_std"),
    )
    return selected_folds, selected_summary


@app.cell
def _(folds, plt):
    # Each held-out plot keeps its color whichever plots are selected. Dark tab20 shades
    # first, so the first ten plots have clearly different hues
    _tab20 = plt.get_cmap("tab20").colors
    plot_colors = dict(
        zip(sorted(folds["held_out"].unique()), _tab20[::2] + _tab20[1::2], strict=False)
    )
    return (plot_colors,)


@app.cell
def _(
    mo,
    pl,
    plot_colors,
    plt,
    rmse_view,
    selected_folds,
    selected_summary,
    sets,
    variable,
):
    l2_fig, _ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for _set, _color, _marker in (("train", "tab:blue", "x"), ("test", "tab:orange", "o")):
        if _set not in sets.value:
            continue
        _rows = selected_summary.filter(pl.col("set") == _set).sort("modifier_l2")
        if rmse_view.value == "Mean ± std":
            _ax.errorbar(
                _rows["modifier_l2"],
                _rows["rmse_fitted_mean"],
                yerr=_rows["rmse_fitted_std"],
                marker="o",
                capsize=3,
                color=_color,
                label=f"{_set}, with modifier",
            )
        else:
            # One point per plot and fold, colored by the fold's held-out plot, with the
            # mean across them as a line
            for (_plot,), _fold_rows in (
                selected_folds.filter(pl.col("set") == _set)
                .sort("held_out")
                .group_by("held_out", maintain_order=True)
            ):
                _ax.scatter(
                    _fold_rows["modifier_l2"],
                    _fold_rows["rmse_fitted"],
                    s=16,
                    alpha=0.6,
                    marker=_marker,
                    color=plot_colors[_plot],
                    label=f"{_set}, held out {str(_plot).removesuffix('_data')}",
                )
            _color = "black" if _set == "test" else "gray"
            _ax.plot(
                _rows["modifier_l2"],
                _rows["rmse_fitted_mean"],
                color=_color,
                label=f"{_set}, with modifier, mean",
            )
        # Without the modifier the RMSE does not depend on the L2 penalty
        _ax.axhline(
            _rows["rmse_phys_mean"].mean(),
            color=_color,
            linestyle="--",
            label=f"{_set}, without modifier",
        )
    _ax.set_xlabel("L2 penalty (modifier_l2)")
    _ax.set_ylabel(f"{variable.value} RMSE")
    _ax.set_title(f"Fold-averaged RMSE vs L2 penalty, {variable.value}")
    _ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="small")
    mo.output.replace(l2_fig)
    return


@app.cell
def _(mo, pl, plt, selected_summary, sets, variable):
    improvement_fig, _ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for _set, _color in (("train", "tab:blue"), ("test", "tab:orange")):
        if _set not in sets.value:
            continue
        _rows = (
            selected_summary.filter(pl.col("set") == _set)
            .sort("modifier_l2")
            # Improvement of the fold-averaged RMSE, as in `loocv_nutrition_modifier._rmse_table`
            .with_columns(
                (
                    100
                    * (pl.col("rmse_phys_mean") - pl.col("rmse_fitted_mean"))
                    / pl.col("rmse_phys_mean")
                ).alias("improvement")
            )
        )
        _ax.plot(_rows["modifier_l2"], _rows["improvement"], marker="o", color=_color, label=_set)
    _ax.axhline(0, color="black", linewidth=0.8)
    _ax.set_xlabel("L2 penalty (modifier_l2)")
    _ax.set_ylabel(f"{variable.value} RMSE improvement (%)")
    _ax.set_title(f"RMSE improvement of the modifier vs L2 penalty, {variable.value}")
    _ax.legend()
    mo.output.replace(improvement_fig)
    return


@app.cell
def _(pl, results_data_folder, run):
    import json

    import numpy as np

    from trunx.gp3.bayesiancalibrations.pymc_icp_plots import prepare_plot_input
    from trunx.gp3.extended_helper import DEPOSITION_CHANNELS
    from trunx.gp3.nn_nutrition_modifier import _months_to_dates
    from trunx.gp3.prepare_data import prepare_data

    # Monthly deposition of each plot of the run, from its 3PG input file
    _job = json.loads(next((results_data_folder / run.value).glob("l2=*/job.json")).read_text())
    _tables = []
    for _plot_id in _job["plot_ids"]:
        _input_data = prepare_data(
            prepare_plot_input(_plot_id, literature_source=_job["literature_source"])
        )
        _values = {
            name: np.asarray(getattr(_input_data.deposition, DEPOSITION_CHANNELS[name]))
            for name in ("N", "S")
        }
        _tables.append(
            pl.DataFrame(
                {
                    # Same plot names as `held_out` of `loocv.csv`
                    "held_out": f"{_plot_id}_data",
                    "date": _months_to_dates(
                        int(_input_data.site.year_i[0]),
                        int(_input_data.site.month_i[0]),
                        len(_values["N"]),
                    ).astype("datetime64[D]"),  # polars has no month resolution
                    **_values,
                }
            )
        )
    deposition = pl.concat(_tables)
    return (deposition,)


@app.cell
def _(deposition, mo):
    _plots = sorted(deposition["held_out"].unique())
    deposition_plots = mo.ui.multiselect(
        {plot.removesuffix("_data"): plot for plot in _plots},
        value=[plot.removesuffix("_data") for plot in _plots],
        label="Deposition plots",
    )
    mo.output.replace(deposition_plots)
    return (deposition_plots,)


@app.cell
def _(deposition, deposition_plots, mo, pl, plot_colors, plt):
    deposition_fig, _axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    for (_plot,), _rows in (
        deposition.filter(pl.col("held_out").is_in(deposition_plots.value))
        .sort("held_out")
        .group_by("held_out", maintain_order=True)
    ):
        for _ax, _name in zip(_axes, ("N", "S"), strict=True):
            _ax.plot(
                _rows["date"],
                _rows[_name],
                linewidth=0.8,
                color=plot_colors[_plot],
                label=str(_plot).removesuffix("_data"),
            )
    for _ax, _name in zip(_axes, ("N", "S"), strict=True):
        _ax.set_ylabel(f"{_name} deposition (kg ha⁻¹ month⁻¹)")
    _axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="small", title="Plot")
    _axes[0].set_title("Monthly deposition of plots")
    mo.output.replace(deposition_fig)
    return


@app.cell
def _(mo, pl, results_data_folder, run, summary, variable):
    # Folder of each L2 penalty, keyed by its value, e.g. `l2=1` and `l2=1.0` alike
    l2_dirs = {
        float(path.name.removeprefix("l2=")): path
        for path in (results_data_folder / run.value).glob("l2=*")
    }
    # Default to the L2 penalty with the lowest held-out RMSE of the variable
    best_l2 = (
        summary.filter(pl.col("variable") == variable.value, pl.col("set") == "test")
        .sort("rmse_fitted_mean")["modifier_l2"]
        .first()
    )
    l2 = mo.ui.dropdown(
        {str(value): value for value in sorted(l2_dirs)},
        value=str(float(best_l2)),
        label="L2 penalty of the figures",
    )
    mo.output.replace(l2)
    return l2, l2_dirs


@app.cell
def _():
    # Figures of `loocv_nutrition_modifier.run_loocv` per category, by file name
    figure_categories = {
        "Fit to observations": ["observed_vs_predicted_held_out"],
        "Modifier response": [
            "modifier_response_surface",
            "modifier_score_before_tanh",
            "modifier_response_curve",
            "f_nutri_classic_learnable_deposition_held_out",
        ],
        "Modifier impact over time": [
            "alpha_c_held_out",
            "f_nutri_classic_learnable_held_out",
            "NPP_held_out",
        ],
        "Modifier impact vs deposition": [
            "alpha_c_deposition_held_out",
            "modifier_pass_through_held_out",
        ],
        "Residual (with − without modifier)": ["alpha_c_residual_deposition_held_out"],
    }
    return (figure_categories,)


@app.cell
def _(figure_categories, l2, l2_dirs, mo):
    # One tab per category, each with a selector of its figures
    _images = {path.stem: path for path in l2_dirs[l2.value].glob("images/*/*.png")}
    _categorized = {stem for stems in figure_categories.values() for stem in stems}
    _groups = {
        category: [stem for stem in stems if stem in _images]
        for category, stems in figure_categories.items()
    } | {"Other": sorted(stem for stem in _images if stem not in _categorized)}
    figure_selects = mo.ui.dictionary(
        {
            category: mo.ui.dropdown(
                {stem: _images[stem] for stem in stems}, value=stems[0], label="Figure"
            )
            for category, stems in _groups.items()
            if stems
        }
    )
    category_tabs = mo.ui.tabs({category: figure_selects[category] for category in figure_selects})
    mo.output.replace(category_tabs)
    return category_tabs, figure_selects


@app.cell
def _(category_tabs, figure_selects, mo):
    # Each plot predicted by the fold that held it out
    mo.output.replace(mo.image(figure_selects.value[category_tabs.value], width="100%"))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
