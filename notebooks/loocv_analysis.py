"""Results of the leave-one-plot-out cross-validation of the nutrition modifier."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import json

    import arviz as az
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from trunx.config import results_data_folder
    from trunx.gp3.bayesiancalibrations.pymc_icp_plots import prepare_plot_input
    from trunx.gp3.extended_helper import (
        DEPOSITION_CHANNELS,
        bivariate_quadratic_nm,
        liebig_nm,
        mlp_nm,
        poly_nm,
        product_nm,
    )
    from trunx.gp3.loocv_nutrition_modifier import default_config
    from trunx.gp3.model_inputs import ExtendedParams
    from trunx.gp3.nn_nutrition_modifier import _months_to_dates as months_to_dates
    from trunx.gp3.nn_nutrition_modifier import (
        initial_dbh,
        input_statistics,
        matched_initial_state,
        neutral_modifier_params,
    )
    from trunx.gp3.prepare_data import prepare_data
    from trunx.gp3.run_3pg import run_3pg

    return (
        DEPOSITION_CHANNELS,
        ExtendedParams,
        az,
        bivariate_quadratic_nm,
        default_config,
        initial_dbh,
        input_statistics,
        jnp,
        json,
        liebig_nm,
        matched_initial_state,
        mlp_nm,
        months_to_dates,
        neutral_modifier_params,
        np,
        pl,
        plt,
        poly_nm,
        prepare_data,
        prepare_plot_input,
        product_nm,
        results_data_folder,
        run_3pg,
    )


@app.cell(hide_code=True)
def _(mo, pl, plt, results_data_folder):
    from pathlib import Path

    # Run folders of the cluster sweep, at
    # cluster/<source>/<inputs>/loocv/<selection>/<modifier>/<species>, see
    # scripts/run_loocv_sweep.py
    _cluster_folder = results_data_folder / "cluster"
    _run_dirs = sorted({path.parent for path in _cluster_folder.glob("*/*/loocv/*/*/*/l2=*")})
    # Position of each setting in the path of a run folder, its default and its label
    _settings = {
        "modifier": (4, "product_nm", "Nutrition modifier"),
        "source": (0, "Forrester", "Literature source"),
        "species": (5, "Picea_abies", "Species"),
        "inputs": (1, "N_S", "Modifier inputs"),
        "selection": (3, "all_plots", "Plot selection"),
    }

    def _setting(run_dir: Path, name: str) -> str:
        """Value of a setting of a run folder."""
        return run_dir.relative_to(_cluster_folder).parts[_settings[name][0]]

    def setting_selectors(*names: str) -> mo.ui.dictionary:
        """Dropdowns of the given settings of the sweep, at their defaults."""
        return mo.ui.dictionary(
            {
                name: mo.ui.dropdown(
                    sorted({_setting(run_dir, name) for run_dir in _run_dirs}),
                    value=_settings[name][1],
                    label=_settings[name][2],
                )
                for name in names
            }
        )

    def matching_runs(settings: dict[str, str]) -> list[Path]:
        """Run folders with the given settings."""
        return [
            run_dir
            for run_dir in _run_dirs
            if all(_setting(run_dir, name) == value for name, value in settings.items())
        ]

    def read_folds(run_dirs: list[Path]) -> pl.DataFrame:
        """RMSE of each plot and fold of the runs, see `leave_one_plot_out`."""
        return pl.concat(
            [
                pl.read_csv(path).with_columns(modifier=pl.lit(_setting(run_dir, "modifier")))
                for run_dir in run_dirs
                for path in run_dir.glob("l2=*/loocv.csv")
            ],
            how="diagonal_relaxed",
        ).with_columns(pl.col("split").str.split(" ").list.first().alias("set"))

    def read_constant_folds(settings: dict[str, str]) -> pl.DataFrame | None:
        """Folds of `constant_nm` with the source, species and plot selection of `settings`.

        `constant_nm` ignores its inputs, so the runs of any inputs fit. None if no fold
        has finished.
        """
        runs = [
            run_dir
            for run_dir in matching_runs(
                {name: settings[name] for name in ("source", "species", "selection")}
                | {"modifier": "constant_nm"}
            )
            if any(run_dir.glob("l2=*/loocv.csv"))
        ]
        return read_folds(runs) if runs else None

    def assign_plot_colors(plots: list[str]) -> dict[str, tuple[float, ...]]:
        """Color of each plot, dark tab20 shades first so the first ten differ in hue."""
        tab20 = plt.get_cmap("tab20").colors
        return dict(zip(sorted(plots), tab20[::2] + tab20[1::2], strict=False))

    return (
        assign_plot_colors,
        matching_runs,
        read_constant_folds,
        read_folds,
        setting_selectors,
    )


@app.cell
def _(mo, setting_selectors):
    deposition_settings = setting_selectors("source", "species", "selection")
    mo.output.replace(
        mo.vstack([mo.md("## Deposition"), deposition_settings.hstack(justify="start")])
    )
    return (deposition_settings,)


@app.cell
def _(
    DEPOSITION_CHANNELS,
    deposition_settings,
    json,
    matching_runs,
    mo,
    months_to_dates,
    np,
    pl,
    prepare_data,
    prepare_plot_input,
):
    # Plots of the settings, the same for all modifiers, inputs and L2 penalties
    _runs = matching_runs(deposition_settings.value)
    mo.stop(not _runs, mo.md("No run with these settings"))
    _job = json.loads(next(_runs[0].glob("l2=*/job.json")).read_text())
    # Monthly deposition of each plot, from its 3PG input file
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
                    "date": months_to_dates(
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
def _(assign_plot_colors, deposition, deposition_plots, mo, pl, plt):
    _colors = assign_plot_colors(deposition["held_out"].unique().to_list())
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
                color=_colors[_plot],
                label=str(_plot).removesuffix("_data"),
            )
    for _ax, _name in zip(_axes, ("N", "S"), strict=True):
        _ax.set_ylabel(f"{_name} deposition (kg ha⁻¹ month⁻¹)")
    _axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="small", title="Plot")
    _axes[0].set_title("Monthly deposition of plots")
    mo.output.replace(deposition_fig)
    return


@app.cell
def _(mo, setting_selectors):
    comparison_settings = setting_selectors("source", "species", "inputs", "selection")
    mo.output.replace(
        mo.vstack([mo.md("## Modifier comparison"), comparison_settings.hstack(justify="start")])
    )
    return (comparison_settings,)


@app.cell
def _(comparison_settings, matching_runs, mo, read_constant_folds, read_folds):
    # Folds of every modifier run with the settings, constant_nm being a baseline
    _runs = [
        run_dir
        for run_dir in matching_runs(comparison_settings.value)
        if run_dir.parent.name != "constant_nm"
    ]
    mo.stop(not _runs, mo.md("No run with these settings"))
    comparison_folds = read_folds(_runs)
    comparison_constant_folds = read_constant_folds(comparison_settings.value)
    comparison_variable = mo.ui.dropdown(
        comparison_folds["variable"].unique(maintain_order=True).to_list(),
        value="DBH",
        label="Variable",
    )
    _baselines = ["without modifier"] + (
        ["constant_nm"] if comparison_constant_folds is not None else []
    )
    comparison_baselines = mo.ui.multiselect(_baselines, value=_baselines, label="Compare with")
    mo.output.replace(mo.hstack([comparison_variable, comparison_baselines], justify="start"))
    return (
        comparison_baselines,
        comparison_constant_folds,
        comparison_folds,
        comparison_variable,
    )


@app.cell
def _(
    comparison_baselines,
    comparison_constant_folds,
    comparison_folds,
    comparison_settings,
    comparison_variable,
    mo,
    pl,
    plt,
):
    def _mean_held_out_rmse(folds: pl.DataFrame) -> pl.DataFrame:
        """Held-out RMSE of each modifier and L2 penalty, averaged over the held-out plots."""
        return (
            folds.filter(pl.col("variable") == comparison_variable.value, pl.col("set") == "test")
            .group_by("modifier", "modifier_l2")
            .agg(pl.col("rmse_fitted", "rmse_phys").mean())
            .sort("modifier", "modifier_l2")
        )

    _rmse = _mean_held_out_rmse(comparison_folds)
    _constant = (
        _mean_held_out_rmse(comparison_constant_folds)
        if "constant_nm" in comparison_baselines.value and comparison_constant_folds is not None
        else None
    )
    # L2 values on evenly spaced positions, since they span five orders of magnitude
    _l2_values = sorted(
        set(_rmse["modifier_l2"]) | (set() if _constant is None else set(_constant["modifier_l2"]))
    )
    _position = {value: index for index, value in enumerate(_l2_values)}
    modifier_comparison_fig, _ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for (_modifier,), _rows in _rmse.group_by("modifier", maintain_order=True):
        _ax.plot(
            [_position[value] for value in _rows["modifier_l2"]],
            _rows["rmse_fitted"],
            marker="o",
            label=_modifier,
        )
    if _constant is not None:
        # A modifier that ignores deposition, so the deposition-dependent modifiers can
        # be told apart from a constant change of alpha_c
        _ax.plot(
            [_position[value] for value in _constant["modifier_l2"]],
            _constant["rmse_fitted"],
            color="gray",
            linestyle="--",
            marker="s",
            label="constant_nm",
        )
    if "without modifier" in comparison_baselines.value:
        # Without the modifier the RMSE depends neither on the modifier nor on the L2 penalty
        _ax.axhline(
            _rmse["rmse_phys"].mean(),
            color="black",
            linestyle="--",
            label="without modifier",
        )
    _ax.set_xticks(range(len(_l2_values)), [f"{value:g}" for value in _l2_values], rotation=45)
    _ax.set_xlabel("L2 penalty (modifier_l2)")
    _ax.set_ylabel(f"{comparison_variable.value} RMSE, held-out plots")
    _ax.set_title(
        f"Held-out {comparison_variable.value} RMSE of each modifier vs L2 penalty\n"
        + ", ".join(comparison_settings.value.values())
    )
    _ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="small")
    mo.output.replace(modifier_comparison_fig)
    return


@app.cell
def _(mo, setting_selectors):
    per_plot_settings = setting_selectors("modifier", "source", "species", "inputs", "selection")
    mo.output.replace(
        mo.vstack([mo.md("## Plot by plot"), per_plot_settings.hstack(justify="start")])
    )
    return (per_plot_settings,)


@app.cell(hide_code=True)
def _(
    matching_runs,
    mo,
    per_plot_settings,
    pl,
    read_constant_folds,
    read_folds,
):
    _runs = matching_runs(per_plot_settings.value)
    mo.stop(not _runs, mo.md("No run with these settings"))
    per_plot_folds = read_folds(_runs)
    per_plot_constant_folds = read_constant_folds(per_plot_settings.value)
    per_plot_reference = mo.ui.dropdown(
        ["without modifier"]
        + (["constant_nm (best L2)"] if per_plot_constant_folds is not None else []),
        value="without modifier",
        label="Compared with",
    )
    per_plot_variable = mo.ui.dropdown(
        per_plot_folds["variable"].unique(maintain_order=True).to_list(),
        value="DBH",
        label="Variable",
    )
    # Default to the L2 penalty with the lowest mean held-out DBH RMSE
    _best_l2 = (
        per_plot_folds.filter(pl.col("variable") == "DBH", pl.col("set") == "test")
        .group_by("modifier_l2")
        .agg(pl.col("rmse_fitted").mean())
        .sort("rmse_fitted")["modifier_l2"]
        .first()
    )
    per_plot_l2 = mo.ui.dropdown(
        {f"{value:g}": value for value in sorted(per_plot_folds["modifier_l2"].unique())},
        value=f"{_best_l2:g}",
        label="L2 penalty",
    )
    mo.output.replace(
        mo.hstack([per_plot_variable, per_plot_l2, per_plot_reference], justify="start")
    )
    return (
        per_plot_constant_folds,
        per_plot_folds,
        per_plot_l2,
        per_plot_reference,
        per_plot_variable,
    )


@app.cell(hide_code=True)
def _(
    mo,
    per_plot_constant_folds,
    per_plot_folds,
    per_plot_l2,
    per_plot_reference,
    per_plot_settings,
    per_plot_variable,
    pl,
    plt,
):
    def _held_out(folds: pl.DataFrame) -> pl.DataFrame:
        """Held-out rows of the variable, one per plot and L2 penalty."""
        return folds.filter(pl.col("variable") == per_plot_variable.value, pl.col("set") == "test")

    _per_plot = _held_out(per_plot_folds).filter(pl.col("modifier_l2") == per_plot_l2.value)
    if per_plot_reference.value == "without modifier" or per_plot_constant_folds is None:
        _reference_label = "without modifier"
        _per_plot = _per_plot.with_columns(reference=pl.col("rmse_phys"))
    else:
        # constant_nm at the L2 penalty with its lowest mean held-out RMSE, since its L2
        # penalties differ from the other modifiers'
        _constant = _held_out(per_plot_constant_folds)
        _constant_l2 = (
            _constant.group_by("modifier_l2")
            .agg(pl.col("rmse_fitted").mean())
            .sort("rmse_fitted")["modifier_l2"]
            .first()
        )
        _reference_label = f"constant_nm (L2 = {_constant_l2:g})"
        _per_plot = _per_plot.join(
            _constant.filter(pl.col("modifier_l2") == _constant_l2).select(
                "plot", reference=pl.col("rmse_fitted")
            ),
            on="plot",
        )
    # Held-out RMSE improvement of each plot, by the fold that held it out
    _per_plot = _per_plot.with_columns(
        (100 * (pl.col("reference") - pl.col("rmse_fitted")) / pl.col("reference")).alias(
            "improvement"
        )
    ).sort("improvement")
    per_plot_fig, _ax = plt.subplots(
        figsize=(8, 0.35 * _per_plot.height + 1.5), layout="constrained"
    )
    _ax.barh(
        _per_plot["plot"].str.strip_suffix("_data"),
        _per_plot["improvement"],
        color=["tab:green" if value > 0 else "tab:red" for value in _per_plot["improvement"]],
    )
    _ax.axvline(0, color="black", linewidth=0.8)
    _ax.set_xlabel(
        f"{per_plot_variable.value} RMSE improvement over {_reference_label} (%), "
        "green better, red worse"
    )
    _improved = (_per_plot["improvement"] > 0).sum()
    _ax.set_title(
        f"Held-out {per_plot_variable.value} RMSE improvement per plot, "
        f"L2 = {per_plot_l2.value:g}: {_improved} of {_per_plot.height} plots improve\n"
        + ", ".join(per_plot_settings.value.values())
    )
    mo.output.replace(per_plot_fig)
    return


@app.cell
def _(mo, setting_selectors):
    run_settings = setting_selectors("modifier", "source", "species", "inputs", "selection")
    mo.output.replace(mo.vstack([mo.md("## One run"), run_settings.hstack(justify="start")]))
    return (run_settings,)


@app.cell(hide_code=True)
def _(matching_runs, mo, pl, run_settings):
    _runs = matching_runs(run_settings.value)
    mo.stop(not _runs, mo.md("No run with these settings"))
    run_dir = _runs[0]
    # Fold-averaged RMSE of each L2 penalty, see loocv_nutrition_modifier.summarize_folds
    summary = pl.concat(
        [pl.read_csv(path) for path in run_dir.glob("l2=*/loocv_summary.csv")],
        how="diagonal_relaxed",
    ).with_columns(pl.col("split").str.split(" ").list.first().alias("set"))
    # RMSE of each plot and fold, see loocv_nutrition_modifier.leave_one_plot_out
    folds = pl.concat(
        [pl.read_csv(path) for path in run_dir.glob("l2=*/loocv.csv")],
        how="diagonal_relaxed",
    ).with_columns(pl.col("split").str.split(" ").list.first().alias("set"))
    return folds, run_dir, summary


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(
    assign_plot_colors,
    folds,
    mo,
    pl,
    plt,
    rmse_view,
    selected_folds,
    selected_summary,
    sets,
    variable,
):
    # Each held-out plot keeps its color whichever plots are selected
    _colors = assign_plot_colors(folds["held_out"].unique().to_list())
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
                    color=_colors[_plot],
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


@app.cell(hide_code=True)
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
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(mo, setting_selectors):
    weight_settings = setting_selectors("modifier", "source", "species", "inputs", "selection")
    mo.output.replace(
        mo.vstack(
            [mo.md("## Weights of the held-out folds"), weight_settings.hstack(justify="start")]
        )
    )
    return (weight_settings,)


@app.cell(hide_code=True)
def _(matching_runs, mo, pl, read_folds, weight_settings):
    _runs = matching_runs(weight_settings.value)
    mo.stop(not _runs, mo.md("No run with these settings"))
    weight_folds = read_folds(_runs)
    mo.stop(
        weight_settings.value["modifier"] == "mlp_nm",
        mo.md("`mlp_nm` weights are not saved in `loocv.csv`, see `modifier_weights`"),
    )
    # Default to the L2 penalty with the lowest mean held-out DBH RMSE
    _best_l2 = (
        weight_folds.filter(pl.col("variable") == "DBH", pl.col("set") == "test")
        .group_by("modifier_l2")
        .agg(pl.col("rmse_fitted").mean())
        .sort("rmse_fitted")["modifier_l2"]
        .first()
    )
    weight_l2 = mo.ui.dropdown(
        {f"{value:g}": value for value in sorted(weight_folds["modifier_l2"].unique())},
        value=f"{_best_l2:g}",
        label="L2 penalty",
    )
    mo.output.replace(weight_l2)
    return weight_folds, weight_l2


@app.cell(hide_code=True)
def _(
    assign_plot_colors,
    mo,
    np,
    pl,
    plt,
    weight_folds,
    weight_l2,
    weight_settings,
):
    _inputs = tuple(weight_settings.value["inputs"].split("_"))
    _modifier = weight_settings.value["modifier"]

    def _label(column: str) -> str:
        """Weight column of `loocv.csv`, e.g. `w01`, with what it multiplies or sets."""
        index = tuple(int(digit) for digit in column[1:])
        if _modifier in ("product_nm", "liebig_nm"):
            # One row (b_k, w_k, raw a_k) per input, see `extended_helper.per_nutrient_nm`
            return f"{('b', 'w', 'raw a')[index[1]]}_{_inputs[index[0]]}"
        if _modifier == "bivariate_quadratic_nm":
            first, second = _inputs
            names = (
                "b",
                f"w_{first}",
                f"w_{second}",
                f"raw a_{first}",
                f"raw a_{second}",
                "raw ρ",
            )
            return names[index[0]]
        # poly_nm: the power of each input in the term, see `extended_helper._cross_term_poly`
        term = "·".join(
            name if power == 1 else f"{name}^{power}"
            for name, power in zip(_inputs, index, strict=True)
            if power
        )
        return f"{column} ({term or 'const'})"

    _columns = [
        column for column in weight_folds.columns if column[0] == "w" and column[1:].isdigit()
    ]
    # One row per fold, the weights being the same in all rows of a fold
    _weights = (
        weight_folds.filter(pl.col("modifier_l2") == weight_l2.value)
        .unique("held_out", keep="first")
        .sort("held_out")
    )
    _colors = assign_plot_colors(_weights["held_out"].to_list())
    # Folds side by side within each weight, so their points do not overlap
    _offsets = np.linspace(-0.3, 0.3, _weights.height)
    weights_fig, _ax = plt.subplots(
        figsize=(max(6, 1.3 * len(_columns)) + 2, 4.5), layout="constrained"
    )
    for _offset, _fold in zip(_offsets, _weights.iter_rows(named=True), strict=True):
        _ax.scatter(
            np.arange(len(_columns)) + _offset,
            [_fold[column] for column in _columns],
            s=22,
            color=_colors[_fold["held_out"]],
            label=_fold["held_out"].removesuffix("_data"),
        )
    # Mean across the folds
    _ax.scatter(
        range(len(_columns)),
        _weights.select(_columns).mean().row(0),
        marker="_",
        s=900,
        color="black",
        label="mean",
    )
    _ax.axhline(0, color="gray", linewidth=0.8)
    _ax.set_xticks(range(len(_columns)), [_label(column) for column in _columns])
    _ax.set_ylabel("Fitted weight")
    _ax.set_title(
        f"Weights of each held-out fold, L2 = {weight_l2.value:g}\n"
        + ", ".join(weight_settings.value.values())
    )
    _ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="small", title="Held-out plot")
    mo.output.replace(weights_fig)
    return


@app.cell
def _():
    return


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
def _(mo, setting_selectors):
    figure_settings = setting_selectors("modifier", "source", "species", "inputs", "selection")
    mo.output.replace(
        mo.vstack([mo.md("## Figures of a run"), figure_settings.hstack(justify="start")])
    )
    return (figure_settings,)


@app.cell(hide_code=True)
def _(figure_settings, matching_runs, mo, pl):
    _runs = matching_runs(figure_settings.value)
    mo.stop(not _runs, mo.md("No run with these settings"))
    # Folder of each L2 penalty, keyed by its value, e.g. `l2=1` and `l2=1.0` alike
    l2_dirs = {float(path.name.removeprefix("l2=")): path for path in _runs[0].glob("l2=*")}
    # Default to the L2 penalty with the lowest held-out DBH RMSE, see
    # loocv_nutrition_modifier.summarize_folds
    _best_l2 = (
        pl.concat(
            [pl.read_csv(path) for path in _runs[0].glob("l2=*/loocv_summary.csv")],
            how="diagonal_relaxed",
        )
        .filter(pl.col("variable") == "DBH", pl.col("split").str.starts_with("test"))
        .sort("rmse_fitted_mean")["modifier_l2"]
        .first()
    )
    l2 = mo.ui.dropdown(
        {f"{value:g}": value for value in sorted(l2_dirs)},
        value=f"{_best_l2:g}",
        label="L2 penalty",
    )
    mo.output.replace(l2)
    return l2, l2_dirs


@app.cell(hide_code=True)
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


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(json, prepare_plot_input, run_dir):
    # Plots and settings of the run, the same for all its L2 penalties
    job = json.loads(next(run_dir.glob("l2=*/job.json")).read_text())
    file_paths = {
        plot_id: prepare_plot_input(plot_id, literature_source=job["literature_source"])
        for plot_id in job["plot_ids"]
    }
    return file_paths, job


@app.cell
def _(
    bivariate_quadratic_nm,
    default_config,
    file_paths,
    job,
    liebig_nm,
    mlp_nm,
    poly_nm,
    product_nm,
    scenario_modifier,
):
    # Configuration of the run with the chosen modifier, see scripts/run_loocv_sweep.py
    _modifiers = {
        "poly_nm": poly_nm,
        "mlp_nm": mlp_nm,
        "product_nm": product_nm,
        "liebig_nm": liebig_nm,
        "bivariate_quadratic_nm": bivariate_quadratic_nm,
    }
    scenario_config = default_config(
        list(file_paths.values()),
        tuple(job["input_vars"]),
        _modifiers[scenario_modifier.value],
    )
    return (scenario_config,)


@app.cell(hide_code=True)
def _(
    ExtendedParams,
    initial_dbh,
    jnp,
    matched_initial_state,
    months_to_dates,
    np,
    prepare_data,
    run_3pg,
    scenario_config,
):
    def simulate(
        file_path: str,
        modifier_params: np.ndarray | None,
        input_stats: tuple[jnp.ndarray, jnp.ndarray] | None,
        n_factor: float,
        s_factor: float,
    ) -> dict[str, np.ndarray]:
        """Monthly DBH, alpha_c and NPP of a plot, with its N and S deposition scaled.

        Parameters
        ----------
        file_path : str
            3PG input file of the plot.
        modifier_params : np.ndarray | None
            Weights of the nutrition modifier, None to run without it.
        input_stats : tuple[jnp.ndarray, jnp.ndarray] | None
            Mean and standard deviation the modifier inputs are z-scored with, None
            without the modifier.
        n_factor, s_factor : float
            Factors of all N and all S deposition channels.

        Returns
        -------
        dict[str, np.ndarray]
            `date` of each month and the simulated `DBH`, `alpha_c` and `NPP`, from the
            DBH-matched start.
        """
        input_data = prepare_data(file_path)
        config = scenario_config
        start_dbh = initial_dbh(file_path, input_data.site, config.observed_sheet)
        initial_state = matched_initial_state(
            input_data.initial_state, input_data.params, start_dbh, config.species_index
        )
        deposition = input_data.deposition._replace(
            **{
                field: getattr(input_data.deposition, field)
                * (n_factor if "_n_" in field else s_factor)
                for field in input_data.deposition._fields
            }
        )
        modifier_args = (
            ()
            if modifier_params is None or input_stats is None
            else (
                deposition,
                ExtendedParams(
                    modifier_params=jnp.asarray(modifier_params),
                    input_mean=input_stats[0],
                    input_std=input_stats[1],
                ),
                config.modifier_fn,
                config.input_vars,
            )
        )
        _, outputs = run_3pg(
            initial_state,
            input_data.climate,
            input_data.params,
            input_data.site,
            input_data.species,
            *modifier_args,
        )
        series = {
            name: np.asarray(outputs[name][:, config.species_index])
            for name in ("DBH", "alpha_c", "NPP")
        }
        start = (int(input_data.site.year_i[0]), int(input_data.site.month_i[0]))
        dates = months_to_dates(*start, len(series["DBH"])).astype("datetime64[D]")
        return {"date": dates, **series}

    return (simulate,)


@app.cell
def _(matching_runs, mo, run_settings):
    # Modifiers run with the other settings of the run
    _settings = {name: value for name, value in run_settings.value.items() if name != "modifier"}
    scenario_modifier = mo.ui.dropdown(
        sorted(run_dir.parent.name for run_dir in matching_runs(_settings)),
        value=run_settings.value["modifier"],
        label="Nutrition modifier",
    )
    mo.output.replace(mo.vstack([mo.md("### Deposition scenarios of the run"), scenario_modifier]))
    return (scenario_modifier,)


@app.cell
def _(job, matching_runs, mo, pl, read_folds, run_settings, scenario_modifier):
    # Folds of the run with the chosen modifier
    scenario_folds = read_folds(
        matching_runs({**run_settings.value, "modifier": scenario_modifier.value})
    )
    # Default to the L2 penalty with the lowest mean held-out DBH RMSE
    _best_l2 = (
        scenario_folds.filter(pl.col("variable") == "DBH", pl.col("set") == "test")
        .group_by("modifier_l2")
        .agg(pl.col("rmse_fitted").mean())
        .sort("rmse_fitted")["modifier_l2"]
        .first()
    )
    scenario_plots = mo.ui.multiselect(job["plot_ids"], value=job["plot_ids"][:1], label="Plots")
    scenario_l2 = mo.ui.dropdown(
        {str(value): value for value in sorted(scenario_folds["modifier_l2"].unique())},
        value=str(_best_l2),
        label="L2 penalty",
    )
    mo.output.replace(mo.hstack([scenario_plots, scenario_l2], justify="start"))
    return scenario_folds, scenario_l2, scenario_plots


@app.cell
def _(mo):
    scenario_mode = mo.ui.radio(
        ["Same for all plots", "Each plot separately"],
        value="Same for all plots",
        label="Deposition change",
        inline=True,
    )
    mo.output.replace(scenario_mode)
    return (scenario_mode,)


@app.cell
def _(mo, scenario_mode, scenario_plots):
    # One N and one S slider for all plots, or for each selected plot
    _groups = (
        ["all plots"] if scenario_mode.value == "Same for all plots" else scenario_plots.value
    )
    deposition_changes = mo.ui.dictionary(
        {
            group: mo.ui.dictionary(
                {
                    name: mo.ui.slider(
                        -200,
                        300,
                        step=5,
                        value=0,
                        label=f"{name} deposition change (%)",
                        show_value=True,
                        debounce=True,
                    )
                    for name in ("N", "S")
                }
            )
            for group in _groups
        }
    )
    mo.output.replace(
        mo.vstack(
            [
                mo.hstack(
                    [mo.md(f"**{group}**"), sliders["N"], sliders["S"]],
                    justify="start",
                    align="center",
                )
                for group, sliders in deposition_changes.items()
            ]
        )
    )
    return (deposition_changes,)


@app.cell(hide_code=True)
def _(
    file_paths,
    input_statistics,
    mlp_nm,
    mo,
    neutral_modifier_params,
    np,
    pl,
    scenario_config,
    scenario_folds,
    scenario_l2,
    scenario_plots,
    simulate,
):
    mo.stop(
        scenario_config.modifier_fn is mlp_nm,
        mo.md("`mlp_nm` weights are not saved in `loocv.csv`, see `modifier_weights`"),
    )
    mo.stop(not scenario_plots.value, mo.md("Select at least one plot"))
    scenario_weights = {}
    scenario_stats = {}
    runs_without_modifier = {}
    runs_with_modifier = {}
    for _plot_id in scenario_plots.value:
        # Weights of the fold that held the plot out, so the plot is out of sample
        _row = scenario_folds.filter(
            pl.col("held_out") == f"{_plot_id}_data",
            pl.col("modifier_l2") == scenario_l2.value,
        ).row(0, named=True)
        _weights = np.asarray(
            neutral_modifier_params(scenario_config.modifier_fn, scenario_config.input_vars)
        ).copy()
        for _index in np.ndindex(_weights.shape):
            _weights[_index] = _row["w" + "".join(map(str, _index))]
        scenario_weights[_plot_id] = _weights
        # Input statistics of all ICP plots, as in `leave_one_plot_out`
        scenario_stats[_plot_id] = input_statistics(tuple(scenario_config.input_vars))
        # Deposition has no effect without the modifier, so this run needs no scaling
        runs_without_modifier[_plot_id] = simulate(file_paths[_plot_id], None, None, 1.0, 1.0)
        runs_with_modifier[_plot_id] = simulate(
            file_paths[_plot_id], _weights, scenario_stats[_plot_id], 1.0, 1.0
        )
    return (
        runs_with_modifier,
        runs_without_modifier,
        scenario_stats,
        scenario_weights,
    )


@app.cell(hide_code=True)
def _(
    deposition_changes,
    file_paths,
    scenario_stats,
    scenario_weights,
    simulate,
):
    # N and S deposition change (%) of each plot, its own or the one of all plots
    scenario_changes = {
        plot_id: deposition_changes.value.get(plot_id, deposition_changes.value.get("all plots"))
        for plot_id in scenario_weights
    }
    # Runs with the modifier and the scaled deposition, rerun when a slider moves
    runs_scaled = {
        plot_id: simulate(
            file_paths[plot_id],
            weights,
            scenario_stats[plot_id],
            1 + scenario_changes[plot_id]["N"] / 100,
            1 + scenario_changes[plot_id]["S"] / 100,
        )
        for plot_id, weights in scenario_weights.items()
    }
    return runs_scaled, scenario_changes


@app.cell(hide_code=True)
def _(
    file_paths,
    mo,
    np,
    pl,
    plt,
    poly_nm,
    runs_scaled,
    runs_with_modifier,
    runs_without_modifier,
    scenario_changes,
    scenario_config,
    scenario_l2,
    scenario_weights,
):
    def _rmse(simulated: dict[str, np.ndarray], observed: pl.DataFrame) -> float:
        """RMSE of a simulated DBH series at the observed months."""
        indices = np.searchsorted(
            simulated["date"], observed["date"].to_numpy().astype("datetime64[D]")
        )
        return float(
            np.sqrt(np.mean((simulated["DBH"][indices] - observed["DBH"].to_numpy()) ** 2))
        )

    _rows = []
    # One panel per selected plot
    _ncols = min(len(scenario_weights), 2)
    _nrows = -(-len(scenario_weights) // _ncols)
    scenario_fig, _axes = plt.subplots(
        _nrows, _ncols, figsize=(6 * _ncols, 3.5 * _nrows), squeeze=False, layout="constrained"
    )
    for _ax, (_plot_id, _weights) in zip(_axes.flat, scenario_weights.items(), strict=False):
        _scaled = runs_scaled[_plot_id]
        _observed = (
            pl.read_excel(file_paths[_plot_id], sheet_name=scenario_config.observed_sheet)
            .filter(pl.col("DBH").is_not_null())
            .with_columns(pl.date("year", "month", 1).alias("date"))
        )
        _ax.plot(
            runs_without_modifier[_plot_id]["date"],
            runs_without_modifier[_plot_id]["DBH"],
            color="gray",
            linestyle="--",
            label="without modifier",
        )
        _ax.plot(
            runs_with_modifier[_plot_id]["date"],
            runs_with_modifier[_plot_id]["DBH"],
            color="tab:blue",
            label="with modifier",
        )
        _ax.plot(
            _scaled["date"],
            _scaled["DBH"],
            color="tab:orange",
            label="with modifier, scaled deposition",
        )
        _ax.scatter(_observed["date"], _observed["DBH"], color="black", zorder=3, label="observed")
        # Each weight, for `poly_nm` with the inputs of its polynomial term, e.g. w11
        # multiplies N·S, see `extended_helper._cross_term_poly`
        _weight_terms = []
        for _index, _value in np.ndenumerate(_weights):
            _name = f"w{''.join(map(str, _index))}"
            if scenario_config.modifier_fn is poly_nm:
                _term = "".join(
                    name
                    for name, power in zip(scenario_config.input_vars, _index, strict=True)
                    if power
                )
                _name += f" ({_term or 'const'})"
            _weight_terms.append(f"{_name} = {_value:.3f}")
        _weight_text = ", ".join(_weight_terms)
        _ax.set_title(f"{_plot_id}, L2 = {scenario_l2.value}\n{_weight_text}", fontsize="medium")
        _ax.set_ylabel("DBH (cm)")

        _n, _s = scenario_changes[_plot_id]["N"], scenario_changes[_plot_id]["S"]
        _row = {
            "plot": _plot_id,
            "N change (%)": _n,
            "S change (%)": _s,
            "RMSE without modifier": _rmse(runs_without_modifier[_plot_id], _observed),
            "RMSE with modifier": _rmse(runs_with_modifier[_plot_id], _observed),
            "RMSE scaled": _rmse(_scaled, _observed),
        }
        # Change of the RMSE with scaled deposition, relative to the unscaled run
        _change = 100 * (_row["RMSE scaled"] / _row["RMSE with modifier"] - 1)
        _verdict = "improved" if _change < 0 else "worse" if _change > 0 else "unchanged"
        _rows.append({**_row, "scaled vs unscaled (%)": _change, "scaling": _verdict})
        _ax.text(
            0.02,
            0.97,
            f"N {_n:+d}%, S {_s:+d}%: RMSE {_row['RMSE with modifier']:.2f} → "
            f"{_row['RMSE scaled']:.2f} cm ({_change:+.1f}%, {_verdict})",
            transform=_ax.transAxes,
            va="top",
            fontsize="small",
            color={"improved": "tab:green", "worse": "tab:red"}.get(_verdict, "black"),
        )
    # Hide the panels left over in the last row
    for _ax in _axes.flat[len(scenario_weights) :]:
        _ax.set_visible(False)
    _handles, _labels = _axes.flat[0].get_legend_handles_labels()
    scenario_fig.legend(_handles, _labels, loc="outside lower center", ncols=4)
    scenario_fig.suptitle(
        f"DBH with the modifier of the fold holding each plot out (L2 = {scenario_l2.value})"
    )

    def _style_cell(_row_id: str, column: str, value: object) -> dict[str, str]:
        """Green for an improved RMSE, red for a worse one, in the comparison columns."""
        if column == "scaled vs unscaled (%)" and isinstance(value, float):
            verdict = "improved" if value < 0 else "worse" if value > 0 else "unchanged"
        elif column == "scaling":
            verdict = str(value)
        else:
            return {}
        color = {"improved": "#1a7f37", "worse": "#cf222e"}.get(verdict)
        return {"color": color, "fontWeight": "bold"} if color else {}

    _table = mo.ui.table(
        pl.DataFrame(_rows).with_columns(pl.selectors.float().round(3)),
        selection=None,
        style_cell=_style_cell,
    )
    mo.output.replace(mo.vstack([scenario_fig, _table]))
    return


@app.cell(hide_code=True)
def _(
    mo,
    pl,
    plt,
    runs_scaled,
    runs_with_modifier,
    runs_without_modifier,
    scenario_changes,
    scenario_l2,
):
    # Mean alpha_c and cumulative NPP over all months, as in `input_sensitivity`
    _outputs = {
        "alpha_c": ("mean", "alpha_c (mol C mol⁻¹ PAR)"),
        "NPP": ("sum", "NPP (t DM ha⁻¹ month⁻¹)"),
    }
    _rows = []
    growth_fig, _axes = plt.subplots(
        len(runs_scaled),
        len(_outputs),
        figsize=(12, 3 * len(runs_scaled)),
        squeeze=False,
        sharex="row",
        layout="constrained",
    )
    for _ax_row, _plot_id in zip(_axes, runs_scaled, strict=True):
        _runs = {
            "without modifier": (runs_without_modifier[_plot_id], "gray", "--"),
            "with modifier": (runs_with_modifier[_plot_id], "tab:blue", "-"),
            "with modifier, scaled deposition": (runs_scaled[_plot_id], "tab:orange", "-"),
        }
        _n, _s = scenario_changes[_plot_id]["N"], scenario_changes[_plot_id]["S"]
        _row = {"plot": _plot_id, "N change (%)": _n, "S change (%)": _s}
        for _ax, (_name, (_stat, _label)) in zip(_ax_row, _outputs.items(), strict=True):
            for _run_label, (_run, _color, _style) in _runs.items():
                _ax.plot(
                    _run["date"],
                    _run[_name],
                    color=_color,
                    linestyle=_style,
                    linewidth=0.8,
                    label=_run_label,
                )
            _unscaled = getattr(runs_with_modifier[_plot_id][_name], _stat)()
            _scaled = getattr(runs_scaled[_plot_id][_name], _stat)()
            # Change with scaled deposition, relative to the unscaled run
            _change = 100 * (_scaled / _unscaled - 1)
            _row |= {
                f"{_stat} {_name} without modifier": getattr(
                    runs_without_modifier[_plot_id][_name], _stat
                )(),
                f"{_stat} {_name} with modifier": _unscaled,
                f"{_stat} {_name} scaled": _scaled,
                f"{_name} scaled vs unscaled (%)": _change,
            }
            _ax.text(
                0.02,
                0.97,
                f"{_stat} {_name}: {_unscaled:.3g} → {_scaled:.3g} ({_change:+.2f}%)",
                transform=_ax.transAxes,
                va="top",
                fontsize="small",
            )
            _ax.set_title(f"{_plot_id}, N {_n:+d}%, S {_s:+d}%", fontsize="medium")
            _ax.set_ylabel(_label)
        _rows.append(_row)
    _handles, _labels = _axes.flat[0].get_legend_handles_labels()
    growth_fig.legend(_handles, _labels, loc="outside lower center", ncols=3)
    growth_fig.suptitle(
        f"alpha_c and NPP with the modifier of the fold holding each plot out "
        f"(L2 = {scenario_l2.value})"
    )
    _table = pl.DataFrame(_rows).with_columns(pl.selectors.float().round(4))
    mo.output.replace(mo.vstack([growth_fig, mo.ui.table(_table, selection=None)]))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(mo, results_data_folder):
    # Folders of the finished PyMC calibrations, local and from the cluster, at
    # <source>/<inputs>/PyMC/[<selection>/]<modifier>, see pymc_nutrition_modifier.py,
    # grouped by the folder holding one run per modifier
    _groups = sorted(
        {
            path.parent.parent
            for folder in ("nutrition_modifier", "cluster")
            for path in (results_data_folder / folder).glob("*/*/PyMC/**/inference_data.nc")
        }
    )
    _heading = mo.md("## PyMC calibration")
    mo.stop(not _groups, mo.vstack([_heading, mo.md("No finished PyMC run")]))
    _options = {str(group.relative_to(results_data_folder)): group for group in _groups}
    pymc_group = mo.ui.dropdown(
        _options,
        value=next(
            (name for name in _options if name.startswith("cluster")), next(iter(_options))
        ),
        label="PyMC runs",
    )
    mo.output.replace(mo.vstack([_heading, pymc_group]))
    return (pymc_group,)


@app.cell
def _(az, mo, pl, plt, pymc_group):
    def _rmse_table(run_summary: str) -> pl.DataFrame:
        """RMSE of the posterior means, from the table of `write_pymc_run_summary`."""
        rows = [
            [cell.strip() for cell in line.strip("│ ").split("┆")]
            for line in run_summary.splitlines()
            if line.startswith(("│ train", "│ test"))
        ]
        return pl.DataFrame(
            rows,
            schema=[
                "split",
                "variable",
                "target",
                "plots",
                "rmse_fitted",
                "rmse_phys",
                "improvement",
            ],
            orient="row",
        ).with_columns(pl.col("plots", "rmse_fitted", "rmse_phys", "improvement").cast(pl.Float64))

    # DBH RMSE of the posterior means and convergence of each modifier's run
    _rows = []
    for _run_dir in sorted(pymc_group.value.glob("*/inference_data.nc")):
        _rmse = _rmse_table((_run_dir.parent / "run_summary.txt").read_text()).filter(
            pl.col("variable") == "DBH"
        )
        _diagnostics = az.summary(az.from_netcdf(_run_dir), kind="diagnostics")
        _row = {"modifier": _run_dir.parent.name}
        for _split in ("train", "test"):
            _split_rmse = _rmse.filter(pl.col("split") == _split).row(0, named=True)
            _row |= {
                f"{_split} plots": int(_split_rmse["plots"]),
                f"{_split} RMSE without modifier": _split_rmse["rmse_phys"],
                f"{_split} RMSE": _split_rmse["rmse_fitted"],
                f"{_split} improvement (%)": _split_rmse["improvement"],
            }
        _rows.append(
            _row
            | {
                "max r_hat": _diagnostics["r_hat"].max(),
                "min ess_bulk": _diagnostics["ess_bulk"].min(),
            }
        )
    pymc_comparison = pl.DataFrame(_rows).sort("test improvement (%)", descending=True)

    pymc_comparison_fig, _ax = plt.subplots(figsize=(9, 4), layout="constrained")
    _x = range(pymc_comparison.height)
    for _offset, _split, _color in ((-0.2, "train", "tab:blue"), (0.2, "test", "tab:orange")):
        _ax.bar(
            [x + _offset for x in _x],
            pymc_comparison[f"{_split} improvement (%)"],
            width=0.4,
            color=_color,
            label=f"{_split} plots",
        )
    _ax.axhline(0, color="black", linewidth=0.8)
    _ax.set_xticks(list(_x), pymc_comparison["modifier"])
    _ax.set_ylabel("DBH RMSE improvement (%)")
    _ax.set_title(
        "DBH RMSE improvement of the posterior-mean modifier vs without modifier\n"
        f"{pymc_group.value.name}, train and test plots of one split"
    )
    _ax.legend()
    mo.output.replace(
        mo.vstack(
            [
                pymc_comparison_fig,
                mo.ui.table(
                    pymc_comparison.with_columns(pl.selectors.float().round(3)), selection=None
                ),
            ]
        )
    )
    return


@app.cell
def _(mo, pymc_group):
    _runs = {
        path.parent.name: path.parent
        for path in sorted(pymc_group.value.glob("*/inference_data.nc"))
    }
    pymc_run = mo.ui.dropdown(
        _runs,
        value="product_nm" if "product_nm" in _runs else next(iter(_runs)),
        label="Nutrition modifier",
    )
    mo.output.replace(pymc_run)
    return (pymc_run,)


@app.cell
def _(mo, pymc_run):
    # Plots, settings and RMSE of the posterior means, see write_pymc_run_summary
    mo.output.replace(mo.md(f"```text\n{(pymc_run.value / 'run_summary.txt').read_text()}\n```"))
    return


@app.cell
def _(az, mo, np, pl, pymc_run):
    # Posterior of the modifier weights and observation errors
    _idata = az.from_netcdf(pymc_run.value / "inference_data.nc")
    _summary = az.summary(_idata).reset_index(names="parameter")
    _axes = az.plot_posterior(
        _idata, figsize=(12, 6), textsize=10, backend_kwargs={"layout": "constrained"}
    )
    pymc_posterior_fig = np.ravel(_axes)[0].figure
    mo.output.replace(
        mo.vstack(
            [
                mo.ui.table(pl.from_pandas(_summary).with_columns(pl.selectors.float().round(3))),
                pymc_posterior_fig,
            ]
        )
    )
    return


@app.cell
def _(mo, pymc_run):
    _images = {path.stem: path for path in sorted(pymc_run.value.glob("images/*.png"))}
    pymc_figure = mo.ui.dropdown(
        _images,
        value="posterior_predictions_test" if "posterior_predictions_test" in _images else None,
        label="Figure",
    )
    mo.output.replace(pymc_figure)
    return (pymc_figure,)


@app.cell
def _(mo, pymc_figure):
    mo.stop(pymc_figure.value is None, mo.md("Select a figure"))
    mo.output.replace(mo.image(pymc_figure.value, width="100%"))
    return


if __name__ == "__main__":
    app.run()
