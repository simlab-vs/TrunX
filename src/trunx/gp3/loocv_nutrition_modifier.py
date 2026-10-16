"""Leave-one-plot-out cross-validation of the gradient-descent nutrition modifier.

Each plot is held out in turn: the modifier is trained on the other plots and evaluated
on the held-out plot, for each L2 penalty in a grid. Each fold fits its own modifier
weights, so the weights differ for each held-out plot. The modifier inputs of every fold
are standardised alike, with the mean and standard deviation over all plots (see
`nn_nutrition_modifier.input_scaling`).
"""

import shutil
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from trunx.config import images_folder, results_data_folder
from trunx.gp3.bayesiancalibrations.pymc_icp_plots import prepare_plot_input
from trunx.gp3.deposition_coverage import selected_plot_ids
from trunx.gp3.extended_helper import poly_nm
from trunx.gp3.nn_nutrition_modifier import (
    NutritionModifierConfig,
    NutritionModifierFitResult,
    build_predicted_series,
    compute_rmse,
    config_summary_lines,
    input_statistics,
    neutral_modifier_params,
    poorly_fitted_without_modifier,
    summarize_rmse,
    train_nutrition_modifier,
    with_target_observations,
)
from trunx.gp3.plots_nutrition_modifier import (
    plot_learned_modifier_vs_deposition,
    plot_modifier_effect_over_time,
    plot_modifier_effect_vs_deposition,
    plot_modifier_pass_through,
    plot_modifier_residual_vs_deposition,
    plot_modifier_response_curve,
    plot_modifier_response_surface,
    plot_observed_vs_predicted,
)


def modifier_weights(fit_result: NutritionModifierFitResult) -> dict[str, float]:
    """Fitted `poly_nm` weights keyed by their polynomial index, e.g. `w01`.

    Empty for other modifiers, such as `mlp_nm`, whose weights are not one array.
    """
    if not isinstance(fit_result.fitted_modifier_params, jax.Array | np.ndarray):
        return {}
    return {
        f"w{''.join(map(str, index))}": float(value)
        for index, value in np.ndenumerate(np.asarray(fit_result.fitted_modifier_params))
    }


def leave_one_plot_out(
    config: NutritionModifierConfig, l2_values: list[float]
) -> tuple[pl.DataFrame, dict[float, list[NutritionModifierFitResult]]]:
    """Train and held-out RMSE of each plot of `config.file_paths`, trained on the other plots.

    Parameters
    ----------
    config : NutritionModifierConfig
        Training configuration; `config.file_paths` are all plots of the cross-validation.
    l2_values : list[float]
        L2 penalties (`modifier_l2`) to cross-validate.

    Returns
    -------
    pl.DataFrame
        `compute_rmse` rows of each fold's training plots and held-out plot, with the
        fold's `held_out` plot, `modifier_l2`, its fitted `poly_nm` weights (`w00`, `w01`,
        ...) and
        `split` (`"train l2=<value>"` or `"test l2=<value>"`, for `summarize_rmse`).
    dict[float, list[NutritionModifierFitResult]]
        Fit of each fold, per L2 penalty.
    """
    # Standardise the inputs of every fold alike, with the statistics of all plots
    if config.standardize_inputs and config.input_stats is None:
        config = replace(
            config, input_stats=input_statistics(config.file_paths, config.input_vars)
        )
    tables = []
    fit_results: dict[float, list[NutritionModifierFitResult]] = {l2: [] for l2 in l2_values}
    for l2 in l2_values:
        for held_out in config.file_paths:
            fold_config = replace(
                config,
                file_paths=[path for path in config.file_paths if path != held_out],
                modifier_l2=l2,
            )
            fit_result = train_nutrition_modifier(
                fold_config,
                initial_modifier_params=neutral_modifier_params(
                    config.modifier_fn, config.input_vars
                ),
            )
            fit_results[l2].append(fit_result)
            weights = modifier_weights(fit_result)
            for set_name, paths in [("train", fold_config.file_paths), ("test", [held_out])]:
                for path in paths:
                    series = build_predicted_series(fold_config, path, fit_result)
                    tables.append(
                        compute_rmse(fold_config, path, series).with_columns(
                            pl.lit(Path(held_out).stem).alias("held_out"),
                            pl.lit(l2).alias("modifier_l2"),
                            pl.lit(f"{set_name} l2={l2}").alias("split"),
                            *(pl.lit(value).alias(name) for name, value in weights.items()),
                        )
                    )
            # Each fold compiles new functions; free them, or the process runs out of
            # memory mappings over the folds
            jax.clear_caches()
    return pl.concat(tables), fit_results


def summarize_folds(loocv_table: pl.DataFrame) -> pl.DataFrame:
    """Mean and standard deviation of the RMSE and fitted weights across folds.

    Parameters
    ----------
    loocv_table : pl.DataFrame
        Table of `leave_one_plot_out`.

    Returns
    -------
    pl.DataFrame
        One row per split and variable: `rows` (plot-fold pairs averaged over),
        `<rmse>_mean` and `<rmse>_std` of each RMSE column, and `<w>_mean` and `<w>_std`
        of each weight across the folds of the split's L2 penalty.
    """
    rmse_cols = [col for col in loocv_table.columns if col.startswith("rmse_")]
    weight_cols = [col for col in loocv_table.columns if col.startswith("w") and col[1:].isdigit()]
    # One row per fold, so each fold's weights count once
    weight_stats = (
        loocv_table.unique(["modifier_l2", "held_out"])
        .group_by("modifier_l2")
        .agg(
            pl.col(weight_cols).mean().name.suffix("_mean"),
            pl.col(weight_cols).std().name.suffix("_std"),
        )
    )
    return (
        loocv_table.group_by("split", "modifier_l2", "variable", "target", maintain_order=True)
        .agg(
            pl.len().alias("rows"),
            pl.col(rmse_cols).mean().name.suffix("_mean"),
            pl.col(rmse_cols).std().name.suffix("_std"),
        )
        .join(weight_stats, on="modifier_l2", how="left", maintain_order="left")
    )


def select_l2(summary_table: pl.DataFrame) -> pl.DataFrame:
    """Held-out RMSE of each target variable per L2 penalty, flagging the lowest.

    Parameters
    ----------
    summary_table : pl.DataFrame
        Table of `summarize_folds`, over one or more L2 penalties.

    Returns
    -------
    pl.DataFrame
        One row per target variable and L2 penalty: `rmse_fitted_mean` and
        `rmse_fitted_std` across the held-out plots, `rmse_phys_mean` (without the
        modifier) and `selected`, true for the penalty with the lowest `rmse_fitted_mean`.
    """
    return (
        summary_table.filter(pl.col("target"), pl.col("split").str.starts_with("test"))
        .select("variable", "modifier_l2", "rmse_fitted_mean", "rmse_fitted_std", "rmse_phys_mean")
        .sort("variable", "modifier_l2")
        .with_columns(
            (
                pl.col("rmse_fitted_mean") == pl.col("rmse_fitted_mean").min().over("variable")
            ).alias("selected")
        )
    )


def _rmse_table(summary_table: pl.DataFrame, variables: list[str]) -> pl.DataFrame:
    """RMSE with and without the modifier and its improvement, one column per variable and set."""
    columns = {"": ["With modifier", "Without modifier", "Improvement"]}
    for var_name in variables:
        for row in summary_table.filter(pl.col("variable") == var_name).iter_rows(named=True):
            set_name = row["split"].split(" ")[0]
            improvement = (
                100 * (row["rmse_phys_mean"] - row["rmse_fitted_mean"]) / row["rmse_phys_mean"]
            )
            columns[f"{var_name} {set_name}"] = [
                f"{row['rmse_fitted_mean']:.2f} ± {row['rmse_fitted_std']:.2f}",
                f"{row['rmse_phys_mean']:.2f} ± {row['rmse_phys_std']:.2f}",
                f"{improvement:.0f}%",
            ]
    return pl.DataFrame(columns)


def print_fold_tables(summary_table: pl.DataFrame) -> None:
    """Print the RMSE and fitted weight tables (mean ± std across folds), per L2 penalty.

    Parameters
    ----------
    summary_table : pl.DataFrame
        Table of `summarize_folds`.
    """
    weight_names = [
        col.removesuffix("_mean")
        for col in summary_table.columns
        if col.startswith("w") and col.endswith("_mean")
    ]
    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
        fmt_str_lengths=50,
        tbl_width_chars=250,
    ):
        for (l2,), l2_summary in summary_table.group_by("modifier_l2", maintain_order=True):
            print(f"\nmodifier_l2 = {l2}")
            for label, is_target in (("Target", True), ("Other", False)):
                variables = (
                    l2_summary.filter(pl.col("target") == is_target)["variable"]
                    .unique(maintain_order=True)
                    .to_list()
                )
                if variables:
                    print(f"{label} variables, RMSE (mean ± std)")
                    print(_rmse_table(l2_summary, variables))
            weights = l2_summary.row(0, named=True)
            print("Fitted weights, different for each held-out plot (mean ± std across folds)")
            print(
                pl.DataFrame(
                    {
                        "Weight": weight_names,
                        "Value": [
                            f"{weights[f'{name}_mean']:.2f} ± {weights[f'{name}_std']:.2f}"
                            for name in weight_names
                        ],
                    }
                )
            )


def default_config(
    file_paths: list[str],
    input_vars: tuple[str, ...] = ("N", "S"),
    modifier_fn: Callable[[Any, jnp.ndarray, tuple[str, ...]], jnp.ndarray] = poly_nm,
) -> NutritionModifierConfig:
    """Training configuration of the cross-validation over the plots of `file_paths`.

    Parameters
    ----------
    file_paths : list[str]
        3PG input files of all plots of the cross-validation.
    input_vars : tuple[str, ...]
        Deposition inputs of the modifier, `DEPOSITION_CHANNELS` keys.
    modifier_fn : Callable
        Modifier function, `poly_nm` or `mlp_nm`.

    Returns
    -------
    NutritionModifierConfig
        Configuration fitting a `modifier_fn` modifier over `input_vars` to DBH.
    """
    return NutritionModifierConfig(
        file_paths=file_paths,
        target_vars=["DBH"],
        plot_variables=["BA", "DBH", "WS", "WF", "WR", "N"],
        # Biomass scaled to the initial stems, as 3-PG is not given the thinning
        observed_sheet="observed",
        input_vars=input_vars,
        matched_start=True,
        standardize_inputs=True,
        fit_phys_params=[],
        optimizer_name="adam",
        learning_rate=1e-3,
        num_epochs=5000,
        loss_tol=1e-3,
        modifier_fn=modifier_fn,
    )


def write_loocv_run_summary(
    save_path: Path,
    config: NutritionModifierConfig,
    fit_results: dict[float, list[NutritionModifierFitResult]],
    species: str,
    literature_source: str,
    plot_ids: list[str],
    tables_text: str,
) -> None:
    """Write the settings, folds and results of a cross-validation run to a text file.

    Parameters
    ----------
    save_path : Path
        Text file to write.
    config : NutritionModifierConfig
        Training configuration, see `leave_one_plot_out`.
    fit_results : dict[float, list[NutritionModifierFitResult]]
        Fit of each fold, per L2 penalty, see `leave_one_plot_out`.
    species : str
        Species of the plots.
    literature_source : str
        Literature source of the 3PG parameters of the plot inputs.
    plot_ids : list[str]
        Plot ids of `config.file_paths`, in the same order; fold i holds out plot i.
    tables_text : str
        Printed RMSE and weight tables of the run.
    """
    lines = [
        "Nutrition modifier leave-one-plot-out cross-validation",
        f"Date: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "Data",
        f"  species: {species}",
        f"  literature_source: {literature_source}",
        f"  plots ({len(plot_ids)}): {', '.join(plot_ids)}",
        "",
        "Settings",
        f"  modifier_l2 (cross-validated): {list(fit_results)}",
        *config_summary_lines(config, skip=("modifier_l2",)),
        "",
    ]
    for l2, l2_fit_results in fit_results.items():
        weights = pl.DataFrame([modifier_weights(fit_result) for fit_result in l2_fit_results])
        lines += [
            f"Folds, modifier_l2 = {l2}",
            *(
                f"  held out {plot_id}: {len(fit.loss_history)} epochs, "
                f"final loss {fit.loss_history[-1]:.6f}, weights "
                + ", ".join(f"{name} {value:.3f}" for name, value in modifier_weights(fit).items())
                for plot_id, fit in zip(plot_ids, l2_fit_results, strict=True)
            ),
            "  weights (different for each held-out plot; mean ± std across folds): "
            + ", ".join(
                f"{name} {weights[name].mean():.3f} ± {weights[name].std():.3f}"
                for name in weights.columns
            ),
            "",
        ]
    lines += ["Results", tables_text]
    save_path.write_text("\n".join(lines) + "\n")


def run_loocv(
    config: NutritionModifierConfig,
    plot_ids: list[str],
    l2_values: list[float],
    output_dir: Path,
    image_dir: Path,
    species: str,
    literature_source: str,
) -> pl.DataFrame:
    """Cross-validate, save the tables, figures and run summary and print the fold tables.

    Parameters
    ----------
    config : NutritionModifierConfig
        Training configuration, see `leave_one_plot_out`.
    plot_ids : list[str]
        Plot ids of `config.file_paths`, in the same order.
    l2_values : list[float]
        L2 penalties to cross-validate.
    output_dir : Path
        Folder of `loocv.csv`, `loocv_summary.csv`, `l2_selection.csv` (see `select_l2`;
        only with several L2 penalties), the printed tables, `loocv_output.txt`, and
        `run_summary.txt` (see `write_loocv_run_summary`).
    image_dir : Path
        Folder of the figures, one `loocv_l2=<value>` subfolder per L2 penalty, each with a
        copy of `run_summary.txt`.
    species : str
        Species of the plots, for the run summary.
    literature_source : str
        Literature source of the plot inputs, for the run summary.

    Returns
    -------
    pl.DataFrame
        Table of `summarize_folds`.
    """
    # Remove an earlier run's results so none are left behind; only this function's
    # files, as the folders can also hold other scripts' results
    for name in (
        "loocv.csv",
        "loocv_summary.csv",
        "l2_selection.csv",
        "loocv_output.txt",
        "run_summary.txt",
    ):
        (output_dir / name).unlink(missing_ok=True)
    for l2 in l2_values:
        shutil.rmtree(image_dir / f"loocv_l2={l2}", ignore_errors=True)

    loocv_table, fit_results = leave_one_plot_out(config, l2_values=l2_values)
    summary_table = summarize_folds(loocv_table)

    output_dir.mkdir(parents=True, exist_ok=True)
    loocv_table.write_csv(output_dir / "loocv.csv")
    summary_table.write_csv(output_dir / "loocv_summary.csv")
    for l2, l2_fit_results in fit_results.items():
        l2_image_dir = image_dir / f"loocv_l2={l2}"
        if len(config.input_vars) == 1:
            plot_modifier_response_curve(
                config,
                plot_ids,
                l2_fit_results,
                save_path=str(l2_image_dir / "modifier_response_curve.png"),
                show=False,
            )
        else:
            for pre_tanh, name in ((False, "response_surface"), (True, "score_before_tanh")):
                plot_modifier_response_surface(
                    config,
                    plot_ids,
                    l2_fit_results,
                    save_path=str(l2_image_dir / f"modifier_{name}.png"),
                    show=False,
                    pre_tanh=pre_tanh,
                )
        # Each plot predicted by the fold that held it out
        held_out_series = [
            build_predicted_series(config, file_path, fit_result)
            for file_path, fit_result in zip(config.file_paths, l2_fit_results, strict=True)
        ]
        plot_observed_vs_predicted(
            config,
            plot_ids,
            held_out_series,
            save_path=str(l2_image_dir / "observed_vs_predicted_held_out.png"),
            show=False,
        )
        for var_name in ("alpha_c", "f_nutri_classic_learnable", "NPP"):
            plot_modifier_effect_over_time(
                config,
                plot_ids,
                held_out_series,
                var_name=var_name,
                save_path=str(l2_image_dir / f"{var_name}_held_out.png"),
                show=False,
            )
        plot_modifier_pass_through(
            plot_ids,
            held_out_series,
            save_path=str(l2_image_dir / "modifier_pass_through_held_out.png"),
            show=False,
        )
        plot_modifier_effect_vs_deposition(
            config,
            held_out_series,
            plot_ids,
            var_name="alpha_c",
            save_path=str(l2_image_dir / "alpha_c_deposition_held_out.png"),
            show=False,
        )
        plot_modifier_residual_vs_deposition(
            config,
            held_out_series,
            plot_ids,
            var_name="alpha_c",
            save_path=str(l2_image_dir / "alpha_c_residual_deposition_held_out.png"),
            show=False,
            title_note="Different weights for each plot: each plot uses the fold that held it out",
        )
        plot_learned_modifier_vs_deposition(
            config,
            held_out_series,
            plot_ids,
            save_path=str(l2_image_dir / "f_nutri_classic_learnable_deposition_held_out.png"),
            show=False,
            title_note="Different weights for each plot: each plot uses the fold that held it out",
        )
    # Capture the printed tables to also save them next to the CSVs
    with redirect_stdout(StringIO()) as output:
        with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_width_chars=250):
            print(
                loocv_table.filter(
                    pl.col("variable") == "DBH", pl.col("split").str.starts_with("test")
                )
            )
            print(summarize_rmse(loocv_table))
        print_fold_tables(summary_table)
        if len(l2_values) > 1:
            l2_selection = select_l2(summary_table)
            l2_selection.write_csv(output_dir / "l2_selection.csv")
            print("\nL2 selection, held-out RMSE averaged over the folds")
            print(l2_selection)
            for var_name, l2 in (
                l2_selection.filter("selected").select("variable", "modifier_l2").rows()
            ):
                print(f"Selected modifier_l2 for {var_name}: {l2}")
    (output_dir / "loocv_output.txt").write_text(output.getvalue())
    write_loocv_run_summary(
        output_dir / "run_summary.txt",
        config,
        fit_results,
        species,
        literature_source,
        plot_ids,
        output.getvalue(),
    )
    for l2 in l2_values:
        shutil.copy(output_dir / "run_summary.txt", image_dir / f"loocv_l2={l2}")
    print(output.getvalue())
    print(
        f"Saved {output_dir / 'loocv.csv'}, {output_dir / 'loocv_summary.csv'}, "
        f"{output_dir / 'loocv_output.txt'} and {output_dir / 'run_summary.txt'}"
    )
    return summary_table


if __name__ == "__main__":
    species = "Picea abies"
    literature_source = "Forrester"
    plot_ids = selected_plot_ids()[species]

    # plot_ids = ["04.1402", "04.1403", "14.0017", "04.0302"]
    file_paths = [
        prepare_plot_input(plot_id, literature_source=literature_source) for plot_id in plot_ids
    ]
    config = default_config(file_paths, input_vars=("N", "S"))
    # Leave out plots without any target observation
    plot_ids, file_paths = with_target_observations(config, plot_ids, file_paths)
    # Leave out plots the run without the modifier already fits well
    plot_ids, file_paths = poorly_fitted_without_modifier(config, plot_ids, file_paths)
    run_loocv(
        replace(config, file_paths=file_paths),
        plot_ids,
        l2_values=[0.05],
        # l2_values=[0.0, 0.01, 0.05, 0.1, 0.3, 1.0],
        output_dir=results_data_folder / "nn_nutrition_modifier",
        image_dir=images_folder / "nutrition_modifier",
        species=species,
        literature_source=literature_source,
    )
