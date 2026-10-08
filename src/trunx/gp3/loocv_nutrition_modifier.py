"""Leave-one-plot-out cross-validation of the gradient-descent nutrition modifier.

Each plot is held out in turn: the modifier is trained on the other plots (with their own
input standardisation, see `nn_nutrition_modifier.input_scaling`) and evaluated on the
held-out plot, for each L2 penalty in a grid.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from trunx.config import images_folder, results_data_folder
from trunx.gp3.bayesiancalibrations.bayesian_config import species_plot_ids
from trunx.gp3.bayesiancalibrations.pymc_icp_plots import prepare_plot_input
from trunx.gp3.extended_helper import poly_nm
from trunx.gp3.nn_nutrition_modifier import (
    NutritionModifierConfig,
    NutritionModifierFitResult,
    build_predicted_series,
    compute_rmse,
    init_modifier_params,
    summarize_rmse,
    train_nutrition_modifier,
)
from trunx.gp3.plots_nutrition_modifier import (
    plot_learned_modifier_vs_deposition,
    plot_modifier_effect_over_time,
    plot_modifier_effect_vs_deposition,
    plot_modifier_response_surface,
    plot_observed_vs_predicted,
)


def leave_one_plot_out(
    config: NutritionModifierConfig, l2_values: list[float]
) -> tuple[pl.DataFrame, dict[float, list[NutritionModifierFitResult]]]:
    """Train and held-out RMSE of each plot of `config.file_paths`, trained on the other plots.

    Parameters
    ----------
    config : NutritionModifierConfig
        Training configuration; `config.file_paths` are all plots of the cross-validation
        and `config.modifier_fn` must be `poly_nm`.
    l2_values : list[float]
        L2 penalties (`modifier_l2`) to cross-validate.

    Returns
    -------
    pl.DataFrame
        `compute_rmse` rows of each fold's training plots and held-out plot, with the
        fold's `held_out` plot, `modifier_l2`, its fitted weights (`w00`, `w01`, ...) and
        `split` (`"train l2=<value>"` or `"test l2=<value>"`, for `summarize_rmse`).
    dict[float, list[NutritionModifierFitResult]]
        Fit of each fold, per L2 penalty.
    """
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
                fold_config, initial_modifier_params=init_modifier_params(config.input_vars)
            )
            fit_results[l2].append(fit_result)
            weights = {
                f"w{''.join(map(str, index))}": float(value)
                for index, value in np.ndenumerate(np.asarray(fit_result.fitted_modifier_params))
            }
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
            print("Fitted weights (mean ± std across folds)")
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


if __name__ == "__main__":
    plot_ids = species_plot_ids["Picea abies"]
    # plot_ids = ["04.1402", "04.1403", "14.0017", "59.0008"]
    file_paths = [
        prepare_plot_input(plot_id, literature_source="Trotsiuk") for plot_id in plot_ids
    ]
    config = NutritionModifierConfig(
        file_paths=file_paths,
        target_vars=["DBH"],
        plot_variables=["BA", "DBH", "WS", "WF", "WR", "N"],
        # Biomass scaled to the initial stems, as 3-PG is not given the thinning
        observed_sheet="observed",
        input_vars=("N", "S"),
        standardize_inputs=True,
        fit_phys_params=[],
        optimizer_name="adam",
        learning_rate=1e-3,
        num_epochs=5000,
        loss_tol=1e-4,
        modifier_fn=poly_nm,
    )

    loocv_table, fit_results = leave_one_plot_out(config, l2_values=[0.05])
    # loocv_table, fit_results = leave_one_plot_out(
    #    config, l2_values=[0.0, 0.01, 0.05, 0.1, 0.3, 1.0]
    # )

    summary_table = summarize_folds(loocv_table)

    output_dir = results_data_folder / "nn_nutrition_modifier"
    output_dir.mkdir(parents=True, exist_ok=True)
    loocv_table.write_csv(output_dir / "loocv.csv")
    summary_table.write_csv(output_dir / "loocv_summary.csv")
    for l2, l2_fit_results in fit_results.items():
        image_dir = images_folder / "nn_nutrition_modifier" / f"loocv_l2={l2}"
        plot_modifier_response_surface(
            config,
            plot_ids,
            l2_fit_results,
            save_path=str(image_dir / "modifier_response_surface.png"),
            show=False,
        )
        # Each plot predicted by the fold that held it out
        held_out_series = [
            build_predicted_series(config, file_path, fit_result)
            for file_path, fit_result in zip(file_paths, l2_fit_results, strict=True)
        ]
        plot_observed_vs_predicted(
            config,
            plot_ids,
            held_out_series,
            save_path=str(image_dir / "observed_vs_predicted_held_out.png"),
            show=False,
        )
        for var_name in ("alpha_c", "f_nutri_classic_learnable"):
            plot_modifier_effect_over_time(
                config,
                plot_ids,
                held_out_series,
                var_name=var_name,
                save_path=str(image_dir / f"{var_name}_held_out.png"),
                show=False,
            )
        plot_modifier_effect_vs_deposition(
            config,
            held_out_series,
            plot_ids,
            var_name="alpha_c",
            save_path=str(image_dir / "alpha_c_deposition_held_out.png"),
            show=False,
        )
        plot_learned_modifier_vs_deposition(
            config,
            held_out_series,
            plot_ids,
            save_path=str(image_dir / "f_nutri_classic_learnable_deposition_held_out.png"),
            show=False,
        )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3):
        print(
            loocv_table.filter(
                pl.col("variable") == "DBH", pl.col("split").str.starts_with("test")
            )
        )
        print(summarize_rmse(loocv_table))
    print_fold_tables(summary_table)
    print(f"Saved {output_dir / 'loocv.csv'} and {output_dir / 'loocv_summary.csv'}")
