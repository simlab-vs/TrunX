"""Per-plot fit of the gradient-descent nutrition modifier.

The modifier is trained on each plot alone, with the inputs of every plot standardised
with the statistics pooled over all plots, so the fitted weights are comparable across
plots. Their mean and standard deviation show how consistent the modifier is.
"""

from dataclasses import replace

import jax
import polars as pl

from trunx.config import results_data_folder
from trunx.gp3.bayesiancalibrations.pymc_icp_plots import prepare_plot_input
from trunx.gp3.deposition_coverage import selected_plot_ids
from trunx.gp3.loocv_nutrition_modifier import default_config, modifier_weights
from trunx.gp3.nn_nutrition_modifier import (
    NutritionModifierConfig,
    build_predicted_series,
    compute_rmse,
    init_modifier_params,
    input_statistics,
    train_nutrition_modifier,
)


def fit_per_plot(config: NutritionModifierConfig, plot_ids: list[str]) -> pl.DataFrame:
    """Train the modifier on each plot of `config.file_paths` alone.

    Parameters
    ----------
    config : NutritionModifierConfig
        Training configuration; `config.file_paths` are the plots, each fitted on its own,
        and `config.modifier_fn` must be `poly_nm`.
    plot_ids : list[str]
        Plot ids of `config.file_paths`, in the same order.

    Returns
    -------
    pl.DataFrame
        `compute_rmse` rows of the target variables of each plot, with its `plot_id` and
        fitted weights (`w00`, `w01`, ...).
    """
    input_stats = input_statistics(config.file_paths, config.input_vars)
    tables = []
    for plot_id, file_path in zip(plot_ids, config.file_paths, strict=True):
        plot_config = replace(config, file_paths=[file_path], input_stats=input_stats)
        fit_result = train_nutrition_modifier(
            plot_config, initial_modifier_params=init_modifier_params(config.input_vars)
        )
        series = build_predicted_series(plot_config, file_path, fit_result)
        tables.append(
            compute_rmse(plot_config, file_path, series)
            .filter(pl.col("target"))
            .with_columns(
                pl.lit(plot_id).alias("plot_id"),
                *(
                    pl.lit(value).alias(name)
                    for name, value in modifier_weights(fit_result).items()
                ),
            )
        )
        # Each fit compiles new functions; free them, or the process runs out of memory
        # mappings over the plots
        jax.clear_caches()
    return pl.concat(tables)


def summarize_weights(per_plot: pl.DataFrame) -> pl.DataFrame:
    """Mean and standard deviation of each fitted weight across plots.

    Parameters
    ----------
    per_plot : pl.DataFrame
        Table of `fit_per_plot`.

    Returns
    -------
    pl.DataFrame
        One row per weight: `weight`, `mean`, `std` and `plots`.
    """
    weight_cols = [col for col in per_plot.columns if col.startswith("w") and col[1:].isdigit()]
    return (
        per_plot.unique("plot_id")
        .unpivot(weight_cols, variable_name="weight")
        .group_by("weight")
        .agg(
            pl.col("value").mean().alias("mean"),
            pl.col("value").std().alias("std"),
            pl.len().alias("plots"),
        )
        .sort("weight")
    )


if __name__ == "__main__":
    plot_ids = selected_plot_ids()["Picea abies"]
    file_paths = [
        prepare_plot_input(plot_id, literature_source="Trotsiuk") for plot_id in plot_ids
    ]
    config = replace(default_config(file_paths), modifier_l2=0.05)

    per_plot = fit_per_plot(config, plot_ids)
    weight_summary = summarize_weights(per_plot)

    output_dir = results_data_folder / "nn_nutrition_modifier"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_plot.write_csv(output_dir / "per_plot.csv")
    weight_summary.write_csv(output_dir / "per_plot_weights.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3):
        print(per_plot)
        print(weight_summary)
    print(f"Saved {output_dir / 'per_plot.csv'} and {output_dir / 'per_plot_weights.csv'}")
