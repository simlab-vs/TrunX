"""Bayesian calibration of the nutrition modifier (`pymc_nutrition_modifier`) on the cluster.

Runs the calibration of `pymc_nutrition_modifier`'s `__main__` from plots chosen
beforehand, as choosing them (`selected_plot_ids`, `split_plot_ids`) reads the raw ICP
deposition, which isn't on the cluster. `--plot-selection` sets the plots: "poorly_fitted"
(the default) keeps those `poorly_fitted_without_modifier` keeps, "all_plots" all of them,
and `--modifier` the modifier function (`poly_nm` by default).

1. `--prepare` (locally, with the raw data) chooses the plots of every plot selection,
   splits them into train and test plots, builds their 3PG input files and writes them to
   `split_file`, shared by the modifiers; sync data/ before submitting.
2. `--plot-selection <selection> --modifier <modifier>` calibrates on the train plots of
   `split_file` and saves to the run folder,
   `nutrition_modifier/<literature source>/<inputs>/PyMC/<plot selection>/<modifier>/`:
   `run_summary.txt`, the sampler results and checkpoints, and `images/`.
"""

import argparse
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import arviz as az
import polars as pl

from trunx.config import results_data_folder
from trunx.gp3.bayesiancalibrations.pymc_icp_plots import prepare_plot_input
from trunx.gp3.extended_helper import (
    bivariate_quadratic_nm,
    liebig_nm,
    mlp_nm,
    poly_nm,
    product_nm,
)
from trunx.gp3.nn_nutrition_modifier import (
    NutritionModifierConfig,
    poorly_fitted_without_modifier,
)
from trunx.gp3.plots_nutrition_modifier import (
    plot_posterior_predictions,
    plot_posterior_residual_vs_deposition,
)
from trunx.gp3.pymc_nutrition_modifier import (
    calibrate_nutrition_modifier_pymc,
    evaluate_posterior_mean,
    predict_posterior_bands,
    write_pymc_run_summary,
)

SPECIES = "Picea abies"
LITERATURE_SOURCE = "Forrester"
CALIBRATED_PARAMS = False  # Use each plot's parameters from the calibration sweep
INPUT_VARS = ("N", "S")
MODIFIERS = {
    "poly_nm": poly_nm,
    "mlp_nm": mlp_nm,
    "product_nm": product_nm,
    "liebig_nm": liebig_nm,
    "bivariate_quadratic_nm": bivariate_quadratic_nm,
}
DEFAULT_MODIFIER = "poly_nm"
# "poorly_fitted": only the plots `poorly_fitted_without_modifier` keeps; "all_plots": all
PLOT_SELECTIONS = ("poorly_fitted", "all_plots")
DEFAULT_PLOT_SELECTION = "poorly_fitted"


def selection_dir(plot_selection: str) -> Path:
    """Folder of `plot_selection`'s runs, as in `pymc_nutrition_modifier`'s `__main__`."""
    return (
        results_data_folder
        / "nutrition_modifier"
        / LITERATURE_SOURCE
        / "_".join(sorted(INPUT_VARS))  # Same folder for ("N", "S") and ("S", "N")
        / "PyMC"
        / plot_selection
    )


def split_file(plot_selection: str) -> Path:
    """Train and test plots of `plot_selection`, shared by the modifiers."""
    return selection_dir(plot_selection) / "plot_split.json"


def run_dir(plot_selection: str, modifier: str) -> Path:
    """Results folder of `plot_selection` and `modifier`."""
    return selection_dir(plot_selection) / modifier


def base_config(
    file_paths: list[str], plot_selection: str, modifier: str = DEFAULT_MODIFIER
) -> NutritionModifierConfig:
    """Calibration settings of `pymc_nutrition_modifier`'s `__main__`."""
    return NutritionModifierConfig(
        file_paths=file_paths,
        target_vars=["DBH"],
        calibrated_params=CALIBRATED_PARAMS,
        plot_variables=["BA", "DBH", "WS", "WF", "WR"],
        input_vars=INPUT_VARS,
        matched_start=True,
        fit_phys_params=[],
        modifier_fn=MODIFIERS[modifier],
        image_dir=str(run_dir(plot_selection, modifier) / "images"),
    )


def prepare() -> None:
    """Choose and split the plots of every plot selection into `split_file`."""
    from trunx.gp3.deposition_coverage import selected_plot_ids
    from trunx.gp3.nn_nutrition_modifier import split_plot_ids

    all_plot_ids = selected_plot_ids()[SPECIES]
    all_file_paths = [
        prepare_plot_input(p, literature_source=LITERATURE_SOURCE, calibrated=CALIBRATED_PARAMS)
        for p in all_plot_ids
    ]
    for plot_selection in PLOT_SELECTIONS:
        plot_ids, file_paths = all_plot_ids, all_file_paths
        if plot_selection == "poorly_fitted":
            plot_ids, file_paths = poorly_fitted_without_modifier(
                base_config(file_paths, plot_selection), plot_ids, file_paths, criterion="rmse"
            )
        train_plot_ids, test_plot_ids = split_plot_ids(SPECIES, plot_ids)
        split_path = split_file(plot_selection)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(
            json.dumps({"train": train_plot_ids, "test": test_plot_ids}, indent=2)
        )
        print(f"{plot_selection}: train {train_plot_ids}, test {test_plot_ids} -> {split_path}")


def calibrate(plot_selection: str, modifier: str) -> None:
    """Calibrate `modifier` on the train plots of `plot_selection`."""
    split_path = split_file(plot_selection)
    if not split_path.exists():
        raise FileNotFoundError(f"{split_path} not found; run with --prepare first")
    split = json.loads(split_path.read_text())
    plot_ids, test_plot_ids = split["train"], split["test"]
    file_paths = [
        prepare_plot_input(p, literature_source=LITERATURE_SOURCE, calibrated=CALIBRATED_PARAMS)
        for p in plot_ids
    ]
    test_file_paths = [
        prepare_plot_input(p, literature_source=LITERATURE_SOURCE, calibrated=CALIBRATED_PARAMS)
        for p in test_plot_ids
    ]
    config = base_config(file_paths, plot_selection, modifier)
    output_dir = str(run_dir(plot_selection, modifier))

    # Start from an empty image folder, so no figure of an earlier run is left behind; the
    # checkpoints in `output_dir` are kept, so an interrupted run still resumes
    image_dir = Path(config.image_dir)
    shutil.rmtree(image_dir, ignore_errors=True)
    image_dir.mkdir(parents=True)

    idata, fit_result = calibrate_nutrition_modifier_pymc(
        config,
        num_warmup=1000,
        num_samples=1000,
        chains=4,
        step_method="nuts",
        output_dir=output_dir,
        checkpoint_every=250,
        resume=False,
    )
    print(az.summary(idata))
    rmse_summary = evaluate_posterior_mean(
        config, idata, fit_result, plot_ids, test_file_paths, test_plot_ids
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3):
        print(rmse_summary)
    run_summary_path = run_dir(plot_selection, modifier) / "run_summary.txt"
    write_pymc_run_summary(
        run_summary_path,
        config,
        idata,
        fit_result,
        SPECIES,
        LITERATURE_SOURCE,
        plot_ids,
        test_plot_ids,
        rmse_summary,
    )

    # 95% posterior prediction interval of the training and test plots, and of the
    # modifier's effect on alpha_c and NPP
    for split_name, split_file_paths, split_ids in (
        ("train", file_paths, plot_ids),
        ("test", test_file_paths, test_plot_ids),
    ):
        split_config = replace(config, file_paths=split_file_paths)
        posterior_series = [
            predict_posterior_bands(config, idata, fit_result, file_path)
            for file_path in split_file_paths
        ]
        plot_posterior_predictions(
            split_config,
            split_ids,
            posterior_series,
            save_path=os.path.join(config.image_dir, f"posterior_predictions_{split_name}.png"),
            show=False,
        )
        for var_name in ("alpha_c", "NPP"):
            plot_posterior_residual_vs_deposition(
                split_config,
                split_ids,
                posterior_series,
                var_name=var_name,
                save_path=os.path.join(
                    config.image_dir, f"{var_name}_residual_posterior_{split_name}.png"
                ),
                show=False,
            )
    print(f"Saved results and {run_summary_path} to {output_dir}, figures to {config.image_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prepare", action="store_true", help="choose and split the plots")
    parser.add_argument(
        "--plot-selection",
        choices=PLOT_SELECTIONS,
        default=DEFAULT_PLOT_SELECTION,
        help="plots to calibrate on",
    )
    parser.add_argument(
        "--modifier", choices=list(MODIFIERS), default=DEFAULT_MODIFIER, help="modifier function"
    )
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        calibrate(args.plot_selection, args.modifier)
