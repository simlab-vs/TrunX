"""Derivative-based Global Sensitivity Measure (DGSM) analysis on log-likelihood."""

import os
import timeit
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import qmc

from trunx.config import results_data_folder, threepg_data_folder
from trunx.gp3.morris_sensitivity import evaluate_batch_samples, setup_parameters

warnings.filterwarnings("ignore")


def sample_parameter_space(bounds: list[list[float]], n_samples: int, seed: int) -> np.ndarray:
    """Draw a Latin Hypercube sample of parameter values within bounds."""
    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
    unit_sample = sampler.random(n=n_samples)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    return qmc.scale(unit_sample, lower, upper)


def evaluate_batch_jacobian(sample_values: np.ndarray, batched_jacobian_fn) -> np.ndarray:
    """Evaluate the log-likelihood Jacobian for a batch of DGSM samples."""
    batch_results = batched_jacobian_fn(jnp.asarray(sample_values, dtype=jnp.float64))
    return np.asarray(jax.block_until_ready(batch_results))


def run_dgsm_analysis(
    file_path: str,
    observed_data: pd.DataFrame,
    calib_params: list[str],
    output_vars: list[str],
    param_bounds: dict[str, tuple[float, float]],
    sigma_param_names: dict[str, str] | None = None,
    n_samples: int = 1000,
    seed: int = 432,
    species_index: int = 0,
    export_csv: bool = True,
    save_dir: str = "",
    include_individual_components: bool = True,
) -> dict[str, Any]:
    """Run DGSM sensitivity analysis on the combined total log-likelihood.

    Uses JAX automatic differentiation to compute the exact partial
    derivatives of the log-likelihood with respect to each calibration
    parameter at a Latin Hypercube sample of points, then aggregates them
    into the Sobol & Kucherenko (2009) DGSM sensitivity measures.

    Parameters
    ----------
    include_individual_components : bool
        Whether to also run and report DGSM analysis for each `output_vars`
        component individually, in addition to the combined `"total"`
        log-likelihood. `"total"` is always analyzed; set this to False to
        analyze only `"total"` (still computed from every output variable's
        likelihood, just not reported per-variable).
    """
    if not output_vars:
        if sigma_param_names:
            output_vars = [name for name in sigma_param_names if name in observed_data.columns]
        if not output_vars:
            raise ValueError("output_vars cannot be empty.")

    sigma_param_names = sigma_param_names or {}
    setup = setup_parameters(
        file_path,
        observed_data,
        calib_params,
        output_vars,
        param_bounds,
        sigma_param_names,
        seed,
        species_index,
    )

    param_names = setup["all_param_names"]
    bounds = [setup["full_param_bounds"][name] for name in param_names]

    # Generate samples
    print(f"\nGenerating DGSM samples (n={n_samples})...")
    param_values = sample_parameter_space(bounds, n_samples, seed)
    print(f"Generated {param_values.shape[0]} samples")

    df_samples = pl.DataFrame(param_values, schema=param_names)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    df_samples.write_parquet(os.path.join(save_dir, "dgsm_samples.parquet"))

    # Build the batched Jacobian of the log-likelihood components
    jacobian_fn = jax.jacrev(setup["log_likelihood_fn"])
    batched_jacobian_fn = jax.jit(jax.vmap(jacobian_fn, in_axes=0))

    component_names = output_vars + ["total"]

    # Evaluate log-likelihood values and their gradients for all samples
    print("\nEvaluating log-likelihood and its gradient with JAX batching...")
    values = np.zeros((param_values.shape[0], len(component_names)))
    gradients = np.zeros((param_values.shape[0], len(component_names), len(param_names)))

    batch_size = min(10000, param_values.shape[0])
    batch_starts = list(range(0, param_values.shape[0], batch_size))
    print_every = max(1, len(batch_starts) // 10)
    for batch_idx, start in enumerate(batch_starts):
        stop = min(start + batch_size, param_values.shape[0])
        batch = param_values[start:stop]
        values[start:stop] = evaluate_batch_samples(batch, setup["batched_log_likelihood"])
        gradients[start:stop] = evaluate_batch_jacobian(batch, batched_jacobian_fn)

        if batch_idx % print_every == 0:
            print(f"  {start}/{param_values.shape[0]}")

    # Analyze each component (always "total"; individual output_vars only if requested)
    analyzed_component_names = component_names if include_individual_components else ["total"]
    range_sq = np.array([(b[1] - b[0]) ** 2 for b in bounds])
    results: dict[str, dict[str, Any]] = {}
    for component_name in analyzed_component_names:
        comp_idx = component_names.index(component_name)
        comp_values = values[:, comp_idx]
        comp_gradients = gradients[:, comp_idx, :]

        valid_mask = np.isfinite(comp_values) & np.all(np.isfinite(comp_gradients), axis=1)
        n_invalid = int(np.sum(~valid_mask))

        if n_invalid == len(comp_values):
            print(f"\nWARNING: All values for {component_name} are invalid, skipping")
            continue

        variance = np.var(comp_values[valid_mask])
        squared_grad = comp_gradients[valid_mask] ** 2
        nu = squared_grad.mean(axis=0)
        nu_std = squared_grad.std(axis=0)
        dgsm = (
            nu * range_sq / (np.pi**2 * variance)
            if variance > 0
            else np.full(len(param_names), np.nan)
        )

        results[component_name] = {
            "nu": nu,
            "nu_std": nu_std,
            "dgsm": dgsm,
            "names": param_names,
            "n_valid": len(comp_values) - n_invalid,
            "n_total": len(comp_values),
            "n_invalid": n_invalid,
        }

    # Create summary dataframe
    combined_df = pd.DataFrame()
    if results:
        combined_data = []
        for component_name, result in results.items():
            for i, param_name in enumerate(result["names"]):
                combined_data.append(
                    {
                        "Component": component_name,
                        "Parameter": param_name,
                        "dgsm": result["dgsm"][i],
                        "nu": result["nu"][i],
                        "nu_std": result["nu_std"][i],
                    }
                )
        combined_df = pd.DataFrame(combined_data)
        combined_df = combined_df.sort_values(["Component", "dgsm"], ascending=[True, False])

        # Print summary
        print("\nSUMMARY\n")
        for comp in combined_df["Component"].unique():
            print(f"\nCOMPONENT: {comp.upper()}")
            print(combined_df[combined_df["Component"] == comp].head(15))

    # Export results
    if export_csv and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        combined_df.to_csv(f"{save_dir}/dgsm_all_components.csv", index=False)
        print(f"\nResults saved to {save_dir}/dgsm_all_components.csv")

    return {"results": results, "combined_df": combined_df, "param_values": param_values}


if __name__ == "__main__":
    # Load data
    file_path = os.path.join(threepg_data_folder, "solling_data.xlsx")
    observed_df = pd.read_excel(file_path, sheet_name="observed")

    # Get parameter bounds
    param_bounds_df = pd.read_excel(file_path, sheet_name="param_bound")
    error_bounds_df = pd.read_excel(file_path, sheet_name="error_param")
    param_bounds_df = pd.concat([param_bounds_df, error_bounds_df]).reset_index(drop=True)

    param_bounds = {}
    for _, row in param_bounds_df.iterrows():
        if pd.notna(row["min"]) and pd.notna(row["max"]):
            param_bounds[row["param_name"]] = (row["min"], row["max"])

    calib_params = list(param_bounds.keys())
    output_vars = ["DBH", "WS", "WR", "WF", "Height", "BA"]
    sigma_param_names = {var: f"err_{var}" for var in output_vars}

    start_time = timeit.default_timer()
    # Run analysis
    results = run_dgsm_analysis(
        file_path=file_path,
        observed_data=observed_df,
        calib_params=calib_params,
        output_vars=output_vars,
        param_bounds=param_bounds,
        n_samples=5000,
        sigma_param_names=sigma_param_names,
        export_csv=True,
        save_dir=os.path.join(results_data_folder, "dgsm_analysis_results_jax"),
        seed=432,
        include_individual_components=False,
    )
    end_time = timeit.default_timer()
    print(f"\nDGSM sensitivity analysis completed in {end_time - start_time:.2f} seconds.")
