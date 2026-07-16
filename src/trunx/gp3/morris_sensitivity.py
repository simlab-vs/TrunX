"""Morris sensitivity analysis on log-likelihood."""

import os
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.stats import norm
from SALib.analyze import morris as morris_analyze
from SALib.sample import morris as morris_sample

from trunx.config import results_data_folder, threepg_data_folder
from trunx.gp3.model_inputs import Params
from trunx.gp3.PG3_model_impl import prepare_data
from trunx.gp3.run_3pg import run_3pg as run_3pg_orig

warnings.filterwarnings("ignore")
PLAUSIBILITY_MULTIPLIER = 100.0
run_3pg = jax.jit(run_3pg_orig)


def prepare_observation_times(site_data, observed_data):
    """Prepare time indices for observations."""
    start_year = int(np.asarray(site_data.year_i).reshape(-1)[0])
    start_month = int(np.asarray(site_data.month_i).reshape(-1)[0])
    if not {"year", "month"}.issubset(observed_data.columns):
        raise ValueError("Observed data must contain year and month columns")
    return np.array(
        [
            (int(row["year"]) - start_year) * 12 + (int(row["month"]) - start_month)
            for _, row in observed_data.iterrows()
        ],
        dtype=int,
    )


def component_log_likelihood(model_values, observations, sigma, obs_indices, species_index):
    """Compute the log-likelihood for one output component."""
    sigma = jnp.asarray(sigma, dtype=jnp.float64)

    if len(model_values.shape) == 2:
        model_values = model_values[:, species_index]

    predictions = jnp.asarray(model_values[obs_indices]).reshape(-1)
    observations = jnp.asarray(observations).reshape(-1)

    valid_mask = ~(jnp.isnan(predictions) | jnp.isnan(observations))
    residuals = predictions - observations
    log_terms = jnp.where(valid_mask, norm.logpdf(residuals, loc=0.0, scale=sigma), 0.0)
    valid_count = jnp.sum(valid_mask)
    result = jnp.where(valid_count > 0, jnp.sum(log_terms), -jnp.inf)

    obs_scale = jnp.max(jnp.abs(observations)) + 1e-8
    plausible = jnp.all(jnp.abs(predictions) < PLAUSIBILITY_MULTIPLIER * obs_scale)
    return jnp.where(plausible, result, -jnp.inf)


def log_likelihood_components(
    param_values,
    params,
    initial_state,
    climate,
    site_data,
    species_data,
    obs_indices,
    obs_values,
    output_vars,
    _param_index,
    _sigma_param_set,
    species_index,
):
    """Compute component log-likelihoods for one sample."""
    params_dict = {field: getattr(params, field) for field in params._fields}
    for param_name, i in _param_index.items():
        if param_name not in _sigma_param_set:
            params_dict[param_name] = param_values[i]

    _, model_outputs = run_3pg(
        initial_state=initial_state,
        climate=climate,
        params=Params(**params_dict),
        site=site_data,
        species=species_data,
    )

    component_values = []
    for var_name in output_vars:
        model_values = model_outputs.get(var_name)

        component_values.append(
            component_log_likelihood(
                model_values,
                obs_values[var_name],
                param_values[_param_index[f"err_{var_name}"]],
                obs_indices,
                species_index,
            )
        )

    component_array = jnp.asarray(component_values, dtype=jnp.float64)
    finite_mask = jnp.isfinite(component_array)
    total = jnp.sum(jnp.where(finite_mask, component_array, 0.0))
    total = jnp.where(jnp.any(finite_mask), total, -jnp.inf)
    return jnp.concatenate([component_array, jnp.asarray([total], dtype=jnp.float64)])


def evaluate_batch_samples(sample_values, batched_log_likelihood):
    """Evaluate a batch of Morris samples."""
    batch_results = batched_log_likelihood(jnp.asarray(sample_values, dtype=jnp.float64))
    return np.asarray(jax.block_until_ready(batch_results))


def setup_parameters(
    file_path,
    observed_data,
    calib_params,
    output_vars,
    param_bounds,
    sigma_param_names,
    n_levels,
    n_trajectories,
    seed,
    species_index,
):
    """Set up all parameters and data for analysis."""
    np.random.seed(seed)

    # Load model data
    print("Loading model data...")
    initial_state, climate, params, site_data, species_data, _, _ = prepare_data(file_path)

    # Setup parameters
    _sigma_param_set = set(sigma_param_names.values())
    all_param_names = list(dict.fromkeys(calib_params + list(_sigma_param_set)))
    _param_index = {name: idx for idx, name in enumerate(all_param_names)}

    full_param_bounds = {param: list(param_bounds[param]) for param in calib_params}
    for sigma_param in _sigma_param_set:
        if sigma_param not in param_bounds:
            raise KeyError(f"Missing bounds for sigma parameter '{sigma_param}'")
        full_param_bounds[sigma_param] = list(param_bounds[sigma_param])

    print(f"\nAnalyzing {len(all_param_names)} parameters (r={n_trajectories}, p={n_levels}):")
    print(f"  Parameters: {', '.join(all_param_names)}")
    print(f"  Outputs: {', '.join(output_vars)}")

    # Prepare observations
    obs_times = prepare_observation_times(site_data, observed_data)
    obs_indices = jnp.array(obs_times, dtype=jnp.int32)
    obs_values = {
        var: jnp.array(observed_data[var].values, dtype=jnp.float32)
        for var in output_vars
        if var in observed_data.columns
    }

    # Create batched function
    def _log_likelihood_components(param_values):
        return log_likelihood_components(
            param_values,
            params,
            initial_state,
            climate,
            site_data,
            species_data,
            obs_indices,
            obs_values,
            output_vars,
            _param_index,
            _sigma_param_set,
            species_index,
        )

    batched_log_likelihood = jax.jit(jax.vmap(_log_likelihood_components, in_axes=0))

    return {
        "all_param_names": all_param_names,
        "full_param_bounds": full_param_bounds,
        "batched_log_likelihood": batched_log_likelihood,
    }


def run_morris_analysis(
    file_path: str,
    observed_data: pd.DataFrame,
    calib_params: list[str],
    output_vars: list[str],
    param_bounds: dict[str, tuple[float, float]],
    sigma_param_names: dict[str, str] | None = None,
    n_levels: int = 20,
    n_trajectories: int = 500,
    seed: int = 432,
    species_index: int = 0,
    export_csv: bool = True,
    save_dir: str = "",
) -> dict[str, Any]:
    """Run Morris sensitivity analysis with individual output components."""
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
        n_levels,
        n_trajectories,
        seed,
        species_index,
    )

    # Create Morris problem
    problem = {
        "num_vars": len(setup["all_param_names"]),
        "names": setup["all_param_names"],
        "bounds": [setup["full_param_bounds"][name] for name in setup["all_param_names"]],
    }

    # Generate samples
    print("\nGenerating Morris samples...")
    param_values = morris_sample.sample(problem, n_trajectories, num_levels=n_levels, seed=seed)
    print(f"Generated {param_values.shape[0]} samples")

    # Evaluate all samples
    print("\nEvaluating log-likelihood with JAX batching...")
    component_names = output_vars + ["total"]
    component_values = {name: np.zeros(param_values.shape[0]) for name in component_names}
    invalid_counts = {name: 0 for name in component_names}

    batch_size = min(100, param_values.shape[0])
    batch_starts = list(range(0, param_values.shape[0], batch_size))
    print_every = max(1, len(batch_starts) // 10)
    for batch_idx, start in enumerate(batch_starts):
        stop = min(start + batch_size, param_values.shape[0])
        batch_results = evaluate_batch_samples(
            param_values[start:stop], setup["batched_log_likelihood"]
        )

        if batch_idx % print_every == 0:
            print(f"  {start}/{param_values.shape[0]}")

        for comp_idx, name in enumerate(component_names):
            component_values[name][start:stop] = batch_results[:, comp_idx]
            invalid_counts[name] += int(np.sum(~np.isfinite(batch_results[:, comp_idx])))

    # Analyze each component
    results: dict[str, dict[str, Any]] = {}
    for component_name in component_names:
        values = component_values[component_name]
        n_invalid = invalid_counts[component_name]

        if n_invalid == len(values):
            print(f"\nWARNING: All values for {component_name} are invalid, skipping")
            continue

        if n_invalid > 0:
            trajectory_size = len(setup["all_param_names"]) + 1
            n_trajectories_actual = values.shape[0] // trajectory_size
            x3 = param_values[: n_trajectories_actual * trajectory_size].reshape(
                n_trajectories_actual, trajectory_size, -1
            )
            y2 = values[: n_trajectories_actual * trajectory_size].reshape(
                n_trajectories_actual, trajectory_size
            )
            valid_traj_mask = np.all(np.isfinite(y2), axis=1)

            if not np.any(valid_traj_mask):
                print(f"\nWARNING: No valid trajectories for {component_name}")
                continue

            values_to_analyze = y2[valid_traj_mask].reshape(-1)
            x_to_analyze = x3[valid_traj_mask].reshape(-1, param_values.shape[1])
        else:
            values_to_analyze = values
            x_to_analyze = param_values

        morris_result = morris_analyze.analyze(
            problem,
            x_to_analyze,
            values_to_analyze,
            num_levels=n_levels,
            conf_level=0.95,
            scaled=False,
            print_to_console=False,
        )

        results[component_name] = {
            "mu": morris_result["mu"],
            "mu_star": morris_result["mu_star"],
            "sigma": morris_result["sigma"],
            "mu_star_conf": morris_result["mu_star_conf"],
            "names": setup["all_param_names"],
            "n_valid": len(values) - n_invalid,
            "n_total": len(values),
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
                        "mu_star": result["mu_star"][i],
                        "sigma": result["sigma"][i],
                        "mu": result["mu"][i],
                    }
                )
        combined_df = pd.DataFrame(combined_data)
        combined_df = combined_df.sort_values(["Component", "mu_star"], ascending=[True, False])

        # Print summary
        print("\nSUMMARY\n")
        for comp in combined_df["Component"].unique():
            print(f"\nCOMPONENT: {comp.upper()}")
            print(combined_df[combined_df["Component"] == comp].head(15))

    # Export results
    if export_csv and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        combined_df.to_csv(f"{save_dir}/morris_all_components.csv", index=False)
        print(f"\nResults saved to {save_dir}/morris_all_components.csv")

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

    # Run analysis
    results = run_morris_analysis(
        file_path=file_path,
        observed_data=observed_df,
        calib_params=calib_params,
        output_vars=output_vars,
        param_bounds=param_bounds,
        n_trajectories=1000,
        n_levels=20,
        sigma_param_names=sigma_param_names,
        export_csv=True,
        save_dir=os.path.join(results_data_folder, "morris_analysis_results_jax"),
        seed=432,
    )
