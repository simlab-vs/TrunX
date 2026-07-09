"""Morris sensitivity analysis on log-likelihood."""

import os
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze
from SALib.sample import morris as morris_sample

from trunx.config import results_data_folder, threepg_data_folder
from trunx.gp3.model_inputs import Params
from trunx.gp3.PG3_model_impl import prepare_data
from trunx.gp3.run_3pg import run_3pg as run_3pg_orig

warnings.filterwarnings("ignore")

# If a prediction exceeds this multiple of the observed data's own scale, the
# model has numerically diverged (e.g. an allometric coefficient sampled at a
# boundary that divides by ~0) and the resulting likelihood is not meaningful,
# regardless of which parameter combination caused it.
PLAUSIBILITY_MULTIPLIER = 100.0

# JIT compile the 3PG model for faster execution
run_3pg = jax.jit(run_3pg_orig)


class MorrisSensitivityOnLikelihood:
    """Morris sensitivity analysis on log-likelihood."""

    def __init__(
        self,
        file_path: str,
        observed_data: pd.DataFrame,
        calib_params: list[str],
        output_vars: list[str],
        param_bounds: dict[str, tuple[float, float]],
        param_best: dict[str, float],
        sigma_param_names: dict[str, str] | None = None,
        n_levels: int = 20,
        n_trajectories: int = 500,
        seed: int = 432,
        species_index: int = 0,
    ):
        self.file_path = file_path
        self.observed_data = observed_data
        self.calib_params = calib_params
        self.output_vars = output_vars
        self.n_levels = n_levels
        self.n_trajectories = n_trajectories
        self.seed = seed
        self.species_index = species_index
        self.sigma_param_names = sigma_param_names or {}
        self.param_bounds = param_bounds or {}
        self.param_best = param_best or {}
        self._sigma_param_set = set(self.sigma_param_names.values())

        np.random.seed(seed)

        # Load model data
        print("Loading model data...")
        (
            self.initial_state,
            self.climate,
            self.params,
            self.site_data,
            self.species_data,
            self.n_species,
            self.species_names,
        ) = prepare_data(file_path)

        self.all_param_names = list(dict.fromkeys(self.calib_params + list(self._sigma_param_set)))
        self._param_index = {name: idx for idx, name in enumerate(self.all_param_names)}
        self.full_param_bounds = {}
        self.par_cal_best = {}

        print("Calibration parameters: ", self.all_param_names)
        for param in self.calib_params:
            self.full_param_bounds[param] = list(self.param_bounds[param])
            self.par_cal_best[param] = self.param_best[param]

        for sigma_param in self._sigma_param_set:
            if sigma_param not in self.param_bounds:
                raise KeyError(
                    f"Missing bounds for sigma parameter '{sigma_param}'. Add it to param_bounds."
                )
            self.full_param_bounds[sigma_param] = list(self.param_bounds[sigma_param])
            if sigma_param in self.param_best:
                self.par_cal_best[sigma_param] = self.param_best[sigma_param]

        self.par_cal_min = [self.full_param_bounds[name][0] for name in self.all_param_names]
        self.par_cal_max = [self.full_param_bounds[name][1] for name in self.all_param_names]

        print(
            f"\nAnalyzing {len(self.all_param_names)} parameters \
                (r={n_trajectories}, p={n_levels}):"
        )
        print(f"  Calibration parameters: {', '.join(self.all_param_names)}")
        print(f"  Output variables for individual analysis: {', '.join(self.output_vars)}")

        # Prepare observation times and values
        self.obs_times = self._prepare_observation_times()
        self.obs_indices = jnp.array(self.obs_times, dtype=jnp.int32)

        self.obs_values = {}
        for var_name in self.output_vars:
            if var_name in observed_data.columns:
                self.obs_values[var_name] = jnp.array(
                    observed_data[var_name].values, dtype=jnp.float32
                )

        self._batched_log_likelihood = jax.jit(
            jax.vmap(self._log_likelihood_components, in_axes=0)
        )

        # Store results for each output
        self.results = {}
        self.param_values = None
        self.all_outputs = None

    def _prepare_observation_times(self) -> np.ndarray:
        """Prepare time indices for observations."""
        start_year = self.site_data.year_i
        start_month = self.site_data.month_i

        obs_months = []
        for _, row in self.observed_data.iterrows():
            obs_year = row["year"]
            months = (obs_year - start_year) * 12 + (12 - start_month)
            obs_months.append(max(0, months))
        return np.array(obs_months, dtype=int)

    def _infer_sigma_param_name(self, var_name: str) -> str | None:
        """Infer the sampled sigma parameter name for a variable."""
        candidates = (
            self.sigma_param_names.get(var_name),
            f"err_{var_name}",
            f"sigma_{var_name}",
            f"sd_{var_name}",
        )
        for candidate in candidates:
            if candidate and candidate in self.full_param_bounds:
                return candidate
        return None

    def _component_log_likelihood(
        self,
        model_values: jnp.ndarray,
        observations: jnp.ndarray,
        sigma: float | jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the log-likelihood for one output component (DBH, WS, etc)."""
        sigma = jnp.asarray(sigma, dtype=jnp.float32)
        if len(model_values.shape) == 2:
            model_values = model_values[:, self.species_index]

        predictions = model_values[self.obs_indices]
        valid_mask = ~(jnp.isnan(predictions) | jnp.isnan(observations))
        residuals = predictions - observations
        log_terms = jnp.where(
            valid_mask,
            -0.5 * ((residuals / sigma) ** 2) - jnp.log(sigma * jnp.sqrt(2 * jnp.pi)),
            0.0,
        )
        valid_count = jnp.sum(valid_mask)
        result = jnp.where(valid_count > 0, jnp.sum(log_terms), -jnp.inf)

        obs_scale = jnp.max(jnp.abs(observations)) + 1e-8
        plausible = jnp.all(jnp.abs(predictions) < PLAUSIBILITY_MULTIPLIER * obs_scale)
        return jnp.where(plausible, result, -jnp.inf)

    def _log_likelihood_components(self, param_values: jnp.ndarray) -> jnp.ndarray:
        """Compute component log-likelihoods for one sample on all components with JAX."""
        params_dict = {field: getattr(self.params, field) for field in self.params._fields}

        for i, param_name in enumerate(self.all_param_names):
            if param_name in self._sigma_param_set:
                continue
            params_dict[param_name] = param_values[i]

        _, model_outputs = run_3pg(
            initial_state=self.initial_state,
            climate=self.climate,
            params=Params(**params_dict),
            site=self.site_data,
            species=self.species_data,
        )

        component_values = []
        for var_name in self.output_vars:
            model_values = model_outputs.get(var_name)
            if model_values is None:
                component_values.append(jnp.asarray(-jnp.inf, dtype=jnp.float64))
                continue

            sigma_param_name = self._infer_sigma_param_name(var_name)
            if sigma_param_name is None:
                raise KeyError(
                    f"No sigma parameter found for output variable '{var_name}'. "
                    "Add an entry to sigma_param_names or an 'err_*' param to param_bounds."
                )

            component_values.append(
                self._component_log_likelihood(
                    model_values,
                    self.obs_values[var_name],
                    param_values[self._param_index[sigma_param_name]],
                )
            )

        component_array = jnp.asarray(component_values, dtype=jnp.float64)
        total = jnp.sum(component_array)
        total = jnp.where(jnp.isnan(total) | (total == 0.0), -jnp.inf, total)
        return jnp.concatenate([component_array, jnp.asarray([total], dtype=jnp.float64)])

    def _evaluate_batch_samples(self, sample_values: np.ndarray) -> np.ndarray:
        """Evaluate a batch of Morris samples with JAX."""
        batch_results = self._batched_log_likelihood(jnp.asarray(sample_values, dtype=jnp.float64))
        return np.asarray(jax.block_until_ready(batch_results))

    def _create_morris_problem(self) -> dict:
        """Create SALib problem definition."""
        return {
            "num_vars": len(self.all_param_names),
            "names": self.all_param_names,
            "bounds": [self.full_param_bounds[name] for name in self.all_param_names],
        }

    def run_analysis(self) -> dict:
        """Run Morris sensitivity analysis on log-likelihood for all components."""
        print("MORRIS SENSITIVITY ANALYSIS ON LOG-LIKELIHOOD")
        print(f"Parameters: {len(self.all_param_names)}")
        print(f"Output components: {', '.join(self.output_vars)} + total")
        print(f"Trajectories (r): {self.n_trajectories}")
        print(f"Levels (p): {self.n_levels}")

        problem = self._create_morris_problem()

        # Generate Morris samples
        print("\nGenerating Morris samples (OAT design)...")
        self.param_values = morris_sample.sample(
            problem,
            self.n_trajectories,
            self.n_levels,
        )
        print(f"Generated {self.param_values.shape[0]} samples")

        # Evaluate log-likelihood for all samples
        print("\nEvaluating log-likelihood for all components with JAX batching...")

        # Initialize storage for each component
        component_names = self.output_vars + ["total"]
        component_values = {name: np.zeros(self.param_values.shape[0]) for name in component_names}

        invalid_samples = {name: [] for name in component_names}
        batch_size = min(100, self.param_values.shape[0])
        for start in range(0, self.param_values.shape[0], batch_size):
            stop = min(start + batch_size, self.param_values.shape[0])
            batch_results = self._evaluate_batch_samples(self.param_values[start:stop])

            if start % max(100, batch_size) == 0:
                print(f"  {start}/{self.param_values.shape[0]}")

            for local_idx, global_idx in enumerate(range(start, stop)):
                values = batch_results[local_idx]
                for comp_idx, name in enumerate(component_names):
                    value = float(values[comp_idx])
                    component_values[name][global_idx] = value

                    if not np.isfinite(value):
                        invalid_samples[name].append(global_idx)

        # Store all outputs for later use
        self.all_outputs = component_values

        for name in component_names:
            n_inf = np.sum(np.isinf(component_values[name]))
            if n_inf > 0:
                print(
                    f"\nWARNING: {name}: {n_inf} infinite values \
                        ({n_inf / len(component_values[name]) * 100:.1f}%)"
                )

        # Analyze each component separately
        self.results: dict[str, Any] = {}
        for component_name in component_names:
            values = component_values[component_name]
            # valid_mask = ~np.isinf(values)
            invalid_indices = invalid_samples[component_name]

            if len(invalid_indices) == len(values):
                print(f"\nWARNING: All values for {component_name} are invalid, skipping analysis")
                continue

            if len(invalid_indices) > 0:
                print(f" Handling {len(invalid_indices)} invalid samples for {component_name}...")

                values_clean = values.copy()
                for idx in invalid_indices:
                    # Find nearest valid index within same trajectory
                    trajectory_size = len(self.all_param_names) + 1
                    start_idx = (idx // trajectory_size) * trajectory_size
                    end_idx = min(start_idx + trajectory_size, len(values))

                    # Look for valid values in the same trajectory
                    valid_in_trajectory = []
                    for j in range(start_idx, end_idx):
                        if j not in invalid_indices and np.isfinite(values[j]):
                            valid_in_trajectory.append(values[j])

                    if valid_in_trajectory:
                        # Take mean of valid values in the same trajectory
                        values_clean[idx] = np.mean(valid_in_trajectory)
                    else:
                        # Global mean of valid values
                        all_valid = values[~np.isnan(values) & np.isfinite(values)]
                        values_clean[idx] = np.mean(all_valid) if len(all_valid) > 0 else 0.0

                values_to_analyze = values_clean
            else:
                values_to_analyze = values

            # Run Morris analysis
            morris_result = morris_analyze.analyze(
                problem,
                self.param_values,
                values_to_analyze,
                num_levels=self.n_levels,
                conf_level=0.95,
                print_to_console=False,
            )

            # Store results
            self.results[component_name] = {
                "mu": morris_result["mu"],
                "mu_star": morris_result["mu_star"],
                "sigma": morris_result["sigma"],
                "mu_star_conf": morris_result["mu_star_conf"],
                "names": self.all_param_names,
                "n_valid": len(values) - len(invalid_indices),
                "n_total": len(values),
                "n_invalid": len(invalid_indices),
            }

        return self.results

    def print_summary(self):
        """Print comprehensive summary for all components."""
        print(" Summary \n ")
        for comp in self.combined_df["Component"].unique():
            print(f"\nCOMPONENT: {comp.upper()}")
            df = self.combined_df[self.combined_df["Component"] == comp].reset_index(drop=True)
            print(df.head(15))

    def export_all_results(self, save_dir: str = "morris_results"):
        """Export all results to CSV files."""
        os.makedirs(save_dir, exist_ok=True)
        # Export combined results
        combined_data = []
        for component_name, result in self.results.items():
            names = list(result["names"])
            mu_star = np.asarray(result["mu_star"])
            sigma = np.asarray(result["sigma"])
            mu = np.asarray(result["mu"])
            for i, param_name in enumerate(names):
                combined_data.append(
                    {
                        "Component": component_name,
                        "Parameter": param_name,
                        "mu_star": mu_star[i],
                        "sigma": sigma[i],
                        "mu": mu[i],
                    }
                )

        self.combined_df = pd.DataFrame(combined_data)
        self.combined_df = self.combined_df.sort_values(
            ["Component", "mu_star"], ascending=[True, False]
        )
        self.combined_df.to_csv(f"{save_dir}/morris_all_components.csv", index=False)
        print(f"\nAll results saved to {save_dir}/morris_all_components.csv")

        return self.combined_df


def run_morris_analysis(
    file_path: str,
    observed_data: pd.DataFrame,
    calib_params: list[str],
    output_vars: list[str],
    params_bounds: dict[str, tuple[float, float]],
    param_best: dict[str, float],
    sigma_param_names: dict[str, str] | None = None,
    n_trajectories: int = 500,
    n_levels: int = 20,
    save_plots: bool = True,
    export_csv: bool = True,
    save_dir: str = "",
) -> MorrisSensitivityOnLikelihood:
    """Run Morris sensitivity analysis with individual output components."""
    analyzer = MorrisSensitivityOnLikelihood(
        file_path=file_path,
        observed_data=observed_data,
        calib_params=calib_params,
        param_bounds=params_bounds,
        param_best=param_best,
        output_vars=output_vars,
        sigma_param_names=sigma_param_names,
        n_levels=n_levels,
        n_trajectories=n_trajectories,
    )

    analyzer.run_analysis()
    analyzer.export_all_results(save_dir=save_dir)
    analyzer.print_summary()

    if export_csv:
        analyzer.export_all_results(save_dir)

    return analyzer


if __name__ == "__main__":
    # Observed data
    file_path = os.path.join(threepg_data_folder, "solling_data.xlsx")
    observed_df = pd.read_excel(file_path, sheet_name="observed")

    # Get parameter bounds
    param_bounds_df = pd.read_excel(file_path, sheet_name="param_bound")
    error_bounds_df = pd.read_excel(file_path, sheet_name="error_param")
    param_bounds_df = pd.concat([param_bounds_df, error_bounds_df]).reset_index(drop=True)
    param_bounds = {}
    param_best = {}
    for _, row in param_bounds_df.iterrows():
        if pd.notna(row["min"]) and pd.notna(row["max"]):
            param_bounds[row["param_name"]] = (row["min"], row["max"])
            param_best[row["param_name"]] = row["default"]

    calib_params = list(param_bounds.keys())

    # Output variables to analyze
    output_vars = ["DBH", "WS", "WR", "WF", "Height", "BA"]

    # sigma_param_names maps each model output key to its corresponding err_* parameter
    sigma_param_names = {
        "DBH": "err_DBH",
        "WS": "err_WS",
        "WR": "err_WR",
        "WF": "err_WF",
        "Height": "err_Height",
        "BA": "err_BA",
    }

    analyzer = run_morris_analysis(
        file_path=file_path,
        observed_data=observed_df,
        calib_params=calib_params,
        output_vars=output_vars,
        params_bounds=param_bounds,
        param_best=param_best,
        n_trajectories=500,
        n_levels=20,
        sigma_param_names=sigma_param_names,
        save_plots=True,
        export_csv=True,
        save_dir=os.path.join(results_data_folder, "morris_analysis_results_jax"),
    )
