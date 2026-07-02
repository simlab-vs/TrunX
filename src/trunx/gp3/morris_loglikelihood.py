"""Morris sensitivity analysis on log-likelihood."""

import os
import warnings
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze
from SALib.sample import morris as morris_sample

from trunx.config import results_data_folder, threepg_data_folder
from trunx.gp3.model_inputs import Params
from trunx.gp3.PG3_model_impl import prepare_data
from trunx.gp3.run_3pg import run_3pg

warnings.filterwarnings("ignore")

# If a prediction exceeds this multiple of the observed data's own scale, the
# model has numerically diverged (e.g. an allometric coefficient sampled at a
# boundary that divides by ~0) and the resulting likelihood is not meaningful,
# regardless of which parameter combination caused it.
PLAUSIBILITY_MULTIPLIER = 100.0


def _load_saved_morris_results(results_csv_path: str) -> pd.DataFrame:
    """Load exported Morris results from CSV.

    Parameters
    ----------
    results_csv_path : str
        Path to the exported `morris_all_components.csv` file.

    Returns
    -------
    pd.DataFrame
        Morris results table.
    """
    if not os.path.exists(results_csv_path):
        raise FileNotFoundError(f"Results file not found: {results_csv_path}")
    return pd.read_csv(results_csv_path)


def plot_component_comparison_from_results(
    results_csv_path: str, output_vars: list[str], save_path: str | None = None
) -> None:
    """Compare top parameters across all components from saved results.

    Parameters
    ----------
    results_csv_path : str
        Path to the exported `morris_all_components.csv` file.
    output_vars : list[str]
        Output variables to include in the comparison.
    save_path : str | None, optional
        Path to save the figure.
    """
    combined_df = _load_saved_morris_results(results_csv_path)
    components = [c for c in ["total", *output_vars] if c in combined_df["Component"].unique()]
    fig, axes = plt.subplots(1, len(components), figsize=(5 * len(components), 8))
    if len(components) == 1:
        axes = [axes]

    for ax, comp in zip(axes, components, strict=True):
        df = combined_df[combined_df["Component"] == comp].nlargest(10, "mu_star")
        colors = plt.get_cmap("viridis")(df["mu_star"] / df["mu_star"].max())
        ax.barh(range(len(df)), df["mu_star"], color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["Parameter"], fontsize=8)
        ax.set_xlabel(r"$\mu^*$", fontsize=10)
        ax.set_title(comp.upper(), fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle(
        "Morris Sensitivity: Parameter Rankings by Component", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_individual_sensitivity_from_results(
    results_csv_path: str, component_name: str, save_path: str | None = None
) -> None:
    """Plot detailed sensitivity for one component from saved results.

    Parameters
    ----------
    results_csv_path : str
        Path to the exported `morris_all_components.csv` file.
    component_name : str
        Component name to plot.
    save_path : str | None, optional
        Path to save the figure.
    """
    combined_df = _load_saved_morris_results(results_csv_path)
    df = combined_df[combined_df["Component"] == component_name].nlargest(20, "mu_star")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(df) * 0.3)))

    ax1.barh(
        range(len(df)),
        df["mu_star"],
        color=plt.get_cmap("RdYlGn_r")(np.linspace(0, 1, len(df))),
    )
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels(df["Parameter"], fontsize=9)
    ax1.set_xlabel(r"$\mu^*$", fontsize=11)
    ax1.set_title(f"{component_name.upper()} - Sensitivity", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")

    all_data = combined_df[combined_df["Component"] == component_name]
    scatter = ax2.scatter(
        all_data["mu_star"],
        all_data["sigma"],
        c=all_data["mu_star"],
        cmap="viridis",
        s=80,
        alpha=0.7,
    )
    ax2.set_xlabel(r"$\mu^*$", fontsize=11)
    ax2.set_ylabel(r"$\sigma$", fontsize=11)
    ax2.set_title("Sensitivity vs Interactions", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label=r"$\mu^*$")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


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

    def _output_log_likelihood_components(
        self, model_outputs: dict, sampled_params: dict[str, float]
    ) -> dict[str, float]:
        """Compute log-likelihood for each output variable individually."""
        log_lik_components = {}

        for var_name in self.output_vars:
            if var_name not in self.obs_values:
                continue

            model_values = model_outputs.get(var_name)
            if model_values is None:
                log_lik_components[var_name] = -np.inf
                continue

            if len(model_values.shape) == 2:
                model_values = model_values[:, self.species_index]

            predictions = model_values[self.obs_indices]
            observations = self.obs_values[var_name]

            valid_mask = ~(jnp.isnan(predictions) | jnp.isnan(observations))
            predictions, observations = predictions[valid_mask], observations[valid_mask]

            if len(predictions) > 0:
                sigma_param_name = self._infer_sigma_param_name(var_name)
                if sigma_param_name is None:
                    raise KeyError(
                        f"No sigma parameter found for output variable '{var_name}'. "
                        "Add an entry to sigma_param_names or an 'err_*' param to param_bounds."
                    )
                sigma = sampled_params[sigma_param_name]

                obs_scale = float(jnp.max(jnp.abs(observations))) + 1e-8
                if float(jnp.max(jnp.abs(predictions))) >= PLAUSIBILITY_MULTIPLIER * obs_scale:
                    log_lik_components[var_name] = -np.inf
                    continue

                # Normal log-likelihood
                residuals = predictions - observations
                log_lik = jnp.sum(
                    -0.5 * ((residuals / sigma) ** 2) - jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
                )
                log_lik_components[var_name] = float(log_lik)
            else:
                log_lik_components[var_name] = -np.inf

        # Total log-likelihood: any non-finite component poisons the total,
        # matching R's `is.nan(logpost) | is.na(logpost) | logpost == 0 -> -Inf`.
        total = sum(log_lik_components.values())
        if np.isnan(total) or total == 0.0:
            total = -np.inf
        log_lik_components["total"] = total

        return log_lik_components

    def _log_likelihood(self, param_values: np.ndarray) -> dict[str, float]:
        """Compute log-likelihood for total and each component."""
        # Update parameters
        params_dict = {field: getattr(self.params, field) for field in self.params._fields}
        sampled_params = dict(zip(self.all_param_names, param_values, strict=True))

        for i, param_name in enumerate(self.all_param_names):
            if param_name in self._sigma_param_set:
                continue
            params_dict[param_name] = param_values[i]

        # Run model
        final_state, model_outputs = run_3pg(
            initial_state=self.initial_state,
            climate=self.climate,
            params=Params(**params_dict),
            site=self.site_data,
            species=self.species_data,
            n_species=self.n_species,
        )

        # Get log-likelihood components
        log_lik_components = self._output_log_likelihood_components(model_outputs, sampled_params)

        # Likelihood components only
        likelihood_components = {}
        for key, log_lik in log_lik_components.items():
            if np.isinf(log_lik) or np.isnan(log_lik) or log_lik == 0.0:
                likelihood_components[key] = -np.inf
            else:
                likelihood_components[key] = log_lik

        return likelihood_components

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
        print("\nEvaluating log-likelihood for all components...")

        # Initialize storage for each component
        component_names = self.output_vars + ["total"]
        component_values = {
            name: np.zeros(
                self.param_values.shape[0],
                dtype=np.float32,
            )
            for name in component_names
        }

        invalid_samples = {name: [] for name in component_names}
        for i in range(self.param_values.shape[0]):
            if i % 100 == 0:
                print(f"  {i}/{self.param_values.shape[0]}")
            try:
                likelihood_components = self._log_likelihood(self.param_values[i])
                for name in component_names:
                    value = likelihood_components.get(name, -np.inf)
                    component_values[name][i] = value

                    if not np.isfinite(value):
                        invalid_samples[name].append(i)
            except Exception as e:
                print(f"Error evaluating sample {i}: {e}")
                for name in component_names:
                    component_values[name][i] = -np.inf
                    invalid_samples[name].append(i)

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
                invalid_idx_set = set(invalid_indices)
                for idx in invalid_indices:
                    # Find nearest valid index within same trajectory
                    trajectory_size = len(self.all_param_names) + 1
                    start_idx = (idx // trajectory_size) * trajectory_size
                    end_idx = min(start_idx + trajectory_size, len(values))

                    # Look for valid values in the same trajectory
                    valid_in_trajectory = []
                    for j in range(start_idx, end_idx):
                        if j not in invalid_idx_set and np.isfinite(values[j]):
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
            print(
                f"\nCOMPONENT: {comp.upper()}\n{'Rank':<5} {'Parameter':<25} \
                    {'mu_star':<20} {'sigma':<20}"
            )
            df = self.combined_df[self.combined_df["Component"] == comp]
            for rank, (_, row) in enumerate(df.head(15).iterrows(), 1):
                print(
                    f"{rank:<5} {row['Parameter']:<25} {row['mu_star']:<20.6f} \
                        {row['sigma']:<20.6f}"
                )

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

    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        results_csv_path = os.path.join(save_dir, "morris_all_components.csv")

        # Plot individual components
        for component in ["total"] + output_vars:
            if component in analyzer.results:
                plot_individual_sensitivity_from_results(
                    results_csv_path,
                    component,
                    save_path=f"{save_dir}/morris_{component}.png",
                )

        # Plot comparison grid
        plot_component_comparison_from_results(
            results_csv_path,
            output_vars,
            save_path=f"{save_dir}/morris_comparison.png",
        )

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
    output_vars = ["DBH", "WS", "WR", "WF", "BA", "Height"]

    # sigma_param_names maps each model output key to its corresponding err_* parameter
    sigma_param_names = {
        "DBH": "err_DBH",
        "WS": "err_WS",
        "WR": "err_WR",
        "WF": "err_WF",
        "BA": "err_BA",
        "Height": "err_Height",
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
        save_dir=os.path.join(results_data_folder, "morris_analysis_results"),
    )
