"""Morris sensitivity analysis on log-posterior."""

import os
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze
from SALib.sample import morris as morris_sample

from trunx.gp3.model_inputs import Params
from trunx.gp3.PG3_model_impl import prepare_data
from trunx.gp3.run_3pg import run_3pg

warnings.filterwarnings("ignore")


class MorrisSensitivityOnPosterior:
    """Morris sensitivity analysis on log-posterior."""

    def __init__(
        self,
        file_path: str,
        observed_data: pd.DataFrame,
        calib_params: list[str],
        output_vars: list[str],
        param_bounds: dict[str, tuple[float, float]],
        param_best: dict[str, float],
        fixed_sigmas: dict[str, float],
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
        self.fixed_sigmas = fixed_sigmas or {}
        self.param_bounds = param_bounds or {}
        self.param_best = param_best or {}

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

        self.all_param_names = self.calib_params
        self.full_param_bounds = {}
        self.par_cal_best = {}

        print("Calibration parameters: ", self.calib_params)
        for param in self.calib_params:
            self.full_param_bounds[param] = list(self.param_bounds[param])
            self.par_cal_best[param] = self.param_best[param]

        self.par_cal_min = [self.full_param_bounds[name][0] for name in self.all_param_names]
        self.par_cal_max = [self.full_param_bounds[name][1] for name in self.all_param_names]

        print(
            f"\nAnalyzing {len(self.all_param_names)} parameters \
                (r={n_trajectories}, p={n_levels}):"
        )
        print(f"  Calibration parameters: {', '.join(self.calib_params)}")
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

    def _log_likelihood_components(self, model_outputs: dict) -> dict[str, float]:
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
                sigma = self.fixed_sigmas.get(var_name, 1.0)

                # Normal log-likelihood
                residuals = predictions - observations
                log_lik = jnp.sum(
                    -0.5 * ((residuals / sigma) ** 2) - jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
                )
                log_lik_components[var_name] = float(log_lik)
            else:
                log_lik_components[var_name] = -np.inf

        # Total log-likelihood is sum of components
        valid_components = [v for v in log_lik_components.values() if not np.isinf(v)]
        log_lik_components["total"] = sum(valid_components) if valid_components else -np.inf

        return log_lik_components

    def _uniform_prior(self, par_v: np.ndarray) -> float:
        """Uniform prior on log scale."""
        for i, (min_val, max_val) in enumerate(
            zip(self.par_cal_min, self.par_cal_max, strict=True)
        ):
            if par_v[i] < min_val or par_v[i] > max_val:
                return -np.inf
        return 0.0

    def _log_posterior_components(self, param_values: np.ndarray) -> dict[str, float]:
        """Compute log posterior for total and each component."""
        # Update parameters
        params_dict = {field: getattr(self.params, field) for field in self.params._fields}

        for i, param_name in enumerate(self.all_param_names):
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
        log_lik_components = self._log_likelihood_components(model_outputs)

        # Add prior (uniform prior)
        log_prior = self._uniform_prior(param_values)
        if np.isinf(log_prior):
            return {key: -np.inf for key in log_lik_components}

        # Posterior = likelihood + prior for each component
        posterior_components = {}
        for key, log_lik in log_lik_components.items():
            if np.isinf(log_lik):
                posterior_components[key] = -np.inf
            else:
                posterior_components[key] = log_lik + log_prior

        return posterior_components

    def _create_morris_problem(self) -> dict:
        """Create SALib problem definition."""
        return {
            "num_vars": len(self.all_param_names),
            "names": self.all_param_names,
            "bounds": [self.full_param_bounds[name] for name in self.all_param_names],
        }

    def run_analysis(self) -> dict:
        """Run Morris sensitivity analysis on log-posterior for all components."""
        print("MORRIS SENSITIVITY ANALYSIS ON LOG-POSTERIOR")
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

        # Evaluate log-posterior for all samples
        print("\nEvaluating log-posterior for all components...")

        # Initialize storage for each component
        component_names = self.output_vars + ["total"]
        component_values = {name: np.zeros(self.param_values.shape[0]) for name in component_names}

        invalid_samples = {name: [] for name in component_names}
        for i in range(self.param_values.shape[0]):
            if i % 100 == 0:
                print(f"  {i}/{self.param_values.shape[0]}")

            try:
                posterior_components = self._log_posterior_components(self.param_values[i])
                for name in component_names:
                    value = posterior_components.get(name, -np.inf)
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
        self.results = {}
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

    def plot_component_comparison(self, save_path=None):
        """Compare top parameters across all components."""
        if not hasattr(self, "combined_df"):
            self.export_all_results()

        components = [
            c for c in ["total"] + self.output_vars if c in self.combined_df["Component"].unique()
        ]
        fig, axes = plt.subplots(1, len(components), figsize=(5 * len(components), 8))
        if len(components) == 1:
            axes = [axes]

        for ax, comp in zip(axes, components, strict=True):
            df = self.combined_df[self.combined_df["Component"] == comp].nlargest(10, "mu_star")
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

    def plot_individual_sensitivity(self, component_name, save_path=None):
        """Plot detailed sensitivity for a specific component."""
        df = self.combined_df[self.combined_df["Component"] == component_name].nlargest(
            20, "mu_star"
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(df) * 0.3)))

        # Bar plot
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

        # Interaction plot
        all_data = self.combined_df[self.combined_df["Component"] == component_name]
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

    def export_all_results(self, save_dir: str = "morris_results"):
        """Export all results to CSV files."""
        os.makedirs(save_dir, exist_ok=True)
        # Export combined results
        combined_data = []
        for component_name, result in self.results.items():
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
    fixed_sigmas: dict[str, float],
    n_trajectories: int = 500,
    n_levels: int = 20,
    save_plots: bool = True,
    export_csv: bool = True,
    save_dir: str = "",
) -> MorrisSensitivityOnPosterior:
    """Run Morris sensitivity analysis with individual output components."""
    if fixed_sigmas is None:
        fixed_sigmas = {"WS": 5.0, "DBH": 1.0, "Height": 2.0, "BA": 3.0}

    analyzer = MorrisSensitivityOnPosterior(
        file_path=file_path,
        observed_data=observed_data,
        calib_params=calib_params,
        param_bounds=params_bounds,
        param_best=param_best,
        output_vars=output_vars,
        n_levels=n_levels,
        n_trajectories=n_trajectories,
        fixed_sigmas=fixed_sigmas,
    )

    analyzer.run_analysis()
    analyzer.export_all_results(save_dir=save_dir)
    analyzer.print_summary()

    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

        # Plot individual components
        for component in ["total"] + output_vars:
            if component in analyzer.results:
                analyzer.plot_individual_sensitivity(
                    component, save_path=f"{save_dir}/morris_{component}.png"
                )

        # Plot comparison grid
        analyzer.plot_component_comparison(save_path=f"{save_dir}/morris_comparison.png")

    if export_csv:
        analyzer.export_all_results(save_dir)

    return analyzer


if __name__ == "__main__":
    # Observed data
    file_path = "./data/solling_data.xlsx"
    observed_df = pd.read_excel(file_path, sheet_name="observed")

    # Get parameter bounds
    param_bounds_df = pd.read_excel(file_path, sheet_name="param_bound")
    param_bounds = {}
    param_best = {}
    for _, row in param_bounds_df.iterrows():
        if pd.notna(row["min"]) and pd.notna(row["max"]):
            param_bounds[row["param_name"]] = (row["min"], row["max"])
            param_best[row["param_name"]] = row["default"]

    calib_params = list(param_bounds.keys())

    # Run analysis for multiple output variables
    analyzer = run_morris_analysis(
        file_path=file_path,
        observed_data=observed_df,
        calib_params=calib_params,
        output_vars=["WS", "DBH", "WF", "WR"],
        params_bounds=param_bounds,
        param_best=param_best,
        n_trajectories=100,
        n_levels=20,
        fixed_sigmas={
            "WS": 1.0,
            "DBH": 1.0,
            "WF": 1.0,
            "WR": 1.0,
        },
        save_plots=True,
        export_csv=True,
        save_dir=os.path.join("./data/", "morris_analysis_results"),
    )
