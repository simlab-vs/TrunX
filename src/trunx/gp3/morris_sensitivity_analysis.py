"""Morris sensitivity analysis on model outputs for 3PG model."""

import os
import warnings
from typing import TypedDict

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


class MorrisResult(TypedDict):
    """Typed Morris analysis result container."""

    mu: np.ndarray
    mu_star: np.ndarray
    sigma: np.ndarray
    mu_star_conf: np.ndarray
    names: list[str]
    n_valid: int
    n_total: int


class MorrisSensitivityOnOutputs:
    """Morris sensitivity analysis on model outputs."""

    def __init__(
        self,
        file_path: str,
        calib_params: list[str],
        output_vars: list[str],
        param_bounds: dict[str, tuple[float, float]],
        param_best: dict[str, float],
        n_levels: int = 20,
        n_trajectories: int = 500,
        seed: int = 432,
        species_index: int = 0,
        time_point: str = "final",  # 'final', 'mean', or index
    ):
        self.file_path = file_path
        self.calib_params = calib_params
        self.output_vars = output_vars
        self.n_levels = n_levels
        self.n_trajectories = n_trajectories
        self.seed = seed
        self.species_index = species_index
        self.param_bounds = param_bounds
        self.param_best = param_best
        self.time_point = time_point

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
             (r={n_trajectories}, p={n_levels})"
        )
        print(f"  Calibration parameters: {', '.join(self.calib_params)}")
        print(f"  Output variables: {', '.join(self.output_vars)}")

        # Store results for each output
        self.results: dict[str, MorrisResult] = {}
        self.param_values = None
        self.all_outputs = None

    def _extract_output_value(self, model_outputs: dict, var_name: str) -> float:
        """Extract output value."""
        if var_name not in model_outputs:
            return np.nan

        values = model_outputs[var_name]

        if len(values.shape) == 2:
            values = values[:, self.species_index]

        if self.time_point == "final":
            return float(values[-1])
        elif self.time_point == "mean":
            return float(np.mean(values))
        else:
            idx = int(self.time_point)
            return float(values[idx]) if idx < len(values) else float(values[-1])

    def _evaluate_model_outputs(self, param_values: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate model and extract outputs for all parameter sets."""
        n_samples = param_values.shape[0]
        outputs = {var: np.zeros(n_samples) for var in self.output_vars}

        print(f"\nEvaluating {n_samples} parameter combinations...")

        for i in range(n_samples):
            if i % 100 == 0:
                print(f"  {i}/{n_samples}")

            # Update parameters
            params_dict = {field: getattr(self.params, field) for field in self.params._fields}
            for j, param_name in enumerate(self.all_param_names):
                params_dict[param_name] = param_values[i][j]

            # Run model
            final_state, model_outputs = run_3pg(
                initial_state=self.initial_state,
                climate=self.climate,
                params=Params(**params_dict),
                site=self.site_data,
                species=self.species_data,
            )

            # Extract outputs
            for var in self.output_vars:
                outputs[var][i] = self._extract_output_value(model_outputs, var)

        return outputs

    def _create_morris_problem(self) -> dict:
        """Create SALib problem definition."""
        return {
            "num_vars": len(self.all_param_names),
            "names": self.all_param_names,
            "bounds": [self.full_param_bounds[name] for name in self.all_param_names],
        }

    def run_analysis(self) -> dict:
        """Run Morris sensitivity analysis on model outputs."""
        print("MORRIS SENSITIVITY ANALYSIS ON MODEL OUTPUTS")

        print(f"Parameters: {len(self.all_param_names)}")
        print(f"Output variables: {', '.join(self.output_vars)}")
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

        # Evaluate model outputs
        print("\nEvaluating model outputs...")
        self.all_outputs = self._evaluate_model_outputs(self.param_values)

        # Analyze each output variable
        print("\nAnalyzing elementary effects...")
        self.results = {}

        for var in self.output_vars:
            values = self.all_outputs[var]

            valid_mask = np.isfinite(values)
            if not np.any(valid_mask):
                print(f"  No valid values for {var}, skipping")
                continue

            if np.all(values[valid_mask] == values[valid_mask][0]):
                print(f"  Constant output for {var}, skipping")
                continue

            n_invalid = np.sum(~valid_mask)
            if n_invalid > 0:
                print(f"  Removing {n_invalid} invalid samples from analysis")
                param_values_valid = self.param_values[valid_mask]
                values_valid = values[valid_mask]
            else:
                param_values_valid = self.param_values
                values_valid = values

            # Run Morris analysis
            morris_result = morris_analyze.analyze(
                problem,
                param_values_valid,
                values_valid,
                num_levels=self.n_levels,
                conf_level=0.95,
                print_to_console=False,
            )

            self.results[var] = {
                "mu": morris_result["mu"],
                "mu_star": morris_result["mu_star"],
                "sigma": morris_result["sigma"],
                "mu_star_conf": morris_result["mu_star_conf"],
                "names": self.all_param_names,
                "n_valid": np.sum(valid_mask),
                "n_total": len(values),
            }
        self.combined_df = self._create_combined_df()
        return self.results

    def _create_combined_df(self) -> pd.DataFrame:
        """Create combined DataFrame from results."""
        data = []
        for var, result in self.results.items():
            mu_star = np.asarray(result["mu_star"])
            sigma = np.asarray(result["sigma"])
            mu = np.asarray(result["mu"])
            for i, name in enumerate(result["names"]):
                data.append(
                    {
                        "Component": var,
                        "Parameter": name,
                        "mu_star": mu_star[i],
                        "sigma": sigma[i],
                        "mu": mu[i],
                    }
                )
        df = pd.DataFrame(data)
        return df.sort_values(["Component", "mu_star"], ascending=[True, False])

    def print_summary(self):
        """Print comprehensive summary for all outputs."""
        print("SENSITIVITY SUMMARY - ALL OUTPUTS")

        for comp in self.combined_df["Component"].unique():
            print(f"\nOUTPUT: {comp.upper()}")
            print(
                f"{'Rank':<5} {'Parameter':<25} {'mu_star(Sensitivity)':<20} \
                    {'sigma (Interaction)':<20}"
            )
            df = self.combined_df[self.combined_df["Component"] == comp]
            for rank, (_, row) in enumerate(df.head(15).iterrows(), 1):
                print(
                    f"{rank:<5} {row['Parameter']:<25} {row['mu_star']:<20.6f} \
                        {row['sigma']:<20.6f}"
                )

    def plot_component_comparison(self, save_path=None):
        """Compare top parameters across all outputs."""
        if not hasattr(self, "combined_df"):
            self.export_all_results()

        components = [c for c in self.output_vars if c in self.combined_df["Component"].unique()]
        fig, axes = plt.subplots(1, len(components), figsize=(5 * len(components), 8))
        if len(components) == 1:
            axes = [axes]

        for ax, comp in zip(axes, components, strict=True):
            df = self.combined_df[self.combined_df["Component"] == comp].nlargest(10, "mu_star")
            colors = plt.get_cmap("tab10")(df["mu_star"] / df["mu_star"].max())
            ax.barh(range(len(df)), df["mu_star"], color=colors)
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df["Parameter"], fontsize=8)
            ax.set_xlabel(r"$\mu^*$", fontsize=10)
            ax.set_title(comp.upper(), fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")

        plt.suptitle(
            "Morris Sensitivity: Parameter Rankings by Output", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def export_all_results(self, save_dir: str = "morris_results"):
        """Export all results to CSV files."""
        os.makedirs(save_dir, exist_ok=True)

        combined_data = []
        for component_name, result in self.results.items():
            mu_star = np.asarray(result["mu_star"])
            sigma = np.asarray(result["sigma"])
            mu = np.asarray(result["mu"])
            for i, param_name in enumerate(result["names"]):
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
        self.combined_df.to_csv(f"{save_dir}/morris_outputs.csv", index=False)
        print(f"\n Results saved to {save_dir}/morris_outputs.csv")

        return self.combined_df


def run_morris_analysis(
    file_path: str,
    calib_params: list[str],
    output_vars: list[str],
    param_bounds: dict[str, tuple[float, float]],
    param_best: dict[str, float],
    n_trajectories: int = 500,
    n_levels: int = 20,
    time_point: str = "final",
    save_plots: bool = True,
    export_csv: bool = True,
    save_dir: str = "morris_results",
) -> MorrisSensitivityOnOutputs:
    """Run Morris sensitivity analysis on model outputs."""
    analyzer = MorrisSensitivityOnOutputs(
        file_path=file_path,
        calib_params=calib_params,
        output_vars=output_vars,
        param_bounds=param_bounds,
        param_best=param_best,
        n_levels=n_levels,
        n_trajectories=n_trajectories,
        time_point=time_point,
    )

    analyzer.run_analysis()
    analyzer.export_all_results(save_dir=save_dir)
    analyzer.print_summary()

    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        analyzer.plot_component_comparison(save_path=f"{save_dir}/morris_comparison_outputs.png")

    if export_csv:
        analyzer.export_all_results(save_dir)

    return analyzer


if __name__ == "__main__":
    file_path = "./data/solling_data.xlsx"

    # Get parameter bounds
    param_bounds_df = pd.read_excel(file_path, sheet_name="param_bound")
    param_bounds = {}
    param_best = {}
    for _, row in param_bounds_df.iterrows():
        if pd.notna(row["min"]) and pd.notna(row["max"]):
            param_bounds[row["param_name"]] = (row["min"], row["max"])
            param_best[row["param_name"]] = row["default"]

    # Run analysis on model outputs
    analyzer = run_morris_analysis(
        file_path=file_path,
        calib_params=[
            "alphaCx",
            "CoeffCond",
            "Y",
        ],
        output_vars=["WS", "DBH"],
        param_bounds=param_bounds,
        param_best=param_best,
        n_trajectories=5,
        n_levels=20,
        time_point="final",  # or 'mean' or index
        save_plots=True,
        export_csv=True,
        save_dir=os.path.join("./data/", "morris_analysis_results"),
    )
