"""Hybrid 3PG model: NN-enhanced nutrition modifier.

The standard modifier fN = 1 - (1-fN0)*(1-FR)^fNn depends only on the fertility
rating FR and is static over time. This module replaces it with:

    fN_nn = clip(f_N_physics(FR, fN0, fNn) * nn_correction(N_dep, S_dep), 0, 1)

where a small MLP learns multiplicative corrections from nitrogen and sulphur
deposition. The correction is applied via the alphaCx parameter so that model_step
requires no changes.

Placeholder deposition values are used until real measurements are connected;
see N_DEP_PLACEHOLDER and S_DEP_PLACEHOLDER.

Usage:
    uv run src/trunx/gp3/nn_3pg.py
"""

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import optax
from jax import Array

from trunx.gp3.helper_function import f_nutrition
from trunx.gp3.model_inputs import ClimateData, Params, SiteData, SpeciesData, State
from trunx.gp3.run_3pg import model_step

# ---------------------------------------------------------------------------
# Placeholder deposition values — swap for real data when available.
# ---------------------------------------------------------------------------
N_DEP_PLACEHOLDER: float = 10.0  # kg N / ha / yr (typical central-European value)
S_DEP_PLACEHOLDER: float = 5.0  # kg S / ha / yr

# Normalisation constants for MLP inputs.
_N_MEAN, _N_STD = 15.0, 10.0  # kg N / ha / yr
_S_MEAN, _S_STD = 6.0, 4.0  # kg S / ha / yr


class NutritionNNParams(NamedTuple):
    """Weights and biases for the two-layer nutrition correction MLP."""

    W1: Array  # (2, hidden)
    b1: Array  # (hidden,)
    W2: Array  # (hidden, 1)
    b2: Array  # (1,)


def init_nn_params(key: Array, hidden: int = 8) -> NutritionNNParams:
    """Initialise MLP weights so the correction equals 1.0 at startup.

    Parameters
    ----------
    key : Array
        JAX random key.
    hidden : int
        Number of hidden units.

    Returns
    -------
    NutritionNNParams
        Near-zero weight initialisation; biases set to zero.
    """
    k1, k2 = jax.random.split(key)
    W1 = jax.random.normal(k1, (2, hidden)) * 0.01
    b1 = jnp.zeros(hidden)
    W2 = jax.random.normal(k2, (hidden, 1)) * 0.01
    b2 = jnp.zeros(1)
    return NutritionNNParams(W1=W1, b1=b1, W2=W2, b2=b2)


def nutrition_net(nn_params: NutritionNNParams, N_dep: float, S_dep: float) -> Array:
    """Compute a multiplicative correction to fN from N and S deposition.

    Parameters
    ----------
    nn_params : NutritionNNParams
        Network weights.
    N_dep : float
        Nitrogen deposition (kg N / ha / yr).
    S_dep : float
        Sulphur deposition (kg S / ha / yr).

    Returns
    -------
    Array
        Scalar correction factor in (0.7, 1.3). Zero-init weights → 1.0.
    """
    x = jnp.array([(N_dep - _N_MEAN) / _N_STD, (S_dep - _S_MEAN) / _S_STD])
    x = jnp.tanh(x @ nn_params.W1 + nn_params.b1)  # (hidden,)
    x = (x @ nn_params.W2 + nn_params.b2)[0]  # scalar
    # Range (0.7, 1.3); tanh(~0) = 0 → correction = 1.0 at zero-init.
    return 1.0 + 0.3 * jnp.tanh(x)


def nn_model_step(
    state: State,
    climate_month: Array,
    params: Params,
    site: SiteData,
    species: SpeciesData,
    n_species: int,
    nn_params: NutritionNNParams,
    N_dep: float = N_DEP_PLACEHOLDER,
    S_dep: float = S_DEP_PLACEHOLDER,
) -> tuple[State, dict]:
    """Single model step with NN-corrected nutrition modifier.

    The NN correction is encoded via alphaCx so that model_step is unchanged:

        alpha_c = alphaCx_adj * fT * fF * fN_physics * phi * ...
                = alphaCx * (fN_nn / fN_physics) * fT * fF * fN_physics * phi * ...
                = alphaCx * fT * fF * fN_nn * phi * ...

    Parameters
    ----------
    state : State
        Current model state.
    climate_month : Array
        Flattened climate row from the scan stack.
    params : Params
        Base physics parameters.
    site : SiteData
        Site data.
    species : SpeciesData
        Species data.
    n_species : int
        Number of species.
    nn_params : NutritionNNParams
        Trained (or freshly initialised) MLP weights.
    N_dep : float
        Nitrogen deposition (kg N / ha / yr).
    S_dep : float
        Sulphur deposition (kg S / ha / yr).

    Returns
    -------
    tuple[State, dict]
        Updated state and output dictionary identical in structure to model_step.
    """
    fN_physics = f_nutrition(species, params)  # shape (n_species,)
    correction = nutrition_net(nn_params, N_dep, S_dep)  # scalar
    fN_nn = jnp.clip(fN_physics * correction, 0.0, 1.0)  # shape (n_species,)

    # Ratio encodes how much fN changed; apply to alphaCx so model_step sees it.
    ratio = fN_nn / jnp.maximum(fN_physics, 1e-8)
    modified_params = params._replace(alphaCx=params.alphaCx * ratio)

    return model_step(state, climate_month, modified_params, site, species, n_species)


def run_nn_3pg(
    initial_state: State,
    climate: ClimateData,
    params: Params,
    site: SiteData,
    species: SpeciesData,
    n_species: int,
    nn_params: NutritionNNParams,
    N_dep: float = N_DEP_PLACEHOLDER,
    S_dep: float = S_DEP_PLACEHOLDER,
) -> tuple[State, dict]:
    """Run the hybrid NN-3PG model over the full climate record.

    Parameters
    ----------
    initial_state : State
        Initial model state.
    climate : ClimateData
        Full climate time series.
    params : Params
        Base physics parameters (unmodified).
    site : SiteData
        Site data.
    species : SpeciesData
        Species data.
    n_species : int
        Number of species.
    nn_params : NutritionNNParams
        MLP weights (static during the scan).
    N_dep : float
        Nitrogen deposition (kg N / ha / yr).
    S_dep : float
        Sulphur deposition (kg S / ha / yr).

    Returns
    -------
    tuple[State, dict]
        Final state and stacked output dictionary.
    """
    climate_stack = jnp.stack(
        [
            climate.T_avg,
            climate.T_max,
            climate.VPD,
            climate.precip,
            climate.solar_rad,
            climate.frost_days,
            climate.co2,
            climate.n_days,
            climate.month,
        ],
        axis=-1,
    )

    def step(state: State, climate_row: Array) -> tuple[State, dict]:
        return nn_model_step(
            state, climate_row, params, site, species, n_species, nn_params, N_dep, S_dep
        )

    return jax.lax.scan(step, initial_state, climate_stack)


def train_nn_3pg(
    base_params: Params,
    initial_state: State,
    climate: ClimateData,
    site: SiteData,
    species: SpeciesData,
    n_species: int,
    obs_WS: Array,
    obs_times: Array,
    N_dep: float = N_DEP_PLACEHOLDER,
    S_dep: float = S_DEP_PLACEHOLDER,
    n_epochs: int = 500,
    lr: float = 1e-3,
    seed: int = 0,
    log_every: int = 50,
) -> tuple[NutritionNNParams, list[float]]:
    """Train the nutrition NN to minimise MSE on observed stem biomass.

    Parameters
    ----------
    base_params : Params
        Fixed physics parameters from the original 3PG model.
    initial_state : State
        Initial model state.
    climate : ClimateData
        Full climate time series.
    site : SiteData
        Site data.
    species : SpeciesData
        Species data.
    n_species : int
        Number of species.
    obs_WS : Array
        Observed stem biomass (Mg/ha), shape (n_obs, n_species).
    obs_times : Array
        Integer timestep indices corresponding to each row of obs_WS.
    N_dep : float
        Nitrogen deposition (kg N / ha / yr).
    S_dep : float
        Sulphur deposition (kg S / ha / yr).
    n_epochs : int
        Number of gradient steps.
    lr : float
        Adam learning rate.
    seed : int
        Random seed for weight initialisation.
    log_every : int
        Print loss every this many epochs.

    Returns
    -------
    NutritionNNParams
        Trained network weights.
    list[float]
        Loss values recorded at each log_every epoch.
    """
    key = jax.random.PRNGKey(seed)
    nn_params = init_nn_params(key)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(nn_params)

    def loss_fn(nn_p: NutritionNNParams) -> Array:
        _, outputs = run_nn_3pg(
            initial_state, climate, base_params, site, species, n_species, nn_p, N_dep, S_dep
        )
        pred_WS = outputs["WS"][obs_times]
        return jnp.mean((pred_WS - obs_WS) ** 2)

    step_fn = jax.jit(jax.value_and_grad(loss_fn))

    losses: list[float] = []
    for epoch in range(n_epochs):
        loss_val, grads = step_fn(nn_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        nn_params = cast(NutritionNNParams, optax.apply_updates(nn_params, updates))

        if epoch % log_every == 0:
            losses.append(float(loss_val))
            print(f"Epoch {epoch:4d}: loss = {float(loss_val):.6f}")

    return nn_params, losses


if __name__ == "__main__":
    import os

    from trunx.config import project_root, threepg_data_folder
    from trunx.gp3.PG3_model_impl import prepare_data
    from trunx.gp3.run_3pg import run_3pg

    os.chdir(project_root)

    file_path = os.path.join(threepg_data_folder, "solling_data.xlsx")
    initial_state, climate, params, site_data, species_data, n_species, species_names = (
        prepare_data(file_path)
    )

    # --- Baseline: pure physics run ---
    _, base_outputs = run_3pg(initial_state, climate, params, site_data, species_data, n_species)
    baseline_WS = base_outputs["WS"]  # shape (T, n_species)

    # --- Sanity check: identity NN should reproduce the baseline exactly ---
    key = jax.random.PRNGKey(0)
    nn_params_init = init_nn_params(key)
    _, nn_outputs = run_nn_3pg(
        initial_state, climate, params, site_data, species_data, n_species, nn_params_init
    )
    max_dev = float(jnp.max(jnp.abs(nn_outputs["WS"] - baseline_WS)))
    print(f"Max deviation from baseline with identity NN: {max_dev:.6f}")

    # --- Inspect physics fN and what the NN correction gives ---
    fN_physics = f_nutrition(species_data, params)
    corr = nutrition_net(nn_params_init, N_DEP_PLACEHOLDER, S_DEP_PLACEHOLDER)
    print(f"\nPhysics fN per species: {fN_physics.tolist()}")
    print(f"NN correction (identity init): {float(corr):.4f}")
    print(f"fN_nn = fN_physics * correction: {(fN_physics * corr).tolist()}")

    # --- Synthetic training: teach the NN to match a +5 % perturbed target ---
    stride = 12  # annual observations
    obs_times = jnp.arange(0, baseline_WS.shape[0], stride)
    obs_WS = baseline_WS[obs_times] * 1.05  # fake 5 % upward bias

    trained_params, losses = train_nn_3pg(
        base_params=params,
        initial_state=initial_state,
        climate=climate,
        site=site_data,
        species=species_data,
        n_species=n_species,
        obs_WS=obs_WS,
        obs_times=obs_times,
        N_dep=N_DEP_PLACEHOLDER,
        S_dep=S_DEP_PLACEHOLDER,
        n_epochs=500,
        lr=1e-3,
    )

    trained_corr = nutrition_net(trained_params, N_DEP_PLACEHOLDER, S_DEP_PLACEHOLDER)
    print(
        f"\nTrained NN correction at N={N_DEP_PLACEHOLDER}, S={S_DEP_PLACEHOLDER}: "
        f"{float(trained_corr):.4f}"
    )
    print(f"Final loss: {losses[-1]:.6f}")
