"""What explains the DBH growth residuals of 3PG: defoliation, deposition or climate."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    from trunx.gp3.extended_helper import (
        bivariate_quadratic_nm,
        liebig_nm,
        mlp_nm,
        poly_nm,
        product_nm,
    )
    from trunx.gp3.nn_nutrition_modifier import neutral_modifier_params

    return (
        bivariate_quadratic_nm,
        jax,
        jnp,
        liebig_nm,
        mlp_nm,
        neutral_modifier_params,
        np,
        plt,
        poly_nm,
        product_nm,
    )


@app.cell
def _(bivariate_quadratic_nm, liebig_nm, mlp_nm, mo, poly_nm, product_nm):
    # Modifiers of scripts/run_loocv_sweep.py, with the formula of each
    modifiers = {
        "poly_nm": (poly_nm, "1 + tanh(w00 + w10·N + w01·S + w11·N·S)"),
        "product_nm": (product_nm, "Π_k 1 + tanh(b_k + w_k·z_k − ½·a_k·z_k²)"),
        "liebig_nm": (liebig_nm, "smooth min_k 1 + tanh(b_k + w_k·z_k − ½·a_k·z_k²)"),
        "bivariate_quadratic_nm": (bivariate_quadratic_nm, "1 + tanh(b + wᵀz − ½·zᵀAz)"),
        "mlp_nm": (mlp_nm, "1 + tanh(MLP(N, S)), one tanh hidden layer of 4"),
    }
    modifier = mo.ui.dropdown(list(modifiers), value="poly_nm", label="Nutrition modifier")
    mo.output.replace(modifier)
    return modifier, modifiers


@app.cell
def _(jax, mo, modifier, modifiers, neutral_modifier_params, np):
    input_vars = ("N", "S")
    # Neutral weights of the modifier, the shape and starting value of its sliders
    neutral_params = neutral_modifier_params(modifiers[modifier.value][0], input_vars)
    _leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(neutral_params)]
    # Name of each weight, in the order of the flattened leaves
    if modifier.value == "poly_nm":
        _labels = [f"w{''.join(map(str, index))}" for index in np.ndindex(_leaves[0].shape)]
    elif modifier.value in ("product_nm", "liebig_nm"):
        # One row (b_k, w_k, raw a_k) per input, a_k = softplus(raw a_k − 5)
        _labels = [f"{weight}_{name}" for name in input_vars for weight in ("b", "w", "raw a")]
    elif modifier.value == "bivariate_quadratic_nm":
        _labels = ["b", "w_N", "w_S", "raw a_N", "raw a_S", "raw ρ"]
    else:
        _labels = [
            f"{jax.tree_util.keystr(path)}{list(index)}"
            for path, leaf in jax.tree_util.tree_flatten_with_path(neutral_params)[0]
            for index in np.ndindex(np.shape(leaf))
        ]
    # The fitted poly_nm weights this notebook started from, each with its own range
    _poly_sliders = {
        "w00": (-2.0, 2.0, 0.05, 0.1951),
        "w10": (-1.0, 1.0, 0.02, -0.050),
        "w01": (-1.0, 1.0, 0.02, 0.570),
        "w11": (-0.5, 0.2, 0.005, -0.32),
    }
    _values = np.concatenate([leaf.ravel() for leaf in _leaves])
    weights = mo.ui.array(
        [
            mo.ui.slider(
                *_poly_sliders.get(label, (-2.0, 2.0, 0.05, round(float(value), 2))),
                label=label,
                show_value=True,
            )
            for label, value in zip(_labels, _values, strict=True)
        ]
    )
    mo.output.replace(mo.hstack(list(weights), wrap=True, justify="start"))
    return input_vars, neutral_params, weights


@app.cell
def _(
    input_vars,
    jax,
    jnp,
    mo,
    modifier,
    modifiers,
    neutral_params,
    np,
    plt,
    weights,
):
    # Slider values back into the weights of the modifier, leaf by leaf
    _values = iter(weights.value)
    _params = jax.tree_util.tree_map(
        lambda leaf: jnp.asarray([next(_values) for _ in range(leaf.size)]).reshape(leaf.shape),
        neutral_params,
    )
    _modifier_fn, _formula = modifiers[modifier.value]
    _N, _S = np.meshgrid(np.linspace(0, 6, 200), np.linspace(0, 6, 200))
    _func = np.asarray(_modifier_fn(_params, jnp.stack([_N, _S], axis=-1), input_vars))

    weights_fig, _ax = plt.subplots(figsize=(6, 5), layout="constrained")
    _surface = _ax.contourf(_N, _S, _func, levels=np.linspace(0, 2, 22), cmap="RdBu")
    _ax.contour(_N, _S, _func, levels=[1.0], colors="black", linewidths=0.8)
    _ax.set_xlabel("N")
    _ax.set_ylabel("S")
    weights_fig.colorbar(_surface, ax=_ax, label="Modifier (1 = no effect)")
    _ax.set_title(f"{modifier.value}\n{_formula}", fontsize="medium")
    mo.output.replace(weights_fig)
    return


if __name__ == "__main__":
    app.run()
