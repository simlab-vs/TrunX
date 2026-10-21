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
    from typing import Any

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    from trunx.gp3.extended_helper import (
        CURVATURE_SHIFT,
        bivariate_quadratic_nm,
        liebig_nm,
        mlp_nm,
        poly_nm,
        product_nm,
    )
    from trunx.gp3.nn_nutrition_modifier import neutral_modifier_params

    return (
        Any,
        CURVATURE_SHIFT,
        TwoSlopeNorm,
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
    # Modifiers of scripts/run_loocv_sweep.py
    modifiers = {
        "poly_nm": poly_nm,
        "product_nm": product_nm,
        "liebig_nm": liebig_nm,
        "bivariate_quadratic_nm": bivariate_quadratic_nm,
        "mlp_nm": mlp_nm,
    }
    modifier = mo.ui.dropdown(list(modifiers), value="poly_nm", label="Nutrition modifier")
    inputs = mo.ui.dropdown(
        {"N": ("N",), "S": ("S",), "N and S": ("N", "S")}, value="N and S", label="Inputs"
    )
    mo.output.replace(mo.hstack([modifier, inputs], justify="start"))
    return inputs, modifier, modifiers


@app.cell
def _(inputs, jax, mo, modifier, modifiers, neutral_modifier_params, np):
    input_vars = inputs.value
    mo.stop(
        modifier.value == "bivariate_quadratic_nm" and len(input_vars) != 2,
        mo.md("`bivariate_quadratic_nm` needs two inputs, N and S"),
    )
    # Neutral weights of the modifier, the shape and starting value of its sliders
    neutral_params = neutral_modifier_params(modifiers[modifier.value], input_vars)
    _leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(neutral_params)]
    # Name of each weight, in the order of the flattened leaves
    if modifier.value == "poly_nm":
        _labels = [f"w{''.join(map(str, index))}" for index in np.ndindex(_leaves[0].shape)]
    elif modifier.value in ("product_nm", "liebig_nm"):
        # One row (b_k, w_k, raw a_k) per input, a_k = _curvature(raw a_k)
        _labels = [f"{weight}_{name}" for name in input_vars for weight in ("b", "w", "raw a")]
    elif modifier.value == "bivariate_quadratic_nm":
        _first, _second = input_vars
        _labels = [
            "b",
            f"w_{_first}",
            f"w_{_second}",
            f"raw a_{_first}",
            f"raw a_{_second}",
            "raw ρ",
        ]
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


@app.cell(hide_code=True)
def _(
    Any,
    CURVATURE_SHIFT,
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
    def _score(params: Any, inputs: jax.Array) -> tuple[np.ndarray, str]:
        """Evaluate the function of the modifier inside its 1 + tanh, with its formula.

        Not for product_nm and liebig_nm, which have one function per input, see
        `_per_input_scores`.
        """
        if modifier.value == "poly_nm":
            # Power of each input in the term of each weight, see `_cross_term_poly`
            score, terms = 0.0, []
            for index in np.ndindex(np.shape(params)):
                score = score + params[index] * jnp.prod(inputs ** jnp.asarray(index), axis=-1)
                term = "·".join(
                    name if power == 1 else f"{name}^{power}"
                    for name, power in zip(input_vars, index, strict=True)
                    if power
                )
                terms.append(f"w{''.join(map(str, index))}" + (f"·{term}" if term else ""))
            return np.asarray(score), " + ".join(terms)
        if modifier.value == "bivariate_quadratic_nm":
            # See `bivariate_quadratic_nm`
            z1, z2 = inputs[..., 0], inputs[..., 1]
            a = jax.nn.softplus(params[3:5] + CURVATURE_SHIFT)
            rho = jnp.tanh(params[5])
            quad = a[0] * z1**2 + a[1] * z2**2 - 2.0 * rho * jnp.sqrt(a[0] * a[1]) * z1 * z2
            return np.asarray(params[0] + inputs @ params[1:3] - 0.5 * quad), "b + wᵀz − ½·zᵀAz"
        # mlp_nm: tanh hidden layers and a linear output, see `mlp_nm`
        x = inputs
        *hidden_layers, output_layer = params.layers
        for layer in hidden_layers:
            x = jnp.tanh(x @ layer.w + layer.b)
        return (
            np.asarray((x @ output_layer.w + output_layer.b).squeeze(-1)),
            f"MLP({', '.join(input_vars)})",
        )

    def _per_input_scores(params: jax.Array, grid: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Evaluate s_k = b_k + w_k·z_k − ½·a_k·z_k² of each input over `grid`.

        The functions inside the 1 + tanh of product_nm and liebig_nm, one per input,
        see `per_nutrient_nm`.
        """
        a = jax.nn.softplus(params[:, 2] + CURVATURE_SHIFT)
        return [
            (name, np.asarray(params[k, 0] + params[k, 1] * grid - 0.5 * a[k] * grid**2))
            for k, name in enumerate(input_vars)
        ]

    # Slider values back into the weights of the modifier, leaf by leaf
    _values = iter(weights.value)
    modifier_params = jax.tree_util.tree_map(
        lambda leaf: jnp.asarray([next(_values) for _ in range(leaf.size)]).reshape(leaf.shape),
        neutral_params,
    )
    # Deposition (kg ha⁻¹ month⁻¹) along one axis per input
    _grid = np.linspace(0, 6, 400 if len(input_vars) == 1 else 200)
    _axis_values = np.meshgrid(*([_grid] * len(input_vars)))
    _inputs = jnp.stack(_axis_values, axis=-1)
    _func = np.asarray(modifiers[modifier.value](modifier_params, _inputs, input_vars))

    # Panels as (values, value range, no-effect value, value label, title), the modifier
    # first, then the function(s) inside its 1 + tanh
    if modifier.value in ("product_nm", "liebig_nm"):
        _scores = _per_input_scores(modifier_params, _grid)
        _limit = max(max(float(np.abs(s).max()) for _, s in _scores), 1e-3)
        _combine = "Π_k" if modifier.value == "product_nm" else "smooth min_k"
        _panels = [
            (_func, (0.0, 2.0), 1.0, "Modifier (1 = no effect)", f"{_combine} 1 + tanh(s_k)")
        ]
        _panels += [
            (
                s,
                (-_limit, _limit),
                0.0,
                f"s_{name} (0 = no effect)",
                f"s_{name} = b_{name} + w_{name}·{name} − ½·a_{name}·{name}²",
            )
            for name, s in _scores
        ]
    else:
        _inner, _inner_formula = _score(modifier_params, _inputs)
        _limit = max(float(np.abs(_inner).max()), 1e-3)
        _panels = [
            (_func, (0.0, 2.0), 1.0, "Modifier (1 = no effect)", f"1 + tanh({_inner_formula})"),
            (
                _inner,
                (-_limit, _limit),
                0.0,
                "Function inside 1 + tanh (0 = no effect)",
                _inner_formula,
            ),
        ]

    weights_fig, _axes = plt.subplots(
        1, len(_panels), figsize=(6 * len(_panels), 5), squeeze=False, layout="constrained"
    )
    for _index, (_ax, (_panel_values, _limits, _no_effect, _label, _title)) in enumerate(
        zip(_axes[0], _panels, strict=True)
    ):
        if _panel_values.ndim == 1:
            # A function of one input, as a curve over it: the per-input functions, or
            # any panel with a single input
            _name = input_vars[0] if len(input_vars) == 1 else input_vars[_index - 1]
            _ax.plot(_grid, _panel_values, color="tab:blue")
            _ax.axhline(_no_effect, color="black", linewidth=0.8)
            _ax.set_ylim(*_limits)
            _ax.set_xlabel(_name)
            _ax.set_ylabel(_label)
        else:
            _surface = _ax.contourf(
                *_axis_values, _panel_values, levels=np.linspace(*_limits, 22), cmap="RdBu"
            )
            _ax.contour(
                *_axis_values, _panel_values, levels=[_no_effect], colors="black", linewidths=0.8
            )
            _ax.set_xlabel(input_vars[0])
            _ax.set_ylabel(input_vars[1])
            weights_fig.colorbar(_surface, ax=_ax, label=_label)
        _ax.set_title(_title, fontsize="medium")
    weights_fig.suptitle(f"{modifier.value}, inputs {', '.join(input_vars)}")
    mo.output.replace(weights_fig)
    return (modifier_params,)


@app.cell(hide_code=True)
def _(
    CURVATURE_SHIFT,
    TwoSlopeNorm,
    input_vars,
    jax,
    mo,
    modifier,
    modifier_params,
    np,
    plt,
):
    def _linear_response(params: jax.Array, grid: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Evaluate 1 + s_k = 1 + b_k + w_k·z_k − ½·a_k·z_k² of each input over `grid`.

        The per-input modifier of `per_nutrient_nm` without its tanh, so neither bounded
        nor saturating.
        """
        a = jax.nn.softplus(params[:, 2] + CURVATURE_SHIFT)
        return [
            (name, np.asarray(1 + params[k, 0] + params[k, 1] * grid - 0.5 * a[k] * grid**2))
            for k, name in enumerate(input_vars)
        ]

    mo.stop(
        modifier.value not in ("product_nm", "liebig_nm"),
        mo.md("1 + b_k + w_k·z_k − ½·a_k·z_k² uses the weights of `product_nm` or `liebig_nm`"),
    )
    _grid = np.linspace(0, 6, 400)
    _responses = _linear_response(modifier_params, _grid)
    # With both inputs, also their combination, as in product_nm or liebig_nm
    _two_inputs = len(_responses) == 2
    linear_fig, _axes = plt.subplots(
        1,
        len(_responses) + _two_inputs,
        figsize=(6 * (len(_responses) + _two_inputs), 5),
        squeeze=False,
        layout="constrained",
    )
    for _ax, (_name, _response) in zip(_axes[0], _responses, strict=False):
        _ax.plot(_grid, _response, color="tab:blue", label=f"1 + s_{_name}")
        # The modifier with the same weights, for comparison
        _ax.plot(
            _grid,
            1 + np.tanh(_response - 1),
            color="tab:gray",
            linestyle="--",
            label=f"1 + tanh(s_{_name})",
        )
        _ax.axhline(1, color="black", linewidth=0.8)
        _ax.set_xlabel(_name)
        _ax.set_ylabel("Response (1 = no effect)")
        _ax.set_title(
            f"1 + b_{_name} + w_{_name}·{_name} − ½·a_{_name}·{_name}²", fontsize="medium"
        )
        _ax.legend(fontsize="small")
    if _two_inputs:
        (_, _response_N), (_, _response_S) = _responses
        # Response of N along the columns and of S along the rows
        _combined = (
            _response_S[:, None] * _response_N[None, :]
            if modifier.value == "product_nm"
            else np.minimum(_response_S[:, None], _response_N[None, :])
        )
        _ax = _axes[0, -1]
        # Unbounded, so the colors are centred on 1 (no effect) whatever the range
        _surface = _ax.contourf(
            _grid,
            _grid,
            _combined,
            levels=22,
            cmap="RdBu",
            norm=TwoSlopeNorm(
                vcenter=1.0,
                vmin=min(float(_combined.min()), 1.0 - 1e-3),
                vmax=max(float(_combined.max()), 1.0 + 1e-3),
            ),
        )
        _ax.contour(_grid, _grid, _combined, levels=[1.0], colors="black", linewidths=0.8)
        _ax.set_xlabel(input_vars[0])
        _ax.set_ylabel(input_vars[1])
        linear_fig.colorbar(_surface, ax=_ax, label="Response (1 = no effect)")
        _combine = "Π_k" if modifier.value == "product_nm" else "min_k"
        _ax.set_title(f"{_combine} (1 + s_k)", fontsize="medium")
    linear_fig.suptitle(f"{modifier.value} weights without tanh, inputs {', '.join(input_vars)}")
    mo.output.replace(linear_fig)
    return


if __name__ == "__main__":
    app.run()
