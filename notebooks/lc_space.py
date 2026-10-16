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
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell
def _(mo):
    # Weights of 1 + tanh(w00 + w10*N + w01*S + w11*N*S)
    w00 = mo.ui.slider(-2.0, 2.0, step=0.05, value=0.1951, label="w00", show_value=True)
    w10 = mo.ui.slider(-1.0, 1.0, step=0.02, value=-0.050, label="w10", show_value=True)
    w01 = mo.ui.slider(-1.0, 1.0, step=0.02, value=0.570, label="w01", show_value=True)
    w11 = mo.ui.slider(-0.5, 0.2, step=0.005, value=-0.32, label="w11", show_value=True)
    mo.hstack([w00, w10, w01, w11])
    return w00, w01, w10, w11


@app.cell
def _(mo, np, plt, w00, w01, w10, w11):
    _N, _S = np.meshgrid(np.linspace(0, 6, 200), np.linspace(0, 5, 200))
    _func = 1 + np.tanh(w00.value + w10.value * _N + w01.value * _S + w11.value * _N * _S)

    weights_fig, _ax = plt.subplots(figsize=(6, 5), layout="constrained")
    _surface = _ax.contourf(_N, _S, _func, levels=np.linspace(0, 2, 22), cmap="RdBu")
    _ax.contour(_N, _S, _func, levels=[1.0], colors="black", linewidths=0.8)
    _ax.set_xlabel("N")
    _ax.set_ylabel("S")
    weights_fig.colorbar(_surface, ax=_ax, label="Modifier (1 = no effect)")
    _ax.set_title("1 + tanh(w00 + w10·N + w01·S + w11·N·S)")
    mo.output.replace(weights_fig)
    return


if __name__ == "__main__":
    app.run()
