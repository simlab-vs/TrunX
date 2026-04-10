"""Gradient descent optimization for 3PG model parameters."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import grad, jit, value_and_grad
from tqdm import tqdm

from trunx.gp3.PG3_model_impl import prepare_data
from trunx.gp3.run_3pg import run_3pg

if __name__ == "__main__":
    file_path = "./data/solling_data.xlsx"
