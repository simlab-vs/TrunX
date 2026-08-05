"""Combine climate and deposition data into a single monthly table for 3PG input."""

import logging
import os

import jax.numpy as jnp
import polars as pl

from trunx.config import threepg_data_folder
from trunx.gp3.model_inputs import MonthlyData
from trunx.gp3.prepare_climate import prepare_climate
from trunx.gp3.prepare_deposition import prepare_deposition

logger = logging.getLogger(__name__)


def prepare_monthlydata(
    monthly_df: pl.DataFrame,
    from_: str,
    to: str,
):
    """Combine climate and deposition data into a single monthly class.

    Climate columns are prepared via :func:`prepare_climate`. Deposition
    columns ``dep_n_tot`` and ``dep_s_so4`` are fetched per plot via
    :func:`get_deposition_df` and joined by ``year`` and ``month``.
    Missing deposition rows are zero-filled.

    Parameters
    ----------
    monthly_df : pl.DataFrame
        Raw monthly data table with climate and deposition columns.
    from_ : str
        Simulation start month formatted as ``YYYY-MM``.
    to : str
        Simulation end month formatted as ``YYYY-MM``.

    """
    climate_df = prepare_climate(monthly_df, from_=from_, to=to)
    deposition_df = prepare_deposition(monthly_df, from_=from_, to=to)

    monthly_data = MonthlyData(
        T_avg=jnp.asarray(climate_df.T_avg, dtype=float),
        T_max=jnp.asarray(climate_df.T_max, dtype=float),
        VPD=jnp.asarray(climate_df.VPD, dtype=float),
        precip=jnp.asarray(climate_df.precip, dtype=float),
        solar_rad=jnp.asarray(climate_df.solar_rad, dtype=float),
        frost_days=jnp.asarray(climate_df.frost_days, dtype=float),
        n_days=jnp.asarray(climate_df.n_days, dtype=float),
        co2=jnp.asarray(climate_df.co2, dtype=float),
        d13catm=jnp.asarray(climate_df.d13catm, dtype=float),
        month=jnp.asarray(climate_df.month, dtype=int),
        dep_n_tot=jnp.asarray(deposition_df.dep_n_tot, dtype=float),
        dep_s_so4=jnp.asarray(deposition_df.dep_s_so4, dtype=float),
    )

    return monthly_data
