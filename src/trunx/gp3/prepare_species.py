"""Prepare species data for 3PG model."""

import os
import warnings

import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl

from trunx.config import SPECIES_INDICES, threepg_data_folder
from trunx.gp3.model_inputs import SpeciesData


def prepare_species(species: pl.DataFrame) -> SpeciesData:
    """Check the species data for consistency."""
    if not isinstance(species, pl.DataFrame):
        species = pl.DataFrame(species)

    required_cols = [
        "species",
        "planted",
        "fertility",
        "stems_n",
        "biom_stem",
        "biom_root",
        "biom_foliage",
    ]

    # Check column names and order (R uses identical())
    if species.columns != required_cols:
        raise ValueError(
            "Columns names of the species table must correspond to: "
            "species, planted, fertility, stems_n, biom_stem, biom_root, biom_foliage"
        )

    # Check for NA / null values
    if species.select(pl.any_horizontal(pl.all().is_null())).to_series().any():
        raise ValueError("Species table should not contain NAs")

    # Fertility range check
    if species.filter((pl.col("fertility") < 0) | (pl.col("fertility") > 1)).height > 0:
        raise ValueError("Fertility shall be within a range of [0:1]")

    # Non-negativity checks
    if species.filter(pl.col("stems_n") < 0).height > 0:
        raise ValueError("Stem number shall be greater than 0")

    if species.filter(pl.col("biom_stem") < 0).height > 0:
        raise ValueError("Biomass stem shall be greater than 0")

    if species.filter(pl.col("biom_root") < 0).height > 0:
        raise ValueError("Biomass root shall be greater than 0")

    if species.filter(pl.col("biom_foliage") < 0).height > 0:
        raise ValueError("Biomass foliage shall be greater than 0")

    # Plausibility warning
    if species.filter(pl.col("biom_stem") > 10000).height > 0:
        warnings.warn("Biomass stem > 10000, unplausible value!", UserWarning, stacklevel=2)

    # Return final table (unchanged, but explicitly selected)
    species = species.select(required_cols)
    species = species.with_columns(
        [
            pl.col("planted").str.split("-").list.get(0).cast(pl.Int32).alias("year_p"),
            pl.col("planted").str.split("-").list.get(1).cast(pl.Int32).alias("month_p"),
            pl.col("planted").str.to_datetime(format="%Y-%m").alias("planted"),
        ]
    )

    for species_name in species["species"]:
        if species_name not in SPECIES_INDICES:
            raise ValueError(f"Species '{species_name}' is not in the SPECIES_INDICES mapping.")

    species_data = SpeciesData(
        specie=jnp.asarray(
            [SPECIES_INDICES[species_name] for species_name in species["species"]], dtype=jnp.int32
        ),
        FR=jnp.asarray(species["fertility"], dtype=jnp.float32),
        WF=jnp.asarray(species["biom_foliage"], dtype=jnp.float32),
        WR=jnp.asarray(species["biom_root"], dtype=jnp.float32),
        WS=jnp.asarray(species["biom_stem"], dtype=jnp.float32),
        N=jnp.asarray(species["stems_n"], dtype=jnp.float32),
        # planted=tuple([np.datetime64(dt, "M") for dt in species["planted"].to_list()]),
        year_p=jnp.asarray(species["year_p"], dtype=jnp.int32),
        month_p=jnp.asarray(species["month_p"], dtype=jnp.int32),
    )

    return species_data


if __name__ == "__main__":
    species = pl.read_excel(
        os.path.join(threepg_data_folder, "data.input.xlsx"), sheet_name="species"
    )
    species_data = prepare_species(species)
    print(species_data)
