"""Prepare species data for 3PG model."""

import warnings

import polars as pl
import pandas as pd
import numpy as np

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

    species_data: list[SpeciesData] = []

    for row in species.iter_rows(named=True):
        year_p, month_p = map(int, row["planted"].split("-"))

        sp = SpeciesData(
            FR=float(row["fertility"]),
            WF=float(row["biom_foliage"]),
            WR=float(row["biom_root"]),
            WS=float(row["biom_stem"]),
            N=float(row["stems_n"]),
            planted=np.datetime64(row["planted"], "M"),
        )

        species_data.append(sp)
    
    return species_data[0]


if __name__ == "__main__":
    species = pl.read_excel("./data/data.input.xlsx", sheet_name="species")
    species_data = prepare_species(species)
    print(species_data)

