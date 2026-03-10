"""Prepare species data for 3PG model."""

import warnings

import polars as pl


def prepare_species(species: pl.DataFrame) -> pl.DataFrame:
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
    return species.select(required_cols)
