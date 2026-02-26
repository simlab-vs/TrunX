"""Prepare thinning for 3PG model."""

import numpy as np
import polars as pl


def prepare_thinning(thinning: pl.DataFrame | None, sp_names: list[str]):
    """Prepare self thinning."""
    if sp_names is None or len(sp_names) == 0 or any(s is None for s in sp_names):
        raise ValueError("sp_names must be provided according to the species table.")

    n_sp = len(sp_names)
    sp_id = {name: i + 1 for i, name in enumerate(sp_names)}  # 1-based like R

    # No thinning
    if thinning is None:
        return np.full((1, 5, n_sp), np.nan, dtype=float)

    # Check columns
    required_cols = ["species", "age", "stems_n", "stem", "root", "foliage"]
    if thinning.columns != required_cols:
        raise ValueError(
            "Column names of the thinning table must correspond to: "
            "species, age, stems_n, stem, root, foliage"
        )

    # Range checks
    condition = (
        (pl.col("stem") < 0)
        | (pl.col("stem") > 5)
        | (pl.col("root") < 0)
        | (pl.col("root") > 5)
        | (pl.col("foliage") < 0)
        | (pl.col("foliage") > 5)
    )

    if thinning.select(condition.any()).item():
        raise ValueError("Thinning values for stem, root, foliage shall be in a range [0, 10]")

    # Keep only known species
    thinning = (
        thinning.filter(pl.col("species").is_in(sp_names))
        .with_columns(pl.col("species").replace(sp_id).cast(pl.Int64))
        .sort(["species", "age"])
    )

    # Count thinnings per species
    counts = thinning.group_by("species").len()

    n_man = counts.select(pl.col("len").max()).item()

    thinning = thinning.with_columns(pl.int_range(1, pl.len() + 1).over("species").alias("thin_n"))

    # Build full species × thinning grid
    grid = pl.DataFrame(
        {
            "species": np.repeat(np.arange(1, n_sp + 1), n_man),
            "thin_n": np.tile(np.arange(1, n_man + 1), n_sp),
        }
    ).with_columns(
        pl.col("species").cast(pl.Int64),
        pl.col("thin_n").cast(pl.Int64),
    )

    # Merge thinning data onto full grid
    thinning = grid.join(thinning, on=["species", "thin_n"], how="left").sort(
        ["species", "thin_n"]
    )

    # Allocate output array
    out = np.full((n_man, 5, n_sp), np.nan, dtype=float)
    vars_ = ["age", "stems_n", "stem", "root", "foliage"]

    for sp in range(1, n_sp + 1):
        out[:, :, sp - 1] = thinning.filter(pl.col("species") == sp).select(vars_).to_numpy()

    return out
