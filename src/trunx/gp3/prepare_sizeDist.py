"""Prepare sizeDist for 3PG model."""

from collections.abc import Iterable

import polars as pl


def prepare_sizeDist(
    i_sizeDist: pl.DataFrame,
    size_dist: pl.DataFrame | None = None,
    sp_names: Iterable[str] = ("Fagus sylvatica", "Pinus sylvestris"),
) -> pl.DataFrame:
    """
    Prepare the size distribution parameter table for 3PG.

    Parameters
    ----------
    i_sizeDist : pl.DataFrame
        Must contain columns: ["parameter", "default"]
    size_dist : pl.DataFrame | None
        Optional user-provided overrides.
        First column must be "parameter".
    sp_names : iterable of str
        Species / cohort names.

    Returns
    -------
    pl.DataFrame
        Parameter table with one column per species.
    """
    # Validate sp_names
    if sp_names is None or any(s is None for s in sp_names):
        raise ValueError("sp_names must be provided according to the species table.")

    # Prepare base output table
    size_dist_out = i_sizeDist.select("parameter").with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(sp) for sp in sp_names]
    )

    # Fill defaults for all species
    for sp in sp_names:
        size_dist_out = size_dist_out.with_columns(pl.col(sp).fill_null(i_sizeDist["default"]))

    # Apply user overrides
    if size_dist is not None:
        # Check first column name
        if size_dist.columns[0] != "parameter":
            raise ValueError("First column name of the size_dist table must be: 'parameter'")

        # Check parameter validity
        invalid_params = set(size_dist["parameter"]) - set(i_sizeDist["parameter"])
        if invalid_params:
            raise ValueError(
                "size_dist input table must contain only parameters present in "
                f"i_sizeDist: {', '.join(i_sizeDist['parameter'])}"
            )

        # Species columns to replace
        sp_names_replace = [sp for sp in sp_names if sp in size_dist.columns]

        if sp_names_replace:
            size_dist_out = size_dist_out.join(
                size_dist.select(["parameter"] + sp_names_replace),
                on="parameter",
                how="left",
                suffix="_new",
            )

            # Replace defaults with provided values
            for sp in sp_names_replace:
                size_dist_out = size_dist_out.with_columns(
                    pl.coalesce(
                        pl.col(f"{sp}_new"),
                        pl.col(sp),
                    ).alias(sp)
                )

            # Drop temporary columns
            size_dist_out = size_dist_out.drop([f"{sp}_new" for sp in sp_names_replace])

    return size_dist_out
