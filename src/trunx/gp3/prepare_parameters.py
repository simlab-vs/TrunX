"""Prepare model parameters for 3PG model."""

from collections.abc import Sequence

import polars as pl


def prepare_parameters(
    i_parameters: pl.DataFrame,
    parameters: pl.DataFrame | None = None,
    sp_names: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Set model parameters for 3PG model."""
    # Check species names
    if sp_names is None or len(sp_names) == 0 or any(x is None for x in sp_names):
        raise ValueError("sp_names must be provided according to the species table.")

    parameters_out = i_parameters.select(["parameter", "default"])

    parameters_out = parameters_out.with_columns(
        [pl.lit(None).cast(pl.Float64).alias(sp) for sp in sp_names]
    )

    parameters_out = parameters_out.with_columns([pl.col("default").alias(sp) for sp in sp_names])

    if parameters is not None:
        if parameters.columns[0] != "parameter":
            raise ValueError(
                "First column name of the parameters table must correspond to: parameter"
            )

        invalid = parameters.select("parameter").join(
            i_parameters.select("parameter"),
            on="parameter",
            how="anti",
        )

        if invalid.height > 0:
            valid = ", ".join(i_parameters["parameter"].to_list())
            raise ValueError(
                "Parameter input table must contains only parameters present in: "
                f"{valid}. Check `param_info` for more details."
            )

        sp_names_replace = [sp for sp in sp_names if sp in parameters.columns]

        if sp_names_replace:
            parameters_out = (
                parameters_out.join(parameters, on="parameter", how="left", suffix="_user")
                .with_columns(
                    [
                        pl.coalesce(
                            pl.col(f"{sp}_user"),
                            pl.col(sp),
                        ).alias(sp)
                        for sp in sp_names_replace
                    ]
                )
                .drop([f"{sp}_user" for sp in sp_names_replace])
            )

    parameters_out = parameters_out.drop("default")

    return parameters_out


if __name__ == "__main__":
    i_parameters = pl.read_csv("./data/i_parameters.csv")

    parameters_out = prepare_parameters(i_parameters)

    print(parameters_out)
