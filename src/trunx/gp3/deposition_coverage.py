"""Tabulate where the monthly deposition of each plot's 3PG input comes from.

For every month of a plot's simulated period, the deposition is classified as measured,
filled while building the monthly deposition table (rolling mean or same month in
adjacent years), filled with the same-calendar-month mean of the nearest 5 years of the
plot's record (`create_combined_inputs.add_deposition_to_weather`), or left missing
(plots without any deposition record, or months the 5-year fill can't reach), where the
nutrition modifier is neutral. The shares are given in % of the months.
"""

import os

import polars as pl

from trunx.config import clean_data_folder, results_data_folder
from trunx.datasets.icp_level2_data import _load_deposition_periods
from trunx.datasets.icp_process_monthly_depositions import aggregate_monthly_deposition
from trunx.gp3.age_regression import fit_models
from trunx.gp3.bayesiancalibrations.bayesian_config import species_plot_ids
from trunx.gp3.create_combined_inputs import build_plot_input_sheets, load_plot_tables

SHARE_COLUMNS = [
    "% measured",
    "% missing",
    "% filled in monthly file",
    "% filled 5-yr calendar mean",
    "% left missing (neutral modifier)",
]
MIN_MEASURED_SHARE = 50.0  # % of the simulated months with measured deposition
# Mean N deposition (kg ha⁻¹ month⁻¹) bounds of the ranges each needing training plots
DEPOSITION_RANGE_EDGES = (0.5, 1.0, 1.5, 2.0)


def plot_deposition_coverage(
    plot_id: str,
    monthly: pl.DataFrame,
    measured: pl.DataFrame,
    age_models: dict[str, tuple[float, float]],
    variables: tuple[str, ...] = ("dep_n_tot",),
) -> dict[str, str | int | float]:
    """Share of each deposition origin over one ICP plot's simulated months.

    With several `variables`, a month counts as measured, in the monthly file or in the
    input only when all of them are.

    Parameters
    ----------
    plot_id : str
        ICP plot identifier.
    monthly : pl.DataFrame
        Monthly deposition table (`icp_monthly_deposition.parquet`).
    measured : pl.DataFrame
        Monthly deposition before any filling, with `plot_id`, `year`, `month` and
        `variables` columns.
    age_models : dict[str, tuple[float, float]]
        Per-species age-vs-DBH models, see `create_combined_inputs.build_plot_input_sheets`.
    variables : tuple[str, ...]
        Deposition variables, among `"dep_n_tot"` and `"dep_s_so4"`.

    Returns
    -------
    dict[str, str | int | float]
        `plot_id`, `months`, one entry per `SHARE_COLUMNS` item, in %, and `mean <var>`,
        the mean measured monthly deposition of each of `variables`.
    """
    climate = pl.from_pandas(
        build_plot_input_sheets(
            plot_id,
            "ICP",
            *load_plot_tables(plot_id, "ICP"),
            age_models=age_models,
            deposition=monthly,
        )["climate"]
    ).with_columns(pl.col("year", "month").cast(pl.Int64))

    def select_variables(df: pl.DataFrame, name: str) -> pl.DataFrame:
        """Select `variables` from `df` suffixed with `name`, keyed by year and month."""
        return df.select(
            pl.col("year", "month").cast(pl.Int64),
            *[pl.col(var).alias(f"{var}_{name}") for var in variables],
        )

    def plot_rows(df: pl.DataFrame, name: str) -> pl.DataFrame:
        """Select the plot's `variables` from `df` suffixed with `name`."""
        return select_variables(df.filter(pl.col("plot_id") == plot_id), name)

    months = (
        select_variables(climate, "input")
        .join(plot_rows(monthly, "in_file"), on=["year", "month"], how="left")
        .join(plot_rows(measured, "measured"), on=["year", "month"], how="left")
    )

    def all_present(name: str) -> pl.Series:
        """Whether all `variables` suffixed with `name` are present in each month."""
        return months.select(
            pl.all_horizontal(pl.col(f"{var}_{name}").is_not_null() for var in variables)
        ).to_series()

    is_measured = all_present("measured")
    in_file = all_present("in_file")
    # Months outside the monthly file get the 5-year calendar mean where it reaches them,
    # and stay missing otherwise
    in_input = all_present("input")
    five_year = ~in_file & in_input
    left_missing = ~in_input

    def share(mask: pl.Series) -> float:
        """Percentage of the plot's months where `mask` is true."""
        return round(100 * int(mask.sum()) / months.height, 1)

    return {
        "plot_id": plot_id,
        "months": months.height,
        "% measured": share(is_measured),
        "% missing": share(~is_measured),
        "% filled in monthly file": share(in_file & ~is_measured),
        "% filled 5-yr calendar mean": share(five_year),
        "% left missing (neutral modifier)": share(left_missing),
        **{f"mean {var}": months.select(pl.mean(f"{var}_measured")).item() for var in variables},
    }


def deposition_coverage(
    plot_ids_by_species: dict[str, list[str]], variables: tuple[str, ...] = ("dep_n_tot",)
) -> pl.DataFrame:
    """Tabulate the deposition origin shares of every plot.

    Parameters
    ----------
    plot_ids_by_species : dict[str, list[str]]
        ICP plot identifiers per species, e.g. `bayesian_config.species_plot_ids`.
    variables : tuple[str, ...]
        Deposition variables, among `"dep_n_tot"` and `"dep_s_so4"`; see
        `plot_deposition_coverage`.

    Returns
    -------
    pl.DataFrame
        One row per plot: `species`, `plot_id`, `months` and the `SHARE_COLUMNS`.
    """
    monthly = pl.read_parquet(os.path.join(clean_data_folder, "icp_monthly_deposition.parquet"))
    measured = aggregate_monthly_deposition(*_load_deposition_periods())
    age_models = fit_models()
    return pl.DataFrame(
        [
            {
                "species": species,
                **plot_deposition_coverage(plot_id, monthly, measured, age_models, variables),
            }
            for species, plot_ids in plot_ids_by_species.items()
            for plot_id in plot_ids
        ]
    )


def summarize_by_species(coverage: pl.DataFrame) -> pl.DataFrame:
    """Month-weighted origin shares per species.

    Parameters
    ----------
    coverage : pl.DataFrame
        Output of `deposition_coverage`.

    Returns
    -------
    pl.DataFrame
        One row per species: `species`, `plots`, `months` and the `SHARE_COLUMNS`.
    """
    return coverage.group_by("species", maintain_order=True).agg(
        pl.len().alias("plots"),
        pl.col("months").sum(),
        *[
            ((pl.col(col) * pl.col("months")).sum() / pl.col("months").sum()).round(1).alias(col)
            for col in SHARE_COLUMNS
        ],
    )


def plots_per_deposition_range(
    coverage: pl.DataFrame,
    variable: str = "dep_n_tot",
    edges: tuple[float, ...] = DEPOSITION_RANGE_EDGES,
) -> pl.DataFrame:
    """Count the plots in each range of mean measured deposition.

    Parameters
    ----------
    coverage : pl.DataFrame
        Output of `deposition_coverage`, with a `mean <variable>` column.
    variable : str
        Deposition variable whose mean defines the ranges.
    edges : tuple[float, ...]
        Bounds between the ranges, in kg ha⁻¹ month⁻¹.

    Returns
    -------
    pl.DataFrame
        One row per species and range: `species`, `range`, `plots` and `plot_ids`.
    """
    return (
        coverage.with_columns(pl.col(f"mean {variable}").cut(list(edges)).alias("range"))
        .group_by("species", "range")
        .agg(pl.len().alias("plots"), pl.col("plot_id").sort().alias("plot_ids"))
        .sort("species", "range")
    )


if __name__ == "__main__":
    variables = ("dep_n_tot", "dep_s_so4")

    icp_df = pl.read_parquet(os.path.join(clean_data_folder, "icp_tree_data.parquet"))
    from trunx.datasets.icp_plot_selection import select_icp_plots

    selected_plots = select_icp_plots(icp_df)

    species_plot_ids = {}
    plot_ids = (
        selected_plots.group_by("specie")
        .agg(
            n_plots=pl.col("plot_id").n_unique(),
            plot_ids=pl.col("plot_id").unique(),
        )
        .filter(pl.col("specie").is_in(["Picea abies"]))
    )
    for idx in range(plot_ids.height):
        species_plot_ids[plot_ids["specie"][idx]] = list(plot_ids["plot_ids"][idx])

    print(species_plot_ids)

    coverage = deposition_coverage(species_plot_ids, variables=variables)
    # summary = summarize_by_species(coverage)

    # # output_dir = os.path.join(results_data_folder, "deposition_coverage")
    # # os.makedirs(output_dir, exist_ok=True)
    # # coverage.write_csv(os.path.join(output_dir, f"{variable}_by_plot.csv"))
    # # summary.write_csv(os.path.join(output_dir, f"{variable}_by_species.csv"))

    coverage = coverage.filter(pl.col("% measured") >= MIN_MEASURED_SHARE)
    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=200,
        fmt_table_cell_list_len=-1,
        fmt_str_lengths=200,
    ):
        print(coverage)
        print(plots_per_deposition_range(coverage))
    #     print(summary)
    # print(f"Saved to {output_dir}")
    print(list(coverage["plot_id"]))
