"""Compare Solling's recorded biomass against biomass derived from its DBH.

Applies the Forrester et al. (2017) eq. 3 allometric equation to the plot's
mean DBH and stem count to derive stem/foliage/root biomass, then plots it
against the biomass already given in `solling_data.xlsx`'s `observed` sheet.
"""

import math
import os

import matplotlib.pyplot as plt
import polars as pl

from trunx.config import threepg_data_folder
from trunx.gp3.allometrics import load_forrester_eq3

COMPONENT_TO_COLUMN = {"sb": "WS", "fb": "WF", "rb": "WR"}
COMPONENT_LABELS = {"sb": "Stem biomass", "fb": "Foliage biomass", "rb": "Root biomass"}


def derive_biomass_from_dbh(observed: pl.DataFrame, species_name: str) -> pl.DataFrame:
    """Derive per-hectare biomass from mean DBH and stem count.

    Parameters
    ----------
    observed : pl.DataFrame
        Plot-level observations with `DBH` (cm, mean tree) and `N` (stems/ha).
    species_name : str
        Species to load Forrester allometric coefficients for.

    Returns
    -------
    pl.DataFrame
        `observed` with one `derived_<column>` (t/ha) per biomass component.
    """
    coefficients = load_forrester_eq3(species=[species_name])[species_name]

    for component, column in COMPONENT_TO_COLUMN.items():
        a0, a1, cf = coefficients[component]
        per_tree_kg = cf * math.exp(a0) * pl.col("DBH").pow(a1)
        per_ha_t = per_tree_kg * pl.col("N") / 1000.0
        observed = observed.with_columns(per_ha_t.alias(f"derived_{column}"))

    return observed


def plot_given_vs_derived(observed: pl.DataFrame, output_path: str) -> None:
    """Plot given vs. DBH-derived biomass for each component, saved to `output_path`."""
    dates = observed["date"].to_list()

    _fig, axes = plt.subplots(1, len(COMPONENT_TO_COLUMN), figsize=(15, 4))
    for ax, (component, column) in zip(axes, COMPONENT_TO_COLUMN.items(), strict=True):
        ax.plot(dates, observed[column], "o-", label="Given")
        ax.plot(dates, observed[f"derived_{column}"], "s--", label="Derived from DBH")
        ax.set_title(COMPONENT_LABELS[component])
        ax.set_xlabel("Date")
        ax.set_ylabel("Biomass (t/ha)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    file_path = os.path.join(threepg_data_folder, "solling_data.xlsx")

    species_df = pl.read_excel(file_path, sheet_name="species")
    species_name = species_df["species"][0]

    observed_df = pl.read_excel(file_path, sheet_name="observed").sort("date")
    observed_df = derive_biomass_from_dbh(observed_df, species_name)

    print(observed_df.select("date", "WS", "derived_WS", "WF", "derived_WF", "WR", "derived_WR"))

    output_path = os.path.join(threepg_data_folder, "solling_given_vs_derived_biomass.png")
    plot_given_vs_derived(observed_df, output_path)
