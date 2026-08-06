"""Allometric biomass and leaf area equations for European tree species.

Implements Forrester et al. (2017) equation 3 (DBH-only model).

Equation form
-------------
``W = CF * exp(a0) * DBH^a1``

where DBH is in cm, W is in kg tree⁻¹ (biomass) or m² tree⁻¹ (leaf area),
and CF is the log-normal back-transformation correction factor (Table A.5).

Reference
---------
Forrester, D.I. et al. (2017). Generalized biomass and leaf area allometric
equations for European tree species incorporating stand structure, tree age
and climate. Forest Ecology and Management, 396, 160-175.
"""

import logging
import math
import os

import openpyxl
import polars as pl

from trunx.config import project_root

logger = logging.getLogger(__name__)

# biomass component labels mapped to the "Component" strings in Table A.5.
COMPONENT_NAMES: dict[str, str] = {
    "sb": "Stem mass",
    "fb": "Foliage mass",
    "rb": "Root mass",
    "la": "Leaf area",
}

_DEFAULT_XLSX = os.path.join(project_root, "literature", "1-s2.0-S0378112717301238-mmc1.xlsx")

# Type alias: {species: {component_label: (a0, a1, CF)}}
CoefficientsDict = dict[str, dict[str, tuple[float, float, float]]]


def load_forrester_eq3(
    xlsx_path: str = _DEFAULT_XLSX,
    species: list[str] | None = None,
    equation: int = 3,
) -> CoefficientsDict:
    """Load allometric coefficients from Forrester et al. (2017) Table A.5.

    Parameters
    ----------
    xlsx_path : str
        Path to the Forrester 2017 supplementary Excel file.  Defaults to the
        copy in the project's ``literature/`` folder.
    species : list[str] | None
        Species names to extract.  ``None`` loads every species in the sheet
        that has a complete set of components for the requested equation.
    equation : int
        Equation number in Table A.5.  3 is the DBH-only model (default).

    Returns
    -------
    CoefficientsDict
        ``{species: {component_label: (a0, a1, CF)}}``.
    """
    target_species = set(species) if species is not None else None
    name_to_label = {v: k for k, v in COMPONENT_NAMES.items()}

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["Table A.5"]

    raw: CoefficientsDict = {}

    for row in ws.iter_rows(values_only=True):
        sp, component, eq = row[0], row[1], row[2]
        if (
            not isinstance(sp, str)
            or not isinstance(component, str)
            or not isinstance(eq, int)
            or component not in name_to_label
            or eq != equation
        ):
            continue
        if target_species is not None and sp not in target_species:
            continue

        a0_raw, a1_raw, cf = row[3], row[4], row[11]
        if (
            not isinstance(a0_raw, str)
            or not isinstance(a1_raw, str)
            or not isinstance(cf, (int, float))
        ):
            continue

        # Cells are formatted as "value  (std_error)"; take the leading token.
        a0 = float(a0_raw.split()[0])
        a1 = float(a1_raw.split()[0])
        raw.setdefault(sp, {})[name_to_label[component]] = (a0, a1, float(cf))

    # Retain only species with all four components; warn and skip the rest.
    required = set(COMPONENT_NAMES)
    result: CoefficientsDict = {}
    for sp, comps in raw.items():
        missing = required - set(comps)
        if missing:
            logger.warning(
                "'%s': missing components %s for equation %d — skipped", sp, missing, equation
            )
        else:
            result[sp] = comps

    if target_species is not None:
        for sp in target_species:
            if sp not in result:
                raise ValueError(
                    f"No complete coefficients found for '{sp}' (equation {equation})"
                )

    return result


def add_allometric_columns(
    df: pl.DataFrame,
    coefficients: CoefficientsDict,
    dbh_col: str = "dbh_cm",
    species_col: str = "specie",
) -> pl.DataFrame:
    """Add per-tree allometric biomass and leaf area columns.

    Parameters
    ----------
    df : pl.DataFrame
        Input data containing DBH and species columns.
    coefficients : CoefficientsDict
        Coefficients as returned by :func:`load_forrester_eq3`.
    dbh_col : str
        Name of the DBH column (values in cm).
    species_col : str
        Name of the species column.

    Returns
    -------
    pl.DataFrame
        Input with additional columns `allo_sb_kg`, `allo_fb_kg`,
        `allo_rb_kg` (kg tree⁻¹) and `allo_la_m2` (m² tree⁻¹).
        Rows whose species is absent from ``coefficients`` receive ``null``.
    """

    def _expr(component: str) -> pl.Expr:
        expr: pl.Expr = pl.lit(None).cast(pl.Float64)
        for sp, comps in coefficients.items():
            a0, a1, cf = comps[component]
            term = pl.lit(cf * math.exp(a0)) * pl.col(dbh_col).pow(a1)
            expr = pl.when(pl.col(species_col) == sp).then(term).otherwise(expr)
        return expr

    return df.with_columns(
        _expr("sb").alias("allo_sb_kg"),
        _expr("fb").alias("allo_fb_kg"),
        _expr("rb").alias("allo_rb_kg"),
        _expr("la").alias("allo_la_m2"),
    )
