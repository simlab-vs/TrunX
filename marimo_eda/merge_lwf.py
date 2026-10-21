"""Merge the nine LWF datasets into a single universal table.

LWF (Long-term Forest Ecosystem Research) provides the nine source datasets:
leaf area index (LAI), two foliage weight (dw100) datasets, monthly and period
deposition, and monthly and period litterfall for both the Davos ICOS and LWF
networks. Each dataset is cleaned and standardized, the standardized frames are
stacked, and rows that describe the same observation are combined.

The pipeline runs in three phases. Every source dataset has its own functions
for the first two phases, so each dataset can be read and changed on its own:

1. Preprocessing (``preprocess_<dataset>``): missing-value markers and infinities
   become null and every column is converted to its expected type. Values that
   cannot be converted become null instead of raising an error.
2. Removal of implausible values (``remove_implausible_<dataset>``): values
   outside the plausible ranges in PLAUSIBLE_RANGES become null. Rows that no
   longer hold the measurements the dataset requires are then dropped by
   ``remove_incomplete_<dataset>`` (see REQUIRED_COLUMNS).
3. Merging: exact duplicates are removed per dataset, the datasets are stacked
   and rows are combined on their strict key.

Every dataset defines its own natural key (see NATURAL_KEYS). Rows are only
combined when all key fields are present and identical, so a missing key field
never matches another row and incomplete identities stay separate. When
combined rows disagree on a measurement, the merged cell is left empty and the
competing values are kept in a ``<column>__CONFLICT`` column and in the
conflict report.

The following files are written to the output directory:

- LWF_universal_dataset.parquet: the merged table.
- LWF_merge_report.csv: row counts, cleaning counts and duplicate statistics
  per dataset.
- LWF_duplicate_keys_report.csv: rows flagged during duplicate detection.
- LWF_conflict_report.csv: cells where combined rows disagreed.

Run it from the command line as follows:

    python merge_lwf.py --data-dir ./data --out-dir ./data
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import polars as pl

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# CONFIG
# ============================================================================

# Switch for strict duplicate handling; disabled by default.
STRICT_MODE = False

# Strings in the raw CSV files that are read as missing values, in addition to
# the Polars defaults. The dot follows the SAS export convention used by the
# litterfall files.
EXTRA_NULL_VALUES = ["NA", "N/A", "."]

# Strings that are treated as missing after reading, whatever the reader did.
# Text and identifier columns match exactly (after trimming); numeric columns
# match case-insensitively. The two lists are separate so that a text value such
# as "None" is never lost, while a numeric cell reading "nan" or "null" is.
TEXT_MISSING_MARKERS = ["", *EXTRA_NULL_VALUES]
NUMERIC_MISSING_MARKERS = ["", "NA", "N/A", ".", "NAN", "NULL", "NONE"]

# Positive and negative infinity in any spelling or case, for numeric columns.
INFINITY_PATTERN = r"(?i)^[+-]?(inf|infinity)$"

# Crosswalk from location code to site name for the 19 LWF sites.
SITE_CROSSWALK: dict[str, str] = {
    "ALP": "Alptal",
    "BEA": "Beatenberg",
    "BET": "Bettlachstock",
    "CEL": "Celerina",
    "CHI": "Chironico",
    "DAV": "Davos",
    "ISO": "Isone",
    "JUS": "Jussy",
    "LAE": "Lägeren",
    "LAN": "Lantsch",
    "LAU": "Lausanne",
    "LEN": "Lens",
    "NAT": "Nationalpark",
    "NEU": "Neunkirch",
    "NOV": "Novaggio",
    "OTH": "Othmarsingen",
    "SCH": "Schänis",
    "VIS": "Visp",
    "VOR": "Vordemwald",
}

# Identity columns that appear across the datasets. Every standardized frame is
# padded to this full set, in this order, so the frames can be stacked with
# consistent types. Grouping uses the strict key rather than these columns
# directly, and site_name is left out because it is derived after coalescing.
IDENTITY_VOCAB_COLS: list[str] = [
    "location_code",
    "date",
    "start_date",
    "end_date",
    "survey_year",
    "tree_id",
    "sample_id",
    "species",
    "leaf_type",
    "leaf_age_class",
    "lai_subplot",
    "plot_type",
    "dep_location",
    "fraction_code",
    "sub_plot",
    "tree_species",
    "fraction_code_lwf",
    "fraction_code_lwf_detailed",
]

# Identity columns that hold dates; all other identity columns are strings.
DATE_VOCAB_COLS = {"date", "start_date", "end_date"}

# Natural key of each dataset. The columns are used to detect duplicates within
# a dataset and to build the key that rows are matched on across datasets, so
# only datasets with the same key columns and values ever merge into one row.
NATURAL_KEYS: dict[str, list[str]] = {
    "data/LAI_Licor_3_rings_all_years": ["location_code", "date", "lai_subplot", "plot_type"],
    "data/lwf_foliage_dw100_i": [
        "location_code",
        "date",
        "sample_id",
        "species",
        "leaf_type",
        "leaf_age_class",
    ],
    "data/lwf_foliage_dw100_plot": [
        "location_code",
        "date",
        "survey_year",
        "species",
        "leaf_type",
        "leaf_age_class",
    ],
    "data/monthly_dep_lwf": ["location_code", "start_date", "end_date", "dep_location"],
    "data/period_dep_lwf": ["location_code", "start_date", "end_date", "dep_location"],
    "data/monthly_litterfall_Davos_ICOS": [
        "location_code",
        "start_date",
        "end_date",
        "fraction_code",
    ],
    "data/monthly_litterfall_LWF": [
        "location_code",
        "start_date",
        "end_date",
        "sub_plot",
        "fraction_code_lwf",
        "tree_species",
    ],
    "data/period_litterfall_Davos_ICOS": [
        "location_code",
        "start_date",
        "end_date",
        "fraction_code",
    ],
    "data/period_litterfall_LWF": [
        "location_code",
        "start_date",
        "end_date",
        "sub_plot",
        "fraction_code_lwf",
        "fraction_code_lwf_detailed",
        "tree_species",
    ],
}

# Date formats tried, in order, when parsing a string date column.
CANDIDATE_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%d.%m.%Y",
    "%d/%m/%y",
    "%d/%m/%Y",
    "%Y/%m/%d",
]

# Names of the nine source datasets. They are the keys of NATURAL_KEYS.
DS_LAI = "data/LAI_Licor_3_rings_all_years"
DS_FOLIAGE_I = "data/lwf_foliage_dw100_i"
DS_FOLIAGE_PLOT = "data/lwf_foliage_dw100_plot"
DS_DEP_MONTHLY = "data/monthly_dep_lwf"
DS_DEP_PERIOD = "data/period_dep_lwf"
DS_LITTER_MONTHLY_DAVOS = "data/monthly_litterfall_Davos_ICOS"
DS_LITTER_MONTHLY_LWF = "data/monthly_litterfall_LWF"
DS_LITTER_PERIOD_DAVOS = "data/period_litterfall_Davos_ICOS"
DS_LITTER_PERIOD_LWF = "data/period_litterfall_LWF"
ALL_DATASETS = [
    DS_LAI,
    DS_FOLIAGE_I,
    DS_FOLIAGE_PLOT,
    DS_DEP_MONTHLY,
    DS_DEP_PERIOD,
    DS_LITTER_MONTHLY_DAVOS,
    DS_LITTER_MONTHLY_LWF,
    DS_LITTER_PERIOD_DAVOS,
    DS_LITTER_PERIOD_LWF,
]
assert set(ALL_DATASETS) == set(NATURAL_KEYS), "dataset names must match NATURAL_KEYS"

# Prefix given to the measurement columns of each dataset after standardization.
DATASET_PREFIX: dict[str, str] = {
    DS_LAI: "lai",
    DS_FOLIAGE_I: "foliage_i",
    DS_FOLIAGE_PLOT: "foliage_plot",
    DS_DEP_MONTHLY: "dep_monthly",
    DS_DEP_PERIOD: "dep_period",
    DS_LITTER_MONTHLY_DAVOS: "litter_monthly_davos",
    DS_LITTER_MONTHLY_LWF: "litter_monthly_lwf",
    DS_LITTER_PERIOD_DAVOS: "litter_period_davos",
    DS_LITTER_PERIOD_LWF: "litter_period_lwf",
}

# ----------------------------------------------------------------------------
# Expected column types (Step 1)
# ----------------------------------------------------------------------------
# Expected type of every source column, keyed by the column name as it is when
# the dataset is preprocessed (before the identity columns are renamed).
#   "str"   text or identifier; never converted to a number
#   "float" decimal number
#   "int"   whole number (a value such as 12.0 is accepted, 12.5 is not)
#   "date"  calendar date
# Columns that are not listed keep the type they were read with.
_DEPOSITION_TYPES: dict[str, str] = {
    name: "float"
    for name in [
        "precip",
        "conductivity",
        "ph",
        "alk",
        "nh4_n",
        "no3_n",
        "so4_s",
        "po4_p",
        "ca",
        "mg",
        "k",
        "na",
        "cl",
        "t_p",
        "t_s",
        "t_n",
        "toc",
    ]
}

TYPE_SCHEMAS: dict[str, dict[str, str]] = {
    DS_LAI: {
        "location_code": "str",
        "lai_subplot": "str",
        "date": "date",
        "plot_type": "str",
        "mean_LAI_Miller": "float",
        "standard_error_LAI_Miller": "float",
        "mean_LAI_NormanCampbell": "float",
        "standard_error_LAI_NormanCampbell": "float",
        "mean_angle_Miller": "float",
        "standard_error_angle_Miller": "float",
        "mean_angle_NormanCampbell": "float",
        "standard_error_angle_NormanCampbell": "float",
        "number_of_points": "int",
        "season": "str",
    },
    DS_FOLIAGE_I: {
        "plot_id": "str",
        "plot_name": "str",
        "survey_year": "int",
        "survey_date": "date",
        "sample_id": "str",
        "species": "str",
        "leaf_type": "str",
        "leaf_age_class": "str",
        "gew100": "float",
    },
    DS_FOLIAGE_PLOT: {
        "plot_id": "str",
        "plot_name": "str",
        "survey_year": "int",
        "survey_date": "date",
        "species": "str",
        "leaf_type": "str",
        "leaf_age_class": "str",
        "gew100": "float",
    },
    DS_DEP_MONTHLY: {
        "plot_id": "str",
        "plot_name": "str",
        "location": "str",
        "survey_year": "int",
        "survey_month": "int",
        **_DEPOSITION_TYPES,
    },
    DS_DEP_PERIOD: {
        "plot_id": "str",
        "plot_name": "str",
        "location": "str",
        "date_start": "date",
        "date_end": "date",
        **_DEPOSITION_TYPES,
    },
    DS_LITTER_MONTHLY_DAVOS: {
        "Year": "int",
        "Month": "int",
        "LWF_plot": "str",
        "LWF_plot_code": "str",
        "Start_date": "date",
        "End_date": "date",
        "Code_fraction": "str",
        "Network": "str",
        "Text_Fraction": "str",
        "Weight": "float",
    },
    DS_LITTER_MONTHLY_LWF: {
        "Year": "int",
        "Month": "int",
        "LWF_plot": "str",
        "LWF_plot_code": "str",
        "Start_date": "date",
        "End_date": "date",
        "Sub_plot": "str",
        "Code_fraction_2": "str",
        "Tree_species": "str",
        "Network": "str",
        "Number_of_days": "int",
        "Text_Fraction_2": "str",
        "Weight": "float",
    },
    DS_LITTER_PERIOD_DAVOS: {
        "LWF_plot": "str",
        "LWF_plot_code": "str",
        "Start_date": "date",
        "End_date": "date",
        "Code_fraction": "str",
        "Network": "str",
        "Period_id": "str",
        "Text_Fraction": "str",
        "Weight": "float",
    },
    DS_LITTER_PERIOD_LWF: {
        "LWF_plot": "str",
        "LWF_plot_code": "str",
        "Start_date": "date",
        "End_date": "date",
        "Sub_plot": "str",
        "Code_fraction_1": "str",
        "Code_fraction_2": "str",
        "Tree_species": "str",
        "Network": "str",
        "Number_of_days": "int",
        "Text_Fraction_1": "str",
        "Text_Fraction_2": "str",
        "Weight": "float",
    },
}

EXPECTED_DTYPES = {"str": pl.Utf8, "float": pl.Float64, "int": pl.Int64, "date": pl.Date}

# ----------------------------------------------------------------------------
# Plausible ranges (Step 2)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Bound:
    """Plausible range of one variable.

    ``lower`` and ``upper`` are inclusive. If ``strict_lower`` is True the lower
    bound is exclusive (value > lower), so the bound itself is rejected. A bound
    left as None is not enforced.
    """

    lower: float | None = None
    upper: float | None = None
    strict_lower: bool = False


# Bounds are applied exactly as supplied, to the values as they are stored in the
# source files. No unit conversion is done, and the source schema does not state
# units, so the same bound is not assumed to mean the same thing in two datasets.
# Out-of-range values are set to null, never dropped directly.

# A. LAI and canopy measurements: LAI_Licor_3_rings_all_years.xlsx.
# "season" is categorical and has no numerical range.
LAI_RANGES: dict[str, Bound] = {
    # mean_LAI_Miller: 0 <= value <= 8
    # https://bionumbers.hms.harvard.edu/bionumber.aspx?s=n&v=2&id=113232
    "mean_LAI_Miller": Bound(lower=0, upper=8),
    # mean_LAI_NormanCampbell: 0 <= value <= 8
    # https://bionumbers.hms.harvard.edu/bionumber.aspx?s=n&v=2&id=113232
    "mean_LAI_NormanCampbell": Bound(lower=0, upper=8),
    # standard_error_LAI_Miller: value > 0 (strict, zero is rejected), no upper bound
    # https://bionumbers.hms.harvard.edu/bionumber.aspx?s=n&v=2&id=113232
    "standard_error_LAI_Miller": Bound(lower=0, strict_lower=True),
    # standard_error_LAI_NormanCampbell: value > 0 (strict, zero is rejected), no upper bound
    # https://bionumbers.hms.harvard.edu/bionumber.aspx?s=n&v=2&id=113232
    "standard_error_LAI_NormanCampbell": Bound(lower=0, strict_lower=True),
    # mean_angle_Miller: 0 <= value <= 90
    # https://datadryad.org/dataset/doi:10.7280/D1T97H#methods
    "mean_angle_Miller": Bound(lower=0, upper=90),
    # mean_angle_NormanCampbell: 0 <= value <= 90
    # https://datadryad.org/dataset/doi:10.7280/D1T97H#methods
    "mean_angle_NormanCampbell": Bound(lower=0, upper=90),
    # standard_error_angle_Miller: value >= 0, no upper bound
    # https://datadryad.org/dataset/doi:10.7280/D1T97H#methods
    "standard_error_angle_Miller": Bound(lower=0),
    # standard_error_angle_NormanCampbell: value >= 0, no upper bound
    # https://datadryad.org/dataset/doi:10.7280/D1T97H#methods
    "standard_error_angle_NormanCampbell": Bound(lower=0),
    # number_of_points: value >= 0, no upper bound
    # https://datadryad.org/dataset/doi:10.7280/D1T97H#methods
    "number_of_points": Bound(lower=0),
}

# B. Foliage weight: lwf_foliage_dw100_i and lwf_foliage_dw100_plot.
FOLIAGE_RANGES: dict[str, Bound] = {
    # gew100: value > 0 (strict, zero is rejected), no upper bound
    # No source URL supplied; this is the user-specified lower bound.
    "gew100": Bound(lower=0, strict_lower=True),
}

# C. Atmospheric deposition: monthly_dep_lwf and period_dep_lwf. The same bounds
# apply to both datasets; they are validated separately and never combined first.
DEPOSITION_RANGES: dict[str, Bound] = {
    # precip: 0 <= value <= 5000
    # https://www.meteoswiss.admin.ch/weather/weather-and-climate-from-a-to-z/precipitation.html
    "precip": Bound(lower=0, upper=5000),
    # conductivity: 0 <= value <= 10000
    # https://atlas-scientific.com/blog/water-conductivity-range/?srsltid=AU7gw4W20O1YdSPpG5_uYOlBXKxlTjDOU-WnCuaICiQF0-JlPxma4QAN
    "conductivity": Bound(lower=0, upper=10000),
    # ph: 0 <= value <= 14
    # No source URL supplied; this is the user-specified range.
    "ph": Bound(lower=0, upper=14),
    # alk: 0 <= value <= 10
    # https://www.greenhousemag.com/article/production-pointers-treating-alkalinity/
    "alk": Bound(lower=0, upper=10),
    # nh4_n: 0 <= value <= 3
    # https://www.sciencedirect.com/science/article/pii/S0301479723019606
    "nh4_n": Bound(lower=0, upper=3),
    # no3_n: 0 <= value <= 30
    # https://archive.epa.gov/water/archive/web/html/vms57.html
    "no3_n": Bound(lower=0, upper=30),
    # so4_s: 0 <= value <= 1014.7
    # https://link.springer.com/article/10.1007/s11270-026-09732-6
    # The upper bound of 1014.7 is kept as specified. Rationale: it is expressed
    # as sulfur, converting sulphate mass to sulfur mass with the ratio
    # 32/96 = 1/3.
    "so4_s": Bound(lower=0, upper=1014.7),
    # po4_p: 0 <= value <= 20
    # https://www.lenntech.com/phosphorous-removal.htm
    "po4_p": Bound(lower=0, upper=20),
    # ca: 0 <= value <= 500
    # https://cdn.who.int/media/docs/default-source/wash-documents/wash-chemicals/hardness2003.pdf
    "ca": Bound(lower=0, upper=500),
    # mg: 0 <= value <= 40
    # https://wq.moenv.gov.tw/EWQP/en/Encyclopedia/NounDefinition/Pedia_12.aspx
    "mg": Bound(lower=0, upper=40),
    # k: 0 <= value <= 51
    # https://iris.who.int/server/api/core/bitstreams/ab162681-e9bc-4392-9aef-2c934cea2b5c/content
    "k": Bound(lower=0, upper=51),
    # na: 0 <= value <= 120000
    # https://www.epa.gov/sites/default/files/2014-09/documents/support_cc1_sodium_dwreport.pdf
    "na": Bound(lower=0, upper=120000),
    # cl: 0 <= value <= 460
    # https://cdn.who.int/media/docs/default-source/wash-documents/wash-chemicals/chloride.pdf?sfvrsn=f7d7502f_4
    "cl": Bound(lower=0, upper=460),
    # t_p: 0 <= value <= 20
    # https://www.researchgate.net/publication/347652156_An_Overview_of_the_Removal_Phosphorus_in_Wastewater
    "t_p": Bound(lower=0, upper=20),
    # t_s: value >= 0, no upper bound
    # No source URL supplied; this is the user-specified lower bound.
    "t_s": Bound(lower=0),
    # t_n: 0 <= value <= 20
    # https://www.erunwas.com/news-detail/id-335.html
    "t_n": Bound(lower=0, upper=20),
    # toc: 0 <= value <= 200
    # https://www.researchgate.net/publication/316351764_An_overview_of_dissolved_organic_carbon_in_groundwater_and_implications_for_drinking_water_safety
    "toc": Bound(lower=0, upper=200),
}

# D. Litterfall: monthly/period litterfall for Davos ICOS and LWF. A range is only
# applied where the dataset has the column (the Davos files have no
# Number_of_days).
LITTERFALL_RANGES: dict[str, Bound] = {
    # Weight: value >= 0, no upper bound
    # No source URL supplied; this is the user-specified lower bound.
    "Weight": Bound(lower=0),
    # Number_of_days: value >= 0, no upper bound
    # No source URL supplied; this is the user-specified lower bound.
    "Number_of_days": Bound(lower=0),
}

# Measurement columns of the deposition files, in file order.
DEPOSITION_MEASUREMENTS = list(DEPOSITION_RANGES)

# The ranges that apply to each dataset.
PLAUSIBLE_RANGES: dict[str, dict[str, Bound]] = {
    DS_LAI: LAI_RANGES,
    DS_FOLIAGE_I: FOLIAGE_RANGES,
    DS_FOLIAGE_PLOT: FOLIAGE_RANGES,
    DS_DEP_MONTHLY: DEPOSITION_RANGES,
    DS_DEP_PERIOD: DEPOSITION_RANGES,
    DS_LITTER_MONTHLY_DAVOS: LITTERFALL_RANGES,
    DS_LITTER_MONTHLY_LWF: LITTERFALL_RANGES,
    DS_LITTER_PERIOD_DAVOS: LITTERFALL_RANGES,
    DS_LITTER_PERIOD_LWF: LITTERFALL_RANGES,
}

# ----------------------------------------------------------------------------
# Completeness requirements (Step 3)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Completeness:
    """Columns a row must hold, after cleaning, to be kept.

    Every column in ``all_of`` must be non-null, and at least one column in
    ``any_of`` must be non-null (if ``any_of`` is not empty). Only measurement
    columns are listed, so a row is never dropped for lacking a measurement that
    belongs to another dataset. Identity (natural-key) columns are not listed:
    a row with an incomplete key is kept and is handled by the strict key, which
    never merges it with another row and reports it as ambiguous.
    """

    all_of: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()


# A row is kept when it still holds the measurement(s) its dataset exists to
# record. A deposition sample is a set of independent analyses, so one analysis
# is enough; foliage weight and litterfall weight are the single measurement of
# their files; an LAI row needs at least one of its two LAI estimates (standard
# errors, angles and point counts alone do not describe the canopy).
REQUIRED_COLUMNS: dict[str, Completeness] = {
    DS_LAI: Completeness(any_of=("mean_LAI_Miller", "mean_LAI_NormanCampbell")),
    DS_FOLIAGE_I: Completeness(all_of=("gew100",)),
    DS_FOLIAGE_PLOT: Completeness(all_of=("gew100",)),
    DS_DEP_MONTHLY: Completeness(any_of=tuple(DEPOSITION_MEASUREMENTS)),
    DS_DEP_PERIOD: Completeness(any_of=tuple(DEPOSITION_MEASUREMENTS)),
    DS_LITTER_MONTHLY_DAVOS: Completeness(all_of=("Weight",)),
    DS_LITTER_MONTHLY_LWF: Completeness(all_of=("Weight",)),
    DS_LITTER_PERIOD_DAVOS: Completeness(all_of=("Weight",)),
    DS_LITTER_PERIOD_LWF: Completeness(all_of=("Weight",)),
}

# Helper columns that mark cells set to null by a cleaning step. They only live
# between the cleaning steps and are dropped when incomplete rows are removed.
BADTYPE_PREFIX = "__badtype__"
BADRANGE_PREFIX = "__badrange__"

# Accumulators for the audit reports. The loaders and the coalescing step append
# to them and main writes them out.
DATASET_REPORT_ROWS: list[dict] = []
DUPLICATE_REPORT_FRAMES: list[pl.DataFrame] = []
CONFLICT_REPORT_FRAMES: list[pl.DataFrame] = []
# Cleaning counts per dataset, filled in by the cleaning steps.
CLEANING_STATS: dict[str, dict] = {}


# ============================================================================
# GENERIC HELPERS
# ============================================================================


def read_semicolon_csv(path: Path) -> pl.DataFrame:
    """Read a semicolon-separated CSV file into a DataFrame of text columns.

    The file is decoded as UTF-8 first and read again as Windows-1252 if Polars
    cannot decode it. The strings in EXTRA_NULL_VALUES are read as nulls. Every
    column is read as text, so that a malformed value can never stop the read;
    the preprocessing step converts each column to its expected type.
    """
    try:
        return pl.read_csv(
            path,
            separator=";",
            infer_schema_length=0,
            encoding="utf8",
            null_values=EXTRA_NULL_VALUES,
        )
    except pl.exceptions.PolarsError:
        return pl.read_csv(
            path,
            separator=";",
            infer_schema_length=0,
            encoding="windows-1252",
            null_values=EXTRA_NULL_VALUES,
        )


def read_litterfall_csv(path: Path) -> pl.DataFrame:
    """Read one of the four litterfall CSV files into a DataFrame of text columns.

    The first two lines of the file are skipped and the text is decoded as
    Windows-1252. The strings in EXTRA_NULL_VALUES are read as nulls, and every
    column is read as text, so that a malformed value can never stop the read.
    Missing markers that the reader does not recognize are handled by the
    preprocessing step.
    """
    return pl.read_csv(
        path,
        separator=";",
        skip_rows=2,
        encoding="windows-1252",
        infer_schema_length=0,
        null_values=EXTRA_NULL_VALUES,
    )


def smart_parse_date(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Convert a column to a Date, detecting the format from the data.

    Columns that are already Date or Datetime, and columns without any non-null
    values, are cast to Date directly. Otherwise each format in
    CANDIDATE_DATE_FORMATS is tried in turn, and the first one that parses every
    non-null value is used and reported. If none fits every value, the format
    that parses the most values is used, a warning is printed, and the values it
    cannot parse become null.
    """
    if df.schema[col] in (pl.Date, pl.Datetime):
        return df.with_columns(pl.col(col).cast(pl.Date))
    non_null_n = df[col].drop_nulls().len()
    if non_null_n == 0:
        return df.with_columns(pl.col(col).cast(pl.Date))
    best_parsed = pl.Series(col, [None] * df.height, dtype=pl.Date)
    best_fmt, best_n = None, 0
    for fmt in CANDIDATE_DATE_FORMATS:
        parsed = df[col].str.strptime(pl.Date, fmt, strict=False)
        n_parsed = parsed.drop_nulls().len()
        if n_parsed == non_null_n:
            print(f"    date column '{col}': detected format {fmt}")
            return df.with_columns(parsed.alias(col))
        if n_parsed > best_n:
            best_parsed, best_fmt, best_n = parsed, fmt, n_parsed
    print(
        f"    WARNING: date column '{col}' does not fit any of {CANDIDATE_DATE_FORMATS}. "
        f"Best format: {best_fmt}. {non_null_n - best_n} value(s) could not be parsed and "
        f"are set to null -- check the raw format and add it to CANDIDATE_DATE_FORMATS."
    )
    return df.with_columns(best_parsed.alias(col))


def strip_key_strings(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Trim leading and trailing whitespace from string columns.

    Only the listed columns that exist in the frame and hold strings are changed,
    so stray spaces in identity values cannot make equal keys look different.
    Other columns are left as they are.
    """
    exprs = []
    for c in cols:
        if c in df.columns and df.schema[c] == pl.Utf8:
            exprs.append(pl.col(c).str.strip_chars().alias(c))
    return df.with_columns(exprs) if exprs else df


def attach_site_name(df: pl.DataFrame, code_col: str = "location_code") -> pl.DataFrame:
    """Add a site_name column by looking up each code in SITE_CROSSWALK.

    The lookup is a left join, so codes missing from the crosswalk get a null
    site_name and are listed in a warning. It is applied once to the final table
    and site_name is never used as part of a key.
    """
    codes_present = set(df[code_col].drop_nulls().unique().to_list())
    unknown = sorted(codes_present - set(SITE_CROSSWALK.keys()))
    if unknown:
        print(
            f"    WARNING: location_code(s) not in the 19-site crosswalk: {unknown}. "
            f"site_name will be NULL for these rows -- check for typos."
        )
    lookup = pl.DataFrame(
        {code_col: list(SITE_CROSSWALK.keys()), "site_name": list(SITE_CROSSWALK.values())}
    )
    return df.join(lookup, on=code_col, how="left")


def prefix_columns(df: pl.DataFrame, cols: list[str], tag: str) -> pl.DataFrame:
    """Rename the listed columns to ``<tag>__<column>``.

    Listed columns that are not in the frame are ignored. The prefix keeps
    measurement columns from different datasets distinct after concatenation.
    """
    rename_map = {c: f"{tag}__{c}" for c in cols if c in df.columns}
    return df.rename(rename_map)


def add_missing_vocab_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Pad a frame with any missing identity columns and put them first.

    Missing identity columns are added as typed nulls (Date for date columns,
    string otherwise) so that every dataset shares the same identity schema. The
    identity columns follow the order of IDENTITY_VOCAB_COLS and are followed by
    the remaining columns.
    """
    exprs = []
    for c in IDENTITY_VOCAB_COLS:
        if c not in df.columns:
            dtype = pl.Date if c in DATE_VOCAB_COLS else pl.Utf8
            exprs.append(pl.lit(None, dtype=dtype).alias(c))
    if exprs:
        df = df.with_columns(exprs)
    other_cols = [c for c in df.columns if c not in IDENTITY_VOCAB_COLS]
    return df.select(IDENTITY_VOCAB_COLS + other_cols)


def add_key_signatures(df: pl.DataFrame, key_cols: list[str], dataset_name: str) -> pl.DataFrame:
    """Add the loose and strict key columns used for duplicate detection.

    Two string columns are built from the given key columns, with each part
    written as ``column=value`` and the parts joined by ``|``:

    - ``__loose_key__`` replaces a null field with a fixed marker, so rows with
      nulls in the same fields compare equal. It is only used to spot rows whose
      identity is incomplete.
    - ``__strict_key__`` replaces a null field with a marker that is unique to
      the dataset, column and row, so a null never equals another row's value.
      Duplicate detection and the cross-dataset coalescing group on this key.
    """
    df = df.with_row_index("__row_idx__")
    loose_parts = []
    strict_parts = []
    for c in key_cols:
        val = pl.col(c).cast(pl.Utf8)
        loose_parts.append(pl.format("{}={}", pl.lit(c), val.fill_null("\u2400NULL\u2400")))
        unique_token = pl.format(
            "\u2400UNIQUE\u2400{}\u2400{}\u2400{}\u2400",
            pl.lit(dataset_name),
            pl.lit(c),
            pl.col("__row_idx__"),
        )
        strict_parts.append(
            pl.format("{}={}", pl.lit(c), pl.when(val.is_null()).then(unique_token).otherwise(val))
        )
    df = df.with_columns(
        [
            pl.concat_str(loose_parts, separator="|").alias("__loose_key__"),
            pl.concat_str(strict_parts, separator="|").alias("__strict_key__"),
        ]
    )
    return df.drop("__row_idx__")


def validate_unique(
    df: pl.DataFrame, key_cols: list[str], dataset_name: str
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Detect duplicates in a dataset while keeping every distinct observation.

    Rows that are identical in all source columns are exact duplicates. They are
    reported and a single copy of each is kept.

    Among the remaining rows, those that share a loose key but not a strict key
    have a null somewhere in the key, so their identity cannot be confirmed. They
    are reported and left unmerged. Rows that share a complete strict key but
    differ in other columns are reported as natural-key collisions, and their
    strict keys receive a row-specific suffix so that they are never merged.

    The updated frame is returned together with a dictionary of the four counts
    used in the dataset report.
    """
    source_cols = [c for c in df.columns if not c.startswith("__")]
    exact_mask = df.select(source_cols).is_duplicated()
    n_exact = int(exact_mask.sum())
    rows_before_deduplication = df.height
    distinct_rows_before = df.select(source_cols).n_unique()

    if n_exact > 0:
        rows = df.filter(exact_mask).with_columns(
            [
                pl.lit(dataset_name).alias("__source_dataset__"),
                pl.lit("exact_source_duplicate").alias("__issue_type__"),
            ]
        )
        DUPLICATE_REPORT_FRAMES.append(rows)
        print(
            f"    WARNING: {n_exact} row(s) in '{dataset_name}' are exact source "
            "duplicates. They are reported and one copy of each is retained."
        )
        df = df.unique(subset=source_cols, keep="last", maintain_order=True)
    n_exact_removed = rows_before_deduplication - df.height
    # Only exact copies may be removed: no exact duplicate is left, and every
    # distinct observation is still present.
    assert df.select(source_cols).is_duplicated().sum() == 0, dataset_name
    assert df.height == distinct_rows_before, (
        f"'{dataset_name}': deduplication removed a distinct observation"
    )

    strict_dup = df.select(pl.col("__strict_key__")).to_series().is_duplicated()
    loose_dup = df.select(pl.col("__loose_key__")).to_series().is_duplicated()
    ambiguous_mask = loose_dup & ~strict_dup
    collision_mask = strict_dup

    n_ambiguous = int(ambiguous_mask.sum())
    n_collisions = int(collision_mask.sum())

    if n_ambiguous > 0:
        rows = df.filter(ambiguous_mask).with_columns(
            [
                pl.lit(dataset_name).alias("__source_dataset__"),
                pl.lit("ambiguous_incomplete_key").alias("__issue_type__"),
            ]
        )
        DUPLICATE_REPORT_FRAMES.append(rows)
        print(
            f"    NOTE: {n_ambiguous} row(s) in '{dataset_name}' share identical non-NULL "
            f"values on {key_cols} but at least one key field is NULL, so identity cannot "
            f"be confirmed. ALL such rows are preserved separately, unmerged -- see "
            f"duplicate report (category=ambiguous_incomplete_key)."
        )

    if n_collisions > 0:
        rows = df.filter(collision_mask).with_columns(
            [
                pl.lit(dataset_name).alias("__source_dataset__"),
                pl.lit("natural_key_collision_retained").alias("__issue_type__"),
            ]
        )
        DUPLICATE_REPORT_FRAMES.append(rows)
        print(
            f"    WARNING: {n_collisions} row(s) in '{dataset_name}' share a complete "
            f"natural key {key_cols} but are not exact source duplicates. "
            "They are retained separately."
        )
        df = (
            df.with_row_index("__collision_row__")
            .with_columns(
                pl.when(pl.col("__strict_key__").is_duplicated())
                .then(
                    pl.concat_str(
                        [
                            pl.col("__strict_key__"),
                            pl.lit("|retained_row="),
                            pl.col("__collision_row__").cast(pl.Utf8),
                        ]
                    )
                )
                .otherwise(pl.col("__strict_key__"))
                .alias("__strict_key__")
            )
            .drop("__collision_row__")
        )

    return df, {
        "exact_source_duplicates": n_exact,
        "exact_source_rows_removed": n_exact_removed,
        "ambiguous_incomplete_key": n_ambiguous,
        "natural_key_collisions_retained": n_collisions,
    }


def add_month_end_bounds(df: pl.DataFrame, year_col: str, month_col: str) -> pl.DataFrame:
    """Derive start_date and end_date from a survey year and month.

    The end date is the last day of the survey month. The start date is the last
    day of the preceding month, which is the convention of the monthly deposition
    file. A row whose month is not 1 to 12, or whose year or month is null, gets
    null dates instead of raising an error.
    """
    valid = pl.col(month_col).is_between(1, 12) & pl.col(year_col).is_between(1, 9999)
    first_of_month = pl.date(pl.col(year_col).clip(1, 9999), pl.col(month_col).clip(1, 12), 1)
    df = df.with_columns(pl.when(valid).then(first_of_month).alias("__first_of_month"))
    df = df.with_columns(
        [
            (pl.col("__first_of_month").dt.offset_by("1mo") - pl.duration(days=1)).alias(
                "end_date"
            ),
            (pl.col("__first_of_month") - pl.duration(days=1)).alias("start_date"),
        ]
    )
    return df.drop("__first_of_month")


def record_dataset_report(
    name: str, raw: pl.DataFrame, standardized: pl.DataFrame, dup_counts: dict
) -> None:
    """Append the row and column counts of one dataset to the audit report.

    The counts cover the raw file, the cleaning steps (collected in
    CLEANING_STATS), the standardized frame and the duplicate statistics
    returned by validate_unique.
    """
    cleaning = CLEANING_STATS[name]
    DATASET_REPORT_ROWS.append(
        {
            "dataset": name,
            "original_rows": raw.height,
            "original_columns": raw.width,
            "required_columns_rule": describe_completeness(REQUIRED_COLUMNS[name]),
            "cells_missing_or_infinite_to_null": cleaning["cells_missing_or_infinite_to_null"],
            "cells_invalid_type_to_null": cleaning["cells_invalid_type_to_null"],
            "cells_out_of_range_to_null": cleaning["cells_out_of_range_to_null"],
            "out_of_range_by_column": "; ".join(
                f"{col}={n}" for col, n in cleaning["out_of_range_by_column"].items() if n
            ),
            "rows_removed_missing_values": cleaning["rows_removed_missing_values"],
            "rows_removed_invalid_type": cleaning["rows_removed_invalid_type"],
            "rows_removed_implausible": cleaning["rows_removed_implausible"],
            "rows_after_cleaning": cleaning["rows_after_cleaning"],
            "rows_after_standardization": standardized.height,
            "exact_source_duplicates": dup_counts["exact_source_duplicates"],
            "exact_source_rows_removed": dup_counts["exact_source_rows_removed"],
            "ambiguous_incomplete_key": dup_counts["ambiguous_incomplete_key"],
            "natural_key_collisions_retained": dup_counts["natural_key_collisions_retained"],
            "standardized_columns": standardized.width,
        }
    )


def finalize_loader(
    df: pl.DataFrame, raw: pl.DataFrame, dataset_name: str, value_cols: list[str], prefix: str
) -> pl.DataFrame:
    """Apply the standardization steps shared by every loader.

    The cleaned frame is checked first. Key strings are trimmed, the loose and
    strict keys are built, duplicates are validated, the measurement columns are
    prefixed with the dataset tag, the identity columns are padded to the common
    set, and the dataset name is stored in ``__source_dataset__``. The loose key
    is then dropped, the per-dataset report row is recorded, and the standardized
    frame is returned.
    """
    validate_cleaned_dataset(df, dataset_name)
    CLEANING_STATS[dataset_name]["rows_after_cleaning"] = df.height
    key_cols = NATURAL_KEYS[dataset_name]
    df = strip_key_strings(df, key_cols)
    df = add_key_signatures(df, key_cols, dataset_name)
    df, dup_counts = validate_unique(df, key_cols, dataset_name)
    df = prefix_columns(df, value_cols, prefix)
    df = add_missing_vocab_cols(df)
    df = df.with_columns(pl.lit(dataset_name).alias("__source_dataset__"))
    df = df.drop("__loose_key__")
    record_dataset_report(dataset_name, raw, df, dup_counts)
    return df


# ============================================================================
# CLEANING ENGINE (shared by the per-dataset functions below)
# ============================================================================


def new_cleaning_stats(original_rows: int = 0) -> dict:
    """Return an empty set of cleaning counters for one dataset."""
    return {
        "original_rows": original_rows,
        "cells_missing_or_infinite_to_null": 0,
        "cells_invalid_type_to_null": 0,
        "cells_out_of_range_to_null": 0,
        "out_of_range_by_column": {},
        "rows_removed_missing_values": 0,
        "rows_removed_invalid_type": 0,
        "rows_removed_implausible": 0,
        "rows_after_cleaning": original_rows,
    }


def start_cleaning_stats(dataset_name: str, raw: pl.DataFrame) -> dict:
    """Create and register the cleaning counters of a dataset from its raw frame."""
    stats = new_cleaning_stats(raw.height)
    CLEANING_STATS[dataset_name] = stats
    return stats


def describe_completeness(rule: Completeness) -> str:
    """Write a completeness rule as text for the merge report."""
    parts = []
    if rule.all_of:
        parts.append("all of: " + ", ".join(rule.all_of))
    if rule.any_of:
        parts.append("at least one of: " + ", ".join(rule.any_of))
    return "; ".join(parts) if parts else "none"


def _count(df: pl.DataFrame, condition: pl.Expr) -> int:
    """Count the rows of a frame where a boolean expression is true."""
    return int(df.select(condition.sum()).item() or 0)


# ---- Step 1: missing values and data types ---------------------------------


def _standardize_text(df: pl.DataFrame, col: str) -> tuple[pl.DataFrame, int, pl.Series | None]:
    """Turn missing markers in a text column into null; other text is untouched."""
    text = pl.col(col).cast(pl.Utf8)
    is_marker = text.str.strip_chars().is_in(TEXT_MISSING_MARKERS)
    n_missing = _count(df, text.is_not_null() & is_marker)
    df = df.with_columns(pl.when(is_marker).then(None).otherwise(text).alias(col))
    return df, n_missing, None


def _standardize_number(
    df: pl.DataFrame, col: str, kind: str
) -> tuple[pl.DataFrame, int, pl.Series]:
    """Convert a column to Float64 or Int64.

    Missing markers, infinities and NaN become null and are counted as missing.
    Text that is not a number, and for "int" columns any fraction, becomes null
    and is flagged as an invalid type.
    """
    text = pl.col(col).cast(pl.Utf8).str.strip_chars()
    is_marker = text.str.to_uppercase().is_in(NUMERIC_MISSING_MARKERS) | text.str.contains(
        INFINITY_PATTERN
    )
    n_marker = _count(df, text.is_not_null() & is_marker)

    parsed = df.select(pl.when(is_marker).then(None).otherwise(text).alias("text"))
    parsed = parsed.with_columns(pl.col("text").cast(pl.Float64, strict=False).alias("num"))
    parsed = parsed.with_columns(
        (pl.col("num").is_nan() | pl.col("num").is_infinite()).fill_null(False).alias("nonfinite"),
        (pl.col("text").is_not_null() & pl.col("num").is_null()).alias("bad_type"),
    )
    n_nonfinite = int(parsed["nonfinite"].sum())
    parsed = parsed.with_columns(
        pl.when(pl.col("nonfinite")).then(None).otherwise(pl.col("num")).alias("num")
    )
    if kind == "int":
        is_whole = (pl.col("num") == pl.col("num").floor()) & (pl.col("num").abs() < 2**53)
        parsed = parsed.with_columns(
            (pl.col("bad_type") | (pl.col("num").is_not_null() & ~is_whole)).alias("bad_type")
        )
        parsed = parsed.with_columns(
            pl.when(pl.col("bad_type")).then(None).otherwise(pl.col("num")).alias("num")
        )
        parsed = parsed.with_columns(pl.col("num").cast(pl.Int64))
    df = df.with_columns(parsed["num"].alias(col))
    return df, n_marker + n_nonfinite, parsed["bad_type"].fill_null(False)


def _standardize_date(df: pl.DataFrame, col: str) -> tuple[pl.DataFrame, int, pl.Series]:
    """Convert a column to Date; markers and unparseable dates become null."""
    if df.schema[col] in (pl.Date, pl.Datetime):
        return (
            df.with_columns(pl.col(col).cast(pl.Date)),
            0,
            pl.Series(col, [False] * df.height, dtype=pl.Boolean),
        )
    text = pl.col(col).cast(pl.Utf8).str.strip_chars()
    is_marker = text.is_in(TEXT_MISSING_MARKERS)
    n_missing = _count(df, text.is_not_null() & is_marker)
    df = df.with_columns(pl.when(is_marker).then(None).otherwise(text).alias(col))
    had_value = df[col].is_not_null()
    df = smart_parse_date(df, col)
    return df, n_missing, (had_value & df[col].is_null()).fill_null(False)


def standardize_types(
    df: pl.DataFrame, schema: dict[str, str], stats: dict, dataset_name: str
) -> pl.DataFrame:
    """Step 1: convert missing values and data types for the columns in a schema.

    Each listed column is converted to its expected type (see TYPE_SCHEMAS).
    Missing markers and infinities become null, and values that cannot be
    converted become null instead of raising an error. A zero is a valid value
    and is never treated as missing, and text and identifier columns are never
    converted to numbers. The number of cells set to null is added to ``stats``.
    A boolean helper column marks the cells that failed the type conversion; it
    is used later to tell invalid types from plain missing values.
    """
    absent = [c for c in schema if c not in df.columns]
    if absent:
        print(f"    NOTE: '{dataset_name}' has no column(s) {absent}; they are skipped.")
    unlisted = [c for c in df.columns if c not in schema and not c.startswith("__")]
    if unlisted:
        print(
            f"    NOTE: '{dataset_name}' column(s) {unlisted} are not in TYPE_SCHEMAS; unchanged."
        )

    for col, kind in schema.items():
        if col not in df.columns:
            continue
        if kind == "str":
            df, n_missing, bad_type = _standardize_text(df, col)
        elif kind in ("float", "int"):
            df, n_missing, bad_type = _standardize_number(df, col, kind)
        elif kind == "date":
            df, n_missing, bad_type = _standardize_date(df, col)
        else:
            raise ValueError(f"unknown column kind '{kind}' for column '{col}'")
        stats["cells_missing_or_infinite_to_null"] += n_missing
        if bad_type is not None:
            stats["cells_invalid_type_to_null"] += int(bad_type.sum())
            df = df.with_columns(bad_type.alias(f"{BADTYPE_PREFIX}{col}"))

    _assert_types_standardized(df, schema, dataset_name)
    print(
        f"    preprocessing: {stats['cells_missing_or_infinite_to_null']} missing/infinite "
        f"cell(s) and {stats['cells_invalid_type_to_null']} invalid-type cell(s) set to null"
    )
    return df


def _assert_types_standardized(df: pl.DataFrame, schema: dict[str, str], name: str) -> None:
    """Check that every listed column has its expected type and no leftover marker."""
    for col, kind in schema.items():
        if col not in df.columns:
            continue
        assert df.schema[col] == EXPECTED_DTYPES[kind], (
            f"'{name}.{col}' has type {df.schema[col]}, expected {EXPECTED_DTYPES[kind]}"
        )
        if kind == "str":
            leftover = df[col].str.strip_chars().is_in(TEXT_MISSING_MARKERS).sum()
            assert leftover == 0, f"'{name}.{col}' still holds a missing-value marker"
        elif kind == "float":
            assert df[col].is_nan().sum() == 0, f"'{name}.{col}' still holds NaN"
            assert df[col].is_infinite().sum() == 0, f"'{name}.{col}' still holds infinity"


# ---- Step 2: plausible ranges ----------------------------------------------


def out_of_range_expr(col: str, bound: Bound) -> pl.Expr:
    """Return a boolean expression that is true where a value breaks its bound.

    A null is never out of range. The lower bound is exclusive when
    ``strict_lower`` is set and inclusive otherwise; the upper bound is inclusive.
    """
    value = pl.col(col)
    violated = pl.lit(False)
    if bound.lower is not None:
        violated = violated | (value <= bound.lower if bound.strict_lower else value < bound.lower)
    if bound.upper is not None:
        violated = violated | (value > bound.upper)
    return (value.is_not_null() & violated).fill_null(False)


def apply_plausible_ranges(
    df: pl.DataFrame, ranges: dict[str, Bound], stats: dict
) -> pl.DataFrame:
    """Step 2: set values outside their plausible range to null.

    Rows are not dropped here. A boolean helper column per variable marks the
    cells that were nulled, and the counts are added to ``stats``.
    """
    for col, bound in ranges.items():
        if col not in df.columns:
            print(f"    NOTE: no column '{col}' to range-check; skipped.")
            continue
        assert df.schema[col].is_numeric(), f"range check on non-numeric column '{col}'"
        flag = f"{BADRANGE_PREFIX}{col}"
        df = df.with_columns(out_of_range_expr(col, bound).alias(flag))
        n_bad = int(df[flag].sum())
        stats["cells_out_of_range_to_null"] += n_bad
        stats["out_of_range_by_column"][col] = n_bad
        df = df.with_columns(pl.when(pl.col(flag)).then(None).otherwise(pl.col(col)).alias(col))
    print(f"    plausible ranges: {stats['cells_out_of_range_to_null']} cell(s) set to null")
    return df


# ---- Step 3: rows without the required measurements ------------------------


def _has_value(df: pl.DataFrame, col: str, restore_range: bool) -> pl.Expr:
    """Return whether a cell holds a value, including range-nulled cells if requested."""
    expr = pl.col(col).is_not_null()
    flag = f"{BADRANGE_PREFIX}{col}"
    if restore_range and flag in df.columns:
        expr = expr | pl.col(flag)
    return expr


def _is_complete(df: pl.DataFrame, rule: Completeness, restore_range: bool) -> pl.Expr:
    """Boolean expression that is true for rows meeting a completeness rule."""
    complete = pl.lit(True)
    if rule.all_of:
        complete = complete & pl.all_horizontal(
            [_has_value(df, c, restore_range) for c in rule.all_of]
        )
    if rule.any_of:
        complete = complete & pl.any_horizontal(
            [_has_value(df, c, restore_range) for c in rule.any_of]
        )
    return complete


def remove_incomplete_rows(df: pl.DataFrame, rule: Completeness, stats: dict) -> pl.DataFrame:
    """Step 3: drop rows that lack the measurements the dataset requires.

    Nothing is imputed. Each removed row is counted once, by the step that left
    it without its required values:

    - plausible-range check: it would have been complete before the range check;
    - invalid type: it was already incomplete after preprocessing and one of the
      required cells failed its type conversion;
    - missing values: it was already incomplete after preprocessing and the
      required cells were plainly missing (or infinite).

    The helper columns of the cleaning steps are dropped from the result.
    """
    candidates = [*rule.all_of, *rule.any_of]
    missing_cols = [c for c in candidates if c not in df.columns]
    assert not missing_cols, f"required column(s) {missing_cols} not in the frame"

    badtype_cols = [
        pl.col(f"{BADTYPE_PREFIX}{c}") for c in candidates if f"{BADTYPE_PREFIX}{c}" in df.columns
    ]
    any_badtype = pl.any_horizontal(badtype_cols) if badtype_cols else pl.lit(False)
    complete_now = _is_complete(df, rule, restore_range=False)
    complete_before_range = _is_complete(df, rule, restore_range=True)

    removed = ~complete_now
    n_removed = _count(df, removed)
    n_implausible = _count(df, removed & complete_before_range)
    n_invalid = _count(df, removed & ~complete_before_range & any_badtype)
    n_missing = _count(df, removed & ~complete_before_range & ~any_badtype)
    assert n_implausible + n_invalid + n_missing == n_removed

    rows_before = df.height
    df = df.filter(complete_now)
    assert rows_before - df.height == n_removed
    stats["rows_removed_missing_values"] += n_missing
    stats["rows_removed_invalid_type"] += n_invalid
    stats["rows_removed_implausible"] += n_implausible
    helpers = [c for c in df.columns if c.startswith((BADTYPE_PREFIX, BADRANGE_PREFIX))]
    print(
        f"    incomplete rows removed ({describe_completeness(rule)}): "
        f"{n_missing} missing, {n_invalid} invalid type, {n_implausible} implausible"
    )
    return df.drop(helpers)


def validate_cleaned_dataset(df: pl.DataFrame, dataset_name: str) -> None:
    """Assert that a cleaned dataset meets every cleaning rule.

    No value is outside its plausible range, every row meets the completeness
    rule, and no helper column of the cleaning steps is left.
    """
    for col, bound in PLAUSIBLE_RANGES[dataset_name].items():
        if col in df.columns:
            n_bad = _count(df, out_of_range_expr(col, bound))
            assert n_bad == 0, f"'{dataset_name}.{col}' still holds {n_bad} out-of-range value(s)"
    complete = _is_complete(df, REQUIRED_COLUMNS[dataset_name], restore_range=False)
    assert _count(df, ~complete) == 0, f"'{dataset_name}' holds rows without required values"
    leftover = [c for c in df.columns if c.startswith((BADTYPE_PREFIX, BADRANGE_PREFIX))]
    assert not leftover, f"'{dataset_name}' still has cleaning helper columns {leftover}"


# ============================================================================
# STEP 1 - PREPROCESSING (one function per dataset)
# ============================================================================
# Each function converts missing values and data types on the source columns
# (see TYPE_SCHEMAS), then applies the dataset's renames and derived columns.


def preprocess_lai(raw: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the LAI Licor spreadsheet.

    Column names are assigned by position, then the types are standardized: the
    location code, subplot, plot type and season stay text, the date is parsed,
    the point count is a whole number and the LAI and angle columns are decimals.
    """
    orig_cols = raw.columns
    print(f"    original columns found (in order): {orig_cols}")
    new_names = [
        "location_code",
        "lai_subplot",
        "date",
        "plot_type",
        "mean_LAI_Miller",
        "standard_error_LAI_Miller",
        "mean_LAI_NormanCampbell",
        "standard_error_LAI_NormanCampbell",
        "mean_angle_Miller",
        "standard_error_angle_Miller",
        "mean_angle_NormanCampbell",
        "standard_error_angle_NormanCampbell",
        "number_of_points",
        "season",
    ]
    if len(orig_cols) != len(new_names):
        print(
            f"    WARNING: expected {len(new_names)} columns based on the screenshot, "
            f"found {len(orig_cols)}. Positional rename below may be WRONG -- verify!"
        )
    df = raw.rename(dict(zip(orig_cols, new_names, strict=False)))
    return standardize_types(df, TYPE_SCHEMAS[DS_LAI], CLEANING_STATS[DS_LAI], DS_LAI)


def preprocess_foliage_i(raw: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the lwf_foliage_dw100_i file.

    The types are standardized, then the plot id and survey date become
    location_code and date. The plot name is kept as plot_name_raw.
    """
    df = standardize_types(
        raw, TYPE_SCHEMAS[DS_FOLIAGE_I], CLEANING_STATS[DS_FOLIAGE_I], DS_FOLIAGE_I
    )
    return df.rename(
        {"plot_id": "location_code", "plot_name": "plot_name_raw", "survey_date": "date"}
    )


def preprocess_foliage_plot(raw: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the lwf_foliage_dw100_plot file.

    The types are standardized, then the plot id and survey date become
    location_code and date. The plot name is kept as plot_name_raw.
    """
    df = standardize_types(
        raw, TYPE_SCHEMAS[DS_FOLIAGE_PLOT], CLEANING_STATS[DS_FOLIAGE_PLOT], DS_FOLIAGE_PLOT
    )
    return df.rename(
        {"plot_id": "location_code", "plot_name": "plot_name_raw", "survey_date": "date"}
    )


def preprocess_dep_monthly(raw: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the monthly_dep_lwf file.

    The types are standardized, then the plot id and location become
    location_code and dep_location, and start_date and end_date are derived from
    the survey year and month.
    """
    df = standardize_types(
        raw, TYPE_SCHEMAS[DS_DEP_MONTHLY], CLEANING_STATS[DS_DEP_MONTHLY], DS_DEP_MONTHLY
    )
    df = df.rename(
        {"plot_id": "location_code", "plot_name": "plot_name_raw", "location": "dep_location"}
    )
    return add_month_end_bounds(df, "survey_year", "survey_month")


def preprocess_dep_period(raw: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the period_dep_lwf file.

    The types are standardized (both period boundaries are parsed as dates), then
    the plot id, location and period boundaries are renamed to location_code,
    dep_location, start_date and end_date.
    """
    df = standardize_types(
        raw, TYPE_SCHEMAS[DS_DEP_PERIOD], CLEANING_STATS[DS_DEP_PERIOD], DS_DEP_PERIOD
    )
    return df.rename(
        {
            "plot_id": "location_code",
            "plot_name": "plot_name_raw",
            "location": "dep_location",
            "date_start": "start_date",
            "date_end": "end_date",
        }
    )


def preprocess_litter_monthly_davos(raw: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the Davos ICOS monthly litterfall file.

    The types are standardized, the year, month and plot name columns are
    dropped, and the key columns are renamed.
    """
    stats = CLEANING_STATS[DS_LITTER_MONTHLY_DAVOS]
    df = standardize_types(
        raw, TYPE_SCHEMAS[DS_LITTER_MONTHLY_DAVOS], stats, DS_LITTER_MONTHLY_DAVOS
    )
    # Drop columns that are not carried into the universal table.
    df = df.drop(["Year", "Month", "LWF_plot"])
    return df.rename(
        {
            "LWF_plot_code": "location_code",
            "Start_date": "start_date",
            "End_date": "end_date",
            "Code_fraction": "fraction_code",
        }
    )


def preprocess_litter_monthly_lwf(raw: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the LWF monthly litterfall file.

    The types are standardized, the year, month and plot name columns are
    dropped, and the key columns are renamed.
    """
    stats = CLEANING_STATS[DS_LITTER_MONTHLY_LWF]
    df = standardize_types(raw, TYPE_SCHEMAS[DS_LITTER_MONTHLY_LWF], stats, DS_LITTER_MONTHLY_LWF)
    df = df.drop(["Year", "Month", "LWF_plot"])
    return df.rename(
        {
            "LWF_plot_code": "location_code",
            "Start_date": "start_date",
            "End_date": "end_date",
            "Sub_plot": "sub_plot",
            "Code_fraction_2": "fraction_code_lwf",
            "Tree_species": "tree_species",
        }
    )


def preprocess_litter_period_davos(raw: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the Davos ICOS period litterfall file.

    The types are standardized, the plot name column is dropped, and the key
    columns are renamed.
    """
    stats = CLEANING_STATS[DS_LITTER_PERIOD_DAVOS]
    df = standardize_types(
        raw, TYPE_SCHEMAS[DS_LITTER_PERIOD_DAVOS], stats, DS_LITTER_PERIOD_DAVOS
    )
    df = df.drop(["LWF_plot"])
    return df.rename(
        {
            "LWF_plot_code": "location_code",
            "Start_date": "start_date",
            "End_date": "end_date",
            "Code_fraction": "fraction_code",
        }
    )


def preprocess_litter_period_lwf(raw: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the LWF period litterfall file.

    The types are standardized, the plot name column is dropped, and the key
    columns are renamed.
    """
    stats = CLEANING_STATS[DS_LITTER_PERIOD_LWF]
    df = standardize_types(raw, TYPE_SCHEMAS[DS_LITTER_PERIOD_LWF], stats, DS_LITTER_PERIOD_LWF)
    df = df.drop(["LWF_plot"])
    return df.rename(
        {
            "LWF_plot_code": "location_code",
            "Start_date": "start_date",
            "End_date": "end_date",
            "Sub_plot": "sub_plot",
            "Code_fraction_1": "fraction_code_lwf_detailed",
            "Code_fraction_2": "fraction_code_lwf",
            "Tree_species": "tree_species",
        }
    )


# ============================================================================
# STEP 2 - REMOVING IMPLAUSIBLE VALUES (one function per dataset)
# ============================================================================
# Each function sets the values outside the ranges in PLAUSIBLE_RANGES to null,
# then drops the rows that no longer hold the dataset's required measurements
# (REQUIRED_COLUMNS). The two parts are separate functions so that each can be
# read on its own.


def remove_implausible_lai(df: pl.DataFrame) -> pl.DataFrame:
    """Set implausible LAI, canopy angle and point-count values in the LAI spreadsheet to null."""
    return apply_plausible_ranges(df, PLAUSIBLE_RANGES[DS_LAI], CLEANING_STATS[DS_LAI])


def remove_implausible_foliage_i(df: pl.DataFrame) -> pl.DataFrame:
    """Set implausible foliage weight (gew100) values in the lwf_foliage_dw100_i file to null."""
    return apply_plausible_ranges(df, PLAUSIBLE_RANGES[DS_FOLIAGE_I], CLEANING_STATS[DS_FOLIAGE_I])


def remove_implausible_foliage_plot(df: pl.DataFrame) -> pl.DataFrame:
    """Set implausible foliage weights in the lwf_foliage_dw100_plot file to null."""
    return apply_plausible_ranges(
        df, PLAUSIBLE_RANGES[DS_FOLIAGE_PLOT], CLEANING_STATS[DS_FOLIAGE_PLOT]
    )


def remove_implausible_dep_monthly(df: pl.DataFrame) -> pl.DataFrame:
    """Set implausible deposition values in the monthly_dep_lwf file to null."""
    return apply_plausible_ranges(
        df, PLAUSIBLE_RANGES[DS_DEP_MONTHLY], CLEANING_STATS[DS_DEP_MONTHLY]
    )


def remove_implausible_dep_period(df: pl.DataFrame) -> pl.DataFrame:
    """Set implausible deposition values in the period_dep_lwf file to null."""
    return apply_plausible_ranges(
        df, PLAUSIBLE_RANGES[DS_DEP_PERIOD], CLEANING_STATS[DS_DEP_PERIOD]
    )


def remove_implausible_litter_monthly_davos(df: pl.DataFrame) -> pl.DataFrame:
    """Set implausible weights in the Davos ICOS monthly litterfall file to null."""
    return apply_plausible_ranges(
        df, PLAUSIBLE_RANGES[DS_LITTER_MONTHLY_DAVOS], CLEANING_STATS[DS_LITTER_MONTHLY_DAVOS]
    )


def remove_implausible_litter_monthly_lwf(df: pl.DataFrame) -> pl.DataFrame:
    """Set implausible weights and day counts in the LWF monthly litterfall file to null."""
    return apply_plausible_ranges(
        df, PLAUSIBLE_RANGES[DS_LITTER_MONTHLY_LWF], CLEANING_STATS[DS_LITTER_MONTHLY_LWF]
    )


def remove_implausible_litter_period_davos(df: pl.DataFrame) -> pl.DataFrame:
    """Set implausible weights in the Davos ICOS period litterfall file to null."""
    return apply_plausible_ranges(
        df, PLAUSIBLE_RANGES[DS_LITTER_PERIOD_DAVOS], CLEANING_STATS[DS_LITTER_PERIOD_DAVOS]
    )


def remove_implausible_litter_period_lwf(df: pl.DataFrame) -> pl.DataFrame:
    """Set implausible weights and day counts in the LWF period litterfall file to null."""
    return apply_plausible_ranges(
        df, PLAUSIBLE_RANGES[DS_LITTER_PERIOD_LWF], CLEANING_STATS[DS_LITTER_PERIOD_LWF]
    )


# ============================================================================
# STEP 3 - DROPPING INCOMPLETE ROWS (one function per dataset)
# ============================================================================
# Each function drops the rows of one dataset that no longer hold the
# measurements listed for it in REQUIRED_COLUMNS. A row is never dropped for
# lacking a measurement that belongs to another dataset.


def remove_incomplete_lai(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows of the LAI spreadsheet without their required measurements."""
    return remove_incomplete_rows(df, REQUIRED_COLUMNS[DS_LAI], CLEANING_STATS[DS_LAI])


def remove_incomplete_foliage_i(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows of the lwf_foliage_dw100_i file without their required measurements."""
    return remove_incomplete_rows(df, REQUIRED_COLUMNS[DS_FOLIAGE_I], CLEANING_STATS[DS_FOLIAGE_I])


def remove_incomplete_foliage_plot(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows of the lwf_foliage_dw100_plot file without their required measurements."""
    return remove_incomplete_rows(
        df, REQUIRED_COLUMNS[DS_FOLIAGE_PLOT], CLEANING_STATS[DS_FOLIAGE_PLOT]
    )


def remove_incomplete_dep_monthly(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows of the monthly_dep_lwf file without their required measurements."""
    return remove_incomplete_rows(
        df, REQUIRED_COLUMNS[DS_DEP_MONTHLY], CLEANING_STATS[DS_DEP_MONTHLY]
    )


def remove_incomplete_dep_period(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows of the period_dep_lwf file without their required measurements."""
    return remove_incomplete_rows(
        df, REQUIRED_COLUMNS[DS_DEP_PERIOD], CLEANING_STATS[DS_DEP_PERIOD]
    )


def remove_incomplete_litter_monthly_davos(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows of the Davos ICOS monthly litterfall file without their required measurements."""
    return remove_incomplete_rows(
        df, REQUIRED_COLUMNS[DS_LITTER_MONTHLY_DAVOS], CLEANING_STATS[DS_LITTER_MONTHLY_DAVOS]
    )


def remove_incomplete_litter_monthly_lwf(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows of the LWF monthly litterfall file without their required measurements."""
    return remove_incomplete_rows(
        df, REQUIRED_COLUMNS[DS_LITTER_MONTHLY_LWF], CLEANING_STATS[DS_LITTER_MONTHLY_LWF]
    )


def remove_incomplete_litter_period_davos(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows of the Davos ICOS period litterfall file without their required measurements."""
    return remove_incomplete_rows(
        df, REQUIRED_COLUMNS[DS_LITTER_PERIOD_DAVOS], CLEANING_STATS[DS_LITTER_PERIOD_DAVOS]
    )


def remove_incomplete_litter_period_lwf(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows of the LWF period litterfall file without their required measurements."""
    return remove_incomplete_rows(
        df, REQUIRED_COLUMNS[DS_LITTER_PERIOD_LWF], CLEANING_STATS[DS_LITTER_PERIOD_LWF]
    )


# ============================================================================
# PER-DATASET LOADERS (read, clean, then standardize)
# ============================================================================
# Every loader runs the same sequence: read the raw file, preprocess it, remove
# implausible values, drop incomplete rows, and standardize the result for the
# merge (finalize_loader checks the cleaned frame and removes exact duplicates).


def load_lai(path: Path) -> pl.DataFrame:
    """Load and standardize the LAI Licor spreadsheet.

    Location, subplot, date and plot type are identity columns. The remaining
    LAI, angle, point count and season columns are prefixed with ``lai``.
    """
    print("Loading LAI_Licor_3_rings_all_years ...")
    # Skip the leading non-data rows; column names are assigned by position.
    raw = pl.read_excel(path).slice(2)
    start_cleaning_stats(DS_LAI, raw)
    df = preprocess_lai(raw)
    df = remove_implausible_lai(df)
    df = remove_incomplete_lai(df)
    # Identity columns (location_code, lai_subplot, date, plot_type) are not
    # listed here and keep their unprefixed names.
    value_cols = [
        "mean_LAI_Miller",
        "standard_error_LAI_Miller",
        "mean_LAI_NormanCampbell",
        "standard_error_LAI_NormanCampbell",
        "mean_angle_Miller",
        "standard_error_angle_Miller",
        "mean_angle_NormanCampbell",
        "standard_error_angle_NormanCampbell",
        "number_of_points",
        "season",
    ]
    return finalize_loader(df, raw, DS_LAI, value_cols, DATASET_PREFIX[DS_LAI])


def load_foliage_i(path: Path) -> pl.DataFrame:
    """Load and standardize the lwf_foliage_dw100_i foliage weight file.

    The survey year, plot name and weight columns are prefixed with ``foliage_i``.
    """
    print("Loading lwf_foliage_dw100_i ...")
    raw = read_semicolon_csv(path)
    start_cleaning_stats(DS_FOLIAGE_I, raw)
    df = preprocess_foliage_i(raw)
    df = remove_implausible_foliage_i(df)
    df = remove_incomplete_foliage_i(df)
    value_cols = ["survey_year", "plot_name_raw", "gew100"]
    return finalize_loader(df, raw, DS_FOLIAGE_I, value_cols, DATASET_PREFIX[DS_FOLIAGE_I])


def load_foliage_plot(path: Path) -> pl.DataFrame:
    """Load and standardize the lwf_foliage_dw100_plot foliage weight file.

    The plot name and weight columns are prefixed with ``foliage_plot``.
    """
    print("Loading lwf_foliage_dw100_plot ...")
    raw = read_semicolon_csv(path)
    start_cleaning_stats(DS_FOLIAGE_PLOT, raw)
    df = preprocess_foliage_plot(raw)
    df = remove_implausible_foliage_plot(df)
    df = remove_incomplete_foliage_plot(df)
    value_cols = ["plot_name_raw", "gew100"]
    return finalize_loader(df, raw, DS_FOLIAGE_PLOT, value_cols, DATASET_PREFIX[DS_FOLIAGE_PLOT])


def load_dep_monthly(path: Path) -> pl.DataFrame:
    """Load and standardize the monthly_dep_lwf deposition file.

    The survey and chemistry columns are prefixed with ``dep_monthly``.
    """
    print("Loading monthly_dep_lwf ...")
    raw = read_semicolon_csv(path)
    start_cleaning_stats(DS_DEP_MONTHLY, raw)
    df = preprocess_dep_monthly(raw)
    df = remove_implausible_dep_monthly(df)
    df = remove_incomplete_dep_monthly(df)
    value_cols = ["survey_year", "survey_month", "plot_name_raw", *DEPOSITION_MEASUREMENTS]
    return finalize_loader(df, raw, DS_DEP_MONTHLY, value_cols, DATASET_PREFIX[DS_DEP_MONTHLY])


def load_dep_period(path: Path) -> pl.DataFrame:
    """Load and standardize the period_dep_lwf deposition file.

    The chemistry columns are prefixed with ``dep_period``.
    """
    print("Loading period_dep_lwf ...")
    raw = read_semicolon_csv(path)
    start_cleaning_stats(DS_DEP_PERIOD, raw)
    df = preprocess_dep_period(raw)
    df = remove_implausible_dep_period(df)
    df = remove_incomplete_dep_period(df)
    value_cols = ["plot_name_raw", *DEPOSITION_MEASUREMENTS]
    return finalize_loader(df, raw, DS_DEP_PERIOD, value_cols, DATASET_PREFIX[DS_DEP_PERIOD])


def load_litter_monthly_davos(path: Path) -> pl.DataFrame:
    """Load and standardize the Davos ICOS monthly litterfall file.

    The network, fraction text and weight columns are prefixed with
    ``litter_monthly_davos``.
    """
    print("Loading monthly_litterfall_Davos_ICOS ...")
    raw = read_litterfall_csv(path)
    start_cleaning_stats(DS_LITTER_MONTHLY_DAVOS, raw)
    df = preprocess_litter_monthly_davos(raw)
    df = remove_implausible_litter_monthly_davos(df)
    df = remove_incomplete_litter_monthly_davos(df)
    value_cols = ["Network", "Text_Fraction", "Weight"]
    prefix = DATASET_PREFIX[DS_LITTER_MONTHLY_DAVOS]
    return finalize_loader(df, raw, DS_LITTER_MONTHLY_DAVOS, value_cols, prefix)


def load_litter_monthly_lwf(path: Path) -> pl.DataFrame:
    """Load and standardize the LWF monthly litterfall file.

    The network, day count, fraction text and weight columns are prefixed with
    ``litter_monthly_lwf``.
    """
    print("Loading monthly_litterfall_LWF ...")
    raw = read_litterfall_csv(path)
    start_cleaning_stats(DS_LITTER_MONTHLY_LWF, raw)
    df = preprocess_litter_monthly_lwf(raw)
    df = remove_implausible_litter_monthly_lwf(df)
    df = remove_incomplete_litter_monthly_lwf(df)
    value_cols = ["Network", "Number_of_days", "Text_Fraction_2", "Weight"]
    prefix = DATASET_PREFIX[DS_LITTER_MONTHLY_LWF]
    return finalize_loader(df, raw, DS_LITTER_MONTHLY_LWF, value_cols, prefix)


def load_litter_period_davos(path: Path) -> pl.DataFrame:
    """Load and standardize the Davos ICOS period litterfall file.

    The network, period id, fraction text and weight columns are prefixed with
    ``litter_period_davos``.
    """
    print("Loading period_litterfall_Davos_ICOS ...")
    raw = read_litterfall_csv(path)
    start_cleaning_stats(DS_LITTER_PERIOD_DAVOS, raw)
    df = preprocess_litter_period_davos(raw)
    df = remove_implausible_litter_period_davos(df)
    df = remove_incomplete_litter_period_davos(df)
    value_cols = ["Network", "Period_id", "Text_Fraction", "Weight"]
    prefix = DATASET_PREFIX[DS_LITTER_PERIOD_DAVOS]
    return finalize_loader(df, raw, DS_LITTER_PERIOD_DAVOS, value_cols, prefix)


def load_litter_period_lwf(path: Path) -> pl.DataFrame:
    """Load and standardize the LWF period litterfall file.

    The network, day count, both fraction texts and the weight columns are
    prefixed with ``litter_period_lwf``.
    """
    print("Loading period_litterfall_LWF ...")
    raw = read_litterfall_csv(path)
    start_cleaning_stats(DS_LITTER_PERIOD_LWF, raw)
    df = preprocess_litter_period_lwf(raw)
    df = remove_implausible_litter_period_lwf(df)
    df = remove_incomplete_litter_period_lwf(df)
    value_cols = ["Network", "Number_of_days", "Text_Fraction_1", "Text_Fraction_2", "Weight"]
    prefix = DATASET_PREFIX[DS_LITTER_PERIOD_LWF]
    return finalize_loader(df, raw, DS_LITTER_PERIOD_LWF, value_cols, prefix)


# ============================================================================
# STEP 3 - MERGING THE CLEANED DATASETS
# ============================================================================


def coalesce_universal(staged: pl.DataFrame) -> pl.DataFrame:
    """Combine rows that share a strict key into universal rows.

    Rows are grouped on ``__strict_key__`` alone. The key is built from each
    dataset's own natural key with nulls made unique, so only rows that match
    exactly on a complete key are combined, and identity columns cannot disagree
    within a group.

    For every column the group keeps its single non-null value. If a column holds
    more than one distinct non-null value in a group, the merged cell is set to
    null, the values are joined into a ``<column>__CONFLICT`` column and a row is
    added to the conflict report. Conflict columns that never hold a value are
    removed. No values are averaged, summed or otherwise aggregated.

    A summary of the row counts before and after combining is printed. The final
    table is returned with the identity columns first, followed by
    ``contributing_datasets`` and the remaining columns.
    """
    # Every column except the two helper columns is coalesced with the same rule.
    value_cols = [c for c in staged.columns if c not in ("__strict_key__", "__source_dataset__")]
    rows_before = staged.height

    # For each column, keep the first non-null value, the number of distinct
    # non-null values, and those distinct values as text for conflict reporting.
    agg_exprs = [pl.col("__source_dataset__").unique().sort().alias("__contrib__")]
    for c in value_cols:
        agg_exprs.append(pl.col(c).drop_nulls().first().alias(f"__val__{c}"))
        agg_exprs.append(pl.col(c).drop_nulls().n_unique().alias(f"__nu__{c}"))
        agg_exprs.append(
            pl.col(c).cast(pl.Utf8, strict=False).drop_nulls().unique().alias(f"__vals__{c}")
        )

    grouped = staged.group_by("__strict_key__").agg(agg_exprs)
    # One universal row per strict key: no duplicate strict key survives.
    assert grouped["__strict_key__"].n_unique() == grouped.height
    assert grouped.height == staged["__strict_key__"].n_unique()
    grouped = grouped.with_columns(
        pl.col("__contrib__").list.join(", ").alias("contributing_datasets")
    )

    # Use the value when at most one distinct value exists, otherwise leave it null.
    final_val_exprs = [
        pl.when(pl.col(f"__nu__{c}") <= 1).then(pl.col(f"__val__{c}")).otherwise(None).alias(c)
        for c in value_cols
    ]
    grouped = grouped.with_columns(final_val_exprs)

    # Report each conflicting cell and keep its competing values in a companion
    # column.
    total_conflicts = 0
    conflict_val_exprs = []
    for c in value_cols:
        conf = grouped.filter(pl.col(f"__nu__{c}") > 1).select(
            value_cols
            + [
                "contributing_datasets",
                pl.lit(c).alias("column_name"),
                pl.col(f"__vals__{c}").list.join("; ").alias("conflicting_values"),
            ]
        )
        if conf.height > 0:
            CONFLICT_REPORT_FRAMES.append(conf)
            total_conflicts += conf.height
        conflict_val_exprs.append(
            pl.when(pl.col(f"__nu__{c}") > 1)
            .then(pl.col(f"__vals__{c}").list.join("; "))
            .otherwise(None)
            .alias(f"{c}__CONFLICT")
        )
    grouped = grouped.with_columns(conflict_val_exprs)

    # Remove the helper columns and any conflict columns that never hold a value.
    drop_cols = (
        ["__strict_key__", "__contrib__"]
        + [f"__val__{c}" for c in value_cols]
        + [f"__nu__{c}" for c in value_cols]
        + [f"__vals__{c}" for c in value_cols]
    )
    real_conflict_cols = [
        f"{c}__CONFLICT" for c in value_cols if grouped[f"{c}__CONFLICT"].drop_nulls().len() > 0
    ]
    empty_conflict_cols = [
        f"{c}__CONFLICT" for c in value_cols if f"{c}__CONFLICT" not in real_conflict_cols
    ]

    final = grouped.drop(drop_cols + empty_conflict_cols)
    other_cols = [
        c for c in final.columns if c not in IDENTITY_VOCAB_COLS and c != "contributing_datasets"
    ]
    final = final.select(IDENTITY_VOCAB_COLS + ["contributing_datasets"] + other_cols)
    rows_after = final.height

    print()
    print("=== COALESCING SUMMARY ===")
    print(f"Universal rows before coalescing: {rows_before}")
    print(f"Universal rows after coalescing:  {rows_after}")
    print(
        "Rows merged by genuine (null-safe, exact, non-NULL) key matches: "
        f"{rows_before - rows_after}"
    )
    print(f"Conflicting cells found: {total_conflicts}")

    return final


def run_cleaning_selfchecks() -> None:
    """Assert the cleaning rules on small synthetic frames before any data is read.

    The checks cover missing markers and infinities (in every case), invalid type
    conversion, inclusive and strict lower bounds, the removal counts and the
    behaviour of the loose and strict keys. They touch no report accumulator.
    """
    print("=== CLEANING SELF-CHECKS ===")
    schema = {"weight": "float", "days": "int", "when": "date", "code": "str"}
    raw = pl.DataFrame(
        {
            "weight": ["1.5", "0", "-2", "NA", "inf", "-Infinity", "INFINITY", "abc"]
            + ["N/A", ".", "", "7"],
            "days": ["10", "0", "x", "5.5", "3", "NA"] + ["10"] * 6,
            "when": ["2020-01-01"] * 10 + ["notadate", "N/A"],
            "code": ["A", " NA ", "N/A", ".", "", "B", "C", "D", "E", "F", "G", "H"],
        }
    )

    # Step 1: markers and infinities -> null, invalid types -> null, zero stays.
    stats = new_cleaning_stats(raw.height)
    typed = standardize_types(raw, schema, stats, "selfcheck")
    assert typed["weight"].to_list() == [1.5, 0.0, -2.0] + [None] * 8 + [7.0]
    assert typed["days"].to_list() == [10, 0, None, None, 3, None] + [10] * 6
    assert typed["when"].null_count() == 2
    assert typed["code"].to_list()[:5] == ["A", None, None, None, None]
    assert stats["cells_missing_or_infinite_to_null"] == 7 + 1 + 1 + 4
    assert stats["cells_invalid_type_to_null"] == 1 + 2 + 1

    # Step 2: zero is valid for an inclusive bound and rejected for a strict one.
    inclusive = apply_plausible_ranges(typed, {"weight": Bound(lower=0)}, new_cleaning_stats())
    assert inclusive["weight"].to_list()[:3] == [1.5, 0.0, None]
    strict_stats = new_cleaning_stats()
    strict = apply_plausible_ranges(
        typed, {"weight": Bound(lower=0, strict_lower=True)}, strict_stats
    )
    assert strict["weight"].to_list()[:3] == [1.5, None, None]
    assert strict_stats["cells_out_of_range_to_null"] == 2
    upper = apply_plausible_ranges(
        typed.with_columns(pl.Series("w2", [1, 8, 9] + [None] * 9, dtype=pl.Float64)),
        {"w2": Bound(lower=0, upper=8)},
        new_cleaning_stats(),
    )
    assert upper["w2"].to_list()[:3] == [1.0, 8.0, None]

    # Step 3: each removed row is counted once, by the step that emptied it.
    range_stats = new_cleaning_stats()
    ranged = apply_plausible_ranges(typed, {"weight": Bound(lower=0)}, range_stats)
    kept = remove_incomplete_rows(ranged, Completeness(all_of=("weight",)), range_stats)
    assert kept["weight"].to_list() == [1.5, 0.0, 7.0]
    assert range_stats["rows_removed_missing_values"] == 7
    assert range_stats["rows_removed_invalid_type"] == 1
    assert range_stats["rows_removed_implausible"] == 1

    # Keys: a null key field never matches; identical complete keys do.
    keyed = add_key_signatures(
        pl.DataFrame({"a": ["x", "x", None, None], "b": ["1", "1", "2", "2"]}), ["a", "b"], "t"
    )
    strict_keys, loose_keys = keyed["__strict_key__"], keyed["__loose_key__"]
    assert strict_keys[0] == strict_keys[1]
    assert strict_keys[2] != strict_keys[3]
    assert loose_keys[2] == loose_keys[3]
    print("All cleaning self-checks passed.")


def run_explicit_validations(final: pl.DataFrame, staged: pl.DataFrame) -> None:
    """Assert the cleaning and merge guarantees on the staged and merged tables.

    The checks cover the plausible ranges, row-count reconciliation per dataset,
    the LAI natural key, the natural-key definitions, unique strict keys, the
    conflict columns and the site_name crosswalk. A failed check raises an
    AssertionError.
    """
    print()
    print("=== EXPLICIT VALIDATION CHECKS ===")

    # Check 1: no out-of-range value, NaN or infinity remains in a measurement column.
    n_checked = 0
    for name, ranges in PLAUSIBLE_RANGES.items():
        for col, bound in ranges.items():
            prefixed = f"{DATASET_PREFIX[name]}__{col}"
            if prefixed not in staged.columns:
                continue
            for label, table in (("staged", staged), ("final", final)):
                n_bad = _count(table, out_of_range_expr(prefixed, bound))
                assert n_bad == 0, f"{label}: {prefixed} holds {n_bad} out-of-range value(s)"
            n_checked += 1
    for label, table in (("staged", staged), ("final", final)):
        for col, dtype in table.schema.items():
            if dtype == pl.Float64:
                assert table[col].is_nan().sum() == 0, f"{label}: NaN in {col}"
                assert table[col].is_infinite().sum() == 0, f"{label}: infinity in {col}"
    print(f"1. {n_checked} plausible-range rules hold in the staged and final tables; no NaN/inf.")

    # Check 2: the documented row counts reconcile with the real frames.
    staged_counts = dict(staged.group_by("__source_dataset__").len().iter_rows())
    for r in DATASET_REPORT_ROWS:
        removed = (
            r["rows_removed_missing_values"]
            + r["rows_removed_invalid_type"]
            + r["rows_removed_implausible"]
        )
        assert r["original_rows"] - removed == r["rows_after_cleaning"], r["dataset"]
        assert (
            r["rows_after_cleaning"] - r["exact_source_rows_removed"]
            == (r["rows_after_standardization"])
        ), r["dataset"]
        assert staged_counts.get(r["dataset"], 0) == r["rows_after_standardization"], r["dataset"]
    total_raw = sum(r["original_rows"] for r in DATASET_REPORT_ROWS)
    total_removed = sum(
        r["rows_removed_missing_values"]
        + r["rows_removed_invalid_type"]
        + r["rows_removed_implausible"]
        for r in DATASET_REPORT_ROWS
    )
    total_exact = sum(r["exact_source_rows_removed"] for r in DATASET_REPORT_ROWS)
    assert total_raw - total_removed - total_exact == staged.height
    print(
        f"2. Row conservation: {total_raw} raw rows - {total_removed} cleaned away - "
        f"{total_exact} exact duplicates = {staged.height} staged rows -> "
        f"{final.height} final rows."
    )

    # Check 3: LAI plot types stay separate. Rows that share a complete LAI key are
    # exactly the natural-key collisions that were reported and kept (never merged
    # or deleted).
    lai_key = NATURAL_KEYS[DS_LAI]
    lai_dup = final.drop_nulls(lai_key).select(lai_key).is_duplicated().sum()
    lai_collisions = next(
        (
            r["natural_key_collisions_retained"]
            for r in DATASET_REPORT_ROWS
            if r["dataset"] == DS_LAI
        ),
        0,
    )
    assert lai_dup == lai_collisions, (lai_dup, lai_collisions)
    print(f"3. LAI rows sharing a complete key: {int(lai_dup)} (= reported collisions, all kept).")

    # Check 4: the natural-key definitions are intact and site_name is never a key.
    for name, key_cols in NATURAL_KEYS.items():
        assert "site_name" not in key_cols, name
        assert set(key_cols) <= set(IDENTITY_VOCAB_COLS), name
    assert "site_name" not in IDENTITY_VOCAB_COLS
    print("4. Natural keys use identity columns only; site_name is never part of a key.")

    # Check 5: no duplicate strict key remains after coalescing.
    assert final.height == staged["__strict_key__"].n_unique()
    print("5. One universal row per strict key (no duplicate strict keys after coalescing).")

    # Check 6: every conflict is preserved in its __CONFLICT column and the report.
    conflict_cols = [c for c in final.columns if c.endswith("__CONFLICT")]
    n_conflict_cells = 0
    for conflict_col in conflict_cols:
        typed_col = conflict_col.removesuffix("__CONFLICT")
        has_conflict = pl.col(conflict_col).is_not_null()
        assert _count(final, has_conflict & pl.col(typed_col).is_not_null()) == 0, typed_col
        n_conflict_cells += _count(final, has_conflict)
    n_reported = sum(frame.height for frame in CONFLICT_REPORT_FRAMES)
    assert n_conflict_cells == n_reported, (n_conflict_cells, n_reported)
    for frame in CONFLICT_REPORT_FRAMES:
        assert frame["conflicting_values"].null_count() == 0
    print(
        f"6. {n_conflict_cells} conflicting cell(s): typed value null, competing values kept in "
        "<column>__CONFLICT and in the conflict report; nothing was averaged or summed."
    )

    # Check 7: site_name was derived from location_code through the crosswalk.
    expected_sites = [SITE_CROSSWALK.get(c) for c in final["location_code"].to_list()]
    assert final["site_name"].to_list() == expected_sites
    print("7. site_name matches SITE_CROSSWALK for every row.")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Parse arguments, run the merge pipeline and write the output files.

    The cleaning rules are self-checked, then the nine source datasets are
    loaded and cleaned, stacked, combined, given site names and checked. The
    parquet table and the CSV audit reports are then written to the output
    directory. The process exits with status 1 if an expected input file is
    missing.
    """
    parser = argparse.ArgumentParser(description="Merge LWF datasets into one universal table.")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("."), help="Directory containing source files"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./output"), help="Output directory")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expected location of each source file inside the data directory.
    paths = {
        "lai": data_dir / "data/LAI_Licor_3_rings_all_years.xlsx",
        "foliage_i": data_dir / "data/lwf_foliage_dw100_i_2026-07-30.csv",
        "foliage_plot": data_dir / "data/lwf_foliage_dw100_plot_2026-07-30.csv",
        "dep_monthly": data_dir / "data/monthly_dep_lwf_2026-07-28.csv",
        "dep_period": data_dir / "data/period_dep_lwf_2026-07-28.csv",
        "litter_monthly_davos": data_dir / "data/monthly_litterfall_Davos_ICOS_2026-07-30.csv",
        "litter_monthly_lwf": data_dir / "data/monthly_litterfall_LWF_2026-07-30.csv",
        "litter_period_davos": data_dir / "data/period_litterfall_Davos_ICOS_2026-07-30.csv",
        "litter_period_lwf": data_dir / "data/period_litterfall_LWF_2026-07-30.csv",
    }

    # Stop early if any expected input file is missing.
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print("ERROR: the following expected input files were not found:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    run_cleaning_selfchecks()
    print()

    # One loader per source file, in the same order as the entries in paths.
    loaders = [
        load_lai,
        load_foliage_i,
        load_foliage_plot,
        load_dep_monthly,
        load_dep_period,
        load_litter_monthly_davos,
        load_litter_monthly_lwf,
        load_litter_period_davos,
        load_litter_period_lwf,
    ]

    frames = [loader(paths[key]) for loader, key in zip(loaders, paths.keys(), strict=False)]

    print()
    print(f"Concatenating {len(frames)} standardized datasets ...")
    # Stack the frames, filling columns a dataset lacks with nulls.
    staged = pl.concat(frames, how="diagonal_relaxed")

    final = coalesce_universal(staged)
    final = attach_site_name(final)  # Descriptive only; never used as a key.

    run_explicit_validations(final, staged)

    # Write the merged table and the audit reports.
    out_parquet = out_dir / "LWF_universal_dataset.parquet"
    final.write_parquet(out_parquet)

    report_df = pl.DataFrame(DATASET_REPORT_ROWS)
    report_path = out_dir / "LWF_merge_report.csv"
    report_df.write_csv(report_path)

    if DUPLICATE_REPORT_FRAMES:
        dup_report = pl.concat(DUPLICATE_REPORT_FRAMES, how="diagonal_relaxed")
        dup_report.write_csv(out_dir / "LWF_duplicate_keys_report.csv")
    else:
        dup_report = pl.DataFrame()

    if CONFLICT_REPORT_FRAMES:
        conflict_report = pl.concat(CONFLICT_REPORT_FRAMES, how="diagonal_relaxed")
    else:
        conflict_report = pl.DataFrame(
            schema={
                "contributing_datasets": pl.String,
                "column_name": pl.String,
                "conflicting_values": pl.String,
            }
        )
    conflict_report.write_csv(out_dir / "LWF_conflict_report.csv")

    # Print a summary of the run.
    print()
    print("=== FINAL SUMMARY ===")
    print(f"Number of source datasets: {len(frames)}")
    print(f"Number of rows in final dataset: {final.height}")
    print(f"Number of columns in final dataset: {final.width}")
    print(f"Number of unique locations: {final['location_code'].n_unique()}")
    print(f"Number of unique sites (site_name): {final['site_name'].n_unique()}")
    n_missing = sum(r["rows_removed_missing_values"] for r in DATASET_REPORT_ROWS)
    n_invalid = sum(r["rows_removed_invalid_type"] for r in DATASET_REPORT_ROWS)
    n_implausible = sum(r["rows_removed_implausible"] for r in DATASET_REPORT_ROWS)
    print(
        f"Rows removed during cleaning: {n_missing} missing values, "
        f"{n_invalid} invalid data types, {n_implausible} implausible values"
    )
    n_exact = sum(r["exact_source_duplicates"] for r in DATASET_REPORT_ROWS)
    n_exact_removed = sum(r["exact_source_rows_removed"] for r in DATASET_REPORT_ROWS)
    n_ambig = sum(r["ambiguous_incomplete_key"] for r in DATASET_REPORT_ROWS)
    n_collisions = sum(r["natural_key_collisions_retained"] for r in DATASET_REPORT_ROWS)
    print(f"Exact duplicate source rows flagged: {n_exact} ({n_exact_removed} copies removed)")
    print(f"Ambiguous (incomplete-key) rows flagged across all source datasets: {n_ambig}")
    print(f"Natural-key collisions retained separately: {n_collisions}")
    print(f"Rows in conflict report: {conflict_report.height}")
    print()
    print(f"Universal dataset written to: {out_parquet}")
    print(f"Merge report written to:      {report_path}")
    if dup_report.height:
        print(f"Duplicate-key report written to: {out_dir / 'LWF_duplicate_keys_report.csv'}")
    if conflict_report.height:
        print(f"Conflict report written to:      {out_dir / 'LWF_conflict_report.csv'}")


if __name__ == "__main__":
    main()
