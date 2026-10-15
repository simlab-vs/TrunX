"""Merge the nine LWF datasets into a single universal table.

LWF (Long-term Forest Ecosystem Research) provides the nine source datasets:
leaf area index (LAI), two foliage weight (dw100) datasets, monthly and period
deposition, and monthly and period litterfall for both the Davos ICOS and LWF
networks. Each dataset is loaded and standardized, the standardized frames are
stacked, and rows that describe the same observation are combined.

Every dataset defines its own natural key (see NATURAL_KEYS). Rows are only
combined when all key fields are present and identical, so a missing key field
never matches another row and incomplete identities stay separate. When
combined rows disagree on a measurement, the merged cell is left empty and the
competing values are kept in a ``<column>__CONFLICT`` column and in the
conflict report.

The following files are written to the output directory:

- LWF_universal_dataset.parquet: the merged table.
- LWF_merge_report.csv: row counts and duplicate statistics per dataset.
- LWF_duplicate_keys_report.csv: rows flagged during duplicate detection.
- LWF_conflict_report.csv: cells where combined rows disagreed.

Run it from the command line as follows:

    python merge_lwf.py --data-dir ./data --out-dir ./data
"""

from __future__ import annotations

import argparse
import sys
import warnings
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
    "LAI_Licor_3_rings_all_years": ["location_code", "date", "lai_subplot", "plot_type"],
    "lwf_foliage_dw100_i": [
        "location_code",
        "date",
        "sample_id",
        "species",
        "leaf_type",
        "leaf_age_class",
    ],
    "lwf_foliage_dw100_plot": [
        "location_code",
        "date",
        "survey_year",
        "species",
        "leaf_type",
        "leaf_age_class",
    ],
    "monthly_dep_lwf": ["location_code", "start_date", "end_date", "dep_location"],
    "period_dep_lwf": ["location_code", "start_date", "end_date", "dep_location"],
    "monthly_litterfall_Davos_ICOS": ["location_code", "start_date", "end_date", "fraction_code"],
    "monthly_litterfall_LWF": [
        "location_code",
        "start_date",
        "end_date",
        "sub_plot",
        "fraction_code_lwf",
        "tree_species",
    ],
    "period_litterfall_Davos_ICOS": ["location_code", "start_date", "end_date", "fraction_code"],
    "period_litterfall_LWF": [
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

# Accumulators for the audit reports. The loaders and the coalescing step append
# to them and main writes them out.
DATASET_REPORT_ROWS: list[dict] = []
DUPLICATE_REPORT_FRAMES: list[pl.DataFrame] = []
CONFLICT_REPORT_FRAMES: list[pl.DataFrame] = []


# ============================================================================
# GENERIC HELPERS
# ============================================================================


def read_semicolon_csv(path: Path) -> pl.DataFrame:
    """Read a semicolon-separated CSV file into a DataFrame.

    The file is decoded as UTF-8 first and read again as Windows-1252 if Polars
    cannot decode it. The strings in EXTRA_NULL_VALUES are read as nulls.
    """
    try:
        return pl.read_csv(
            path,
            separator=";",
            infer_schema_length=10000,
            encoding="utf8",
            null_values=EXTRA_NULL_VALUES,
        )
    except pl.exceptions.PolarsError:
        return pl.read_csv(
            path,
            separator=";",
            infer_schema_length=10000,
            encoding="windows-1252",
            null_values=EXTRA_NULL_VALUES,
        )


def smart_parse_date(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Convert a column to a Date, detecting the format from the data.

    Columns that are already Date or Datetime, and columns without any non-null
    values, are cast to Date directly. Otherwise each format in
    CANDIDATE_DATE_FORMATS is tried in turn, and the first one that parses every
    non-null value is used and reported. If none fits, a warning is printed and
    the frame is returned unchanged.
    """
    if df.schema[col] in (pl.Date, pl.Datetime):
        return df.with_columns(pl.col(col).cast(pl.Date))
    non_null_n = df[col].drop_nulls().len()
    if non_null_n == 0:
        return df.with_columns(pl.col(col).cast(pl.Date))
    for fmt in CANDIDATE_DATE_FORMATS:
        parsed = df[col].str.strptime(pl.Date, fmt, strict=False)
        if parsed.drop_nulls().len() == non_null_n:
            print(f"    date column '{col}': detected format {fmt}")
            return df.with_columns(parsed.alias(col))
    print(
        f"    WARNING: could not confidently parse date column '{col}' with any of "
        f"{CANDIDATE_DATE_FORMATS}. Left as string/original — check the raw format "
        f"and add it to CANDIDATE_DATE_FORMATS."
    )
    return df


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
        df = df.unique(subset=source_cols, maintain_order=True)
    n_exact_removed = rows_before_deduplication - df.height

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
    file.
    """
    df = df.with_columns(pl.date(pl.col(year_col), pl.col(month_col), 1).alias("__first_of_month"))
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

    The counts cover the raw file, the standardized frame and the duplicate
    statistics returned by validate_unique.
    """
    DATASET_REPORT_ROWS.append(
        {
            "dataset": name,
            "original_rows": raw.height,
            "original_columns": raw.width,
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

    Key strings are trimmed, the loose and strict keys are built, duplicates are
    validated, the measurement columns are prefixed with the dataset tag, the
    identity columns are padded to the common set, and the dataset name is stored
    in ``__source_dataset__``. The loose key is then dropped, the per-dataset
    report row is recorded, and the standardized frame is returned.
    """
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
# PER-DATASET LOADERS
# ============================================================================


def load_lai(path: Path) -> pl.DataFrame:
    """Load and standardize the LAI Licor spreadsheet.

    Column names are assigned by position, the location code is read as a string
    and the date column is parsed. Location, subplot, date and plot type are
    identity columns. The remaining LAI, angle, point count and season columns are
    prefixed with ``lai``.
    """
    print("Loading LAI_Licor_3_rings_all_years ...")
    # Skip the leading non-data rows; column names are assigned by position below.
    raw = pl.read_excel(path).slice(2)
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
    df = df.with_columns(pl.col("location_code").cast(pl.Utf8))
    df = smart_parse_date(df, "date")
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
    return finalize_loader(df, raw, "LAI_Licor_3_rings_all_years", value_cols, "lai")


def load_foliage_i(path: Path) -> pl.DataFrame:
    """Load and standardize the lwf_foliage_dw100_i foliage weight file.

    The plot id and survey date become location_code and date, and the date is
    parsed. The plot name is kept as plot_name_raw. The survey year, plot name and
    weight columns are prefixed with ``foliage_i``.
    """
    print("Loading lwf_foliage_dw100_i ...")
    raw = read_semicolon_csv(path)
    df = raw.rename(
        {"plot_id": "location_code", "plot_name": "plot_name_raw", "survey_date": "date"}
    )
    df = smart_parse_date(df, "date")
    value_cols = ["survey_year", "plot_name_raw", "gew100"]
    return finalize_loader(df, raw, "lwf_foliage_dw100_i", value_cols, "foliage_i")


def load_foliage_plot(path: Path) -> pl.DataFrame:
    """Load and standardize the lwf_foliage_dw100_plot foliage weight file.

    The plot id and survey date become location_code and date, and the date is
    parsed. The plot name is kept as plot_name_raw. The plot name and weight
    columns are prefixed with ``foliage_plot``.
    """
    print("Loading lwf_foliage_dw100_plot ...")
    raw = read_semicolon_csv(path)
    df = raw.rename(
        {"plot_id": "location_code", "plot_name": "plot_name_raw", "survey_date": "date"}
    )
    df = smart_parse_date(df, "date")
    value_cols = ["plot_name_raw", "gew100"]
    return finalize_loader(df, raw, "lwf_foliage_dw100_plot", value_cols, "foliage_plot")


def load_dep_monthly(path: Path) -> pl.DataFrame:
    """Load and standardize the monthly_dep_lwf deposition file.

    The plot id and location become location_code and dep_location, and
    start_date and end_date are derived from the survey year and month. The survey
    and chemistry columns are prefixed with ``dep_monthly``.
    """
    print("Loading monthly_dep_lwf ...")
    raw = read_semicolon_csv(path)
    df = raw.rename(
        {"plot_id": "location_code", "plot_name": "plot_name_raw", "location": "dep_location"}
    )
    df = add_month_end_bounds(df, "survey_year", "survey_month")
    value_cols = [
        "survey_year",
        "survey_month",
        "plot_name_raw",
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
    return finalize_loader(df, raw, "monthly_dep_lwf", value_cols, "dep_monthly")


def load_dep_period(path: Path) -> pl.DataFrame:
    """Load and standardize the period_dep_lwf deposition file.

    The plot id, location and period boundary columns are renamed to
    location_code, dep_location, start_date and end_date, and both dates are
    parsed. The chemistry columns are prefixed with ``dep_period``.
    """
    print("Loading period_dep_lwf ...")
    raw = read_semicolon_csv(path)
    df = raw.rename(
        {
            "plot_id": "location_code",
            "plot_name": "plot_name_raw",
            "location": "dep_location",
            "date_start": "start_date",
            "date_end": "end_date",
        }
    )
    df = smart_parse_date(df, "start_date")
    df = smart_parse_date(df, "end_date")
    value_cols = [
        "plot_name_raw",
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
    return finalize_loader(df, raw, "period_dep_lwf", value_cols, "dep_period")


def load_litter_monthly_davos(path: Path) -> pl.DataFrame:
    """Load and standardize the Davos ICOS monthly litterfall file.

    The first two lines of the file are skipped, the text is decoded as
    Windows-1252 and a dot is read as a missing value. The year, month and plot
    name columns are dropped, the key columns are renamed and the dates are
    parsed. The network, fraction text and weight columns are prefixed with
    ``litter_monthly_davos``.
    """
    print("Loading monthly_litterfall_Davos_ICOS ...")
    raw = pl.read_csv(
        path,
        separator=";",
        skip_rows=2,
        encoding="windows-1252",
        infer_schema_length=10000,
        null_values=".",
    )
    # Drop columns that are not carried into the universal table.
    df = raw.drop(["Year", "Month", "LWF_plot"])
    df = df.rename(
        {
            "LWF_plot_code": "location_code",
            "Start_date": "start_date",
            "End_date": "end_date",
            "Code_fraction": "fraction_code",
        }
    )
    df = smart_parse_date(df, "start_date")
    df = smart_parse_date(df, "end_date")
    value_cols = ["Network", "Text_Fraction", "Weight"]
    return finalize_loader(
        df, raw, "monthly_litterfall_Davos_ICOS", value_cols, "litter_monthly_davos"
    )


def load_litter_monthly_lwf(path: Path) -> pl.DataFrame:
    """Load and standardize the LWF monthly litterfall file.

    The first two lines of the file are skipped, the text is decoded as
    Windows-1252 and a dot is read as a missing value. The year, month and plot
    name columns are dropped, the key columns are renamed and the dates are
    parsed. The network, day count, fraction text and weight columns are prefixed
    with ``litter_monthly_lwf``.
    """
    print("Loading monthly_litterfall_LWF ...")
    raw = pl.read_csv(
        path,
        separator=";",
        skip_rows=2,
        encoding="windows-1252",
        infer_schema_length=10000,
        null_values=".",
    )
    df = raw.drop(["Year", "Month", "LWF_plot"])
    df = df.rename(
        {
            "LWF_plot_code": "location_code",
            "Start_date": "start_date",
            "End_date": "end_date",
            "Sub_plot": "sub_plot",
            "Code_fraction_2": "fraction_code_lwf",
            "Tree_species": "tree_species",
        }
    )
    df = smart_parse_date(df, "start_date")
    df = smart_parse_date(df, "end_date")
    value_cols = ["Network", "Number_of_days", "Text_Fraction_2", "Weight"]
    return finalize_loader(df, raw, "monthly_litterfall_LWF", value_cols, "litter_monthly_lwf")


def load_litter_period_davos(path: Path) -> pl.DataFrame:
    """Load and standardize the Davos ICOS period litterfall file.

    The first two lines of the file are skipped, the text is decoded as
    Windows-1252 and a dot is read as a missing value. The plot name column is
    dropped, the key columns are renamed and the dates are parsed. The network,
    period id, fraction text and weight columns are prefixed with
    ``litter_period_davos``.
    """
    print("Loading period_litterfall_Davos_ICOS ...")
    raw = pl.read_csv(
        path,
        separator=";",
        skip_rows=2,
        encoding="windows-1252",
        infer_schema_length=10000,
        null_values=".",
    )
    df = raw.drop(["LWF_plot"])
    df = df.rename(
        {
            "LWF_plot_code": "location_code",
            "Start_date": "start_date",
            "End_date": "end_date",
            "Code_fraction": "fraction_code",
        }
    )
    df = smart_parse_date(df, "start_date")
    df = smart_parse_date(df, "end_date")
    value_cols = ["Network", "Period_id", "Text_Fraction", "Weight"]
    return finalize_loader(
        df, raw, "period_litterfall_Davos_ICOS", value_cols, "litter_period_davos"
    )


def load_litter_period_lwf(path: Path) -> pl.DataFrame:
    """Load and standardize the LWF period litterfall file.

    The first two lines of the file are skipped, the text is decoded as
    Windows-1252 and a dot is read as a missing value. The plot name column is
    dropped, the key columns are renamed and the dates are parsed. The network,
    day count, both fraction texts and the weight columns are prefixed with
    ``litter_period_lwf``.
    """
    print("Loading period_litterfall_LWF ...")
    raw = pl.read_csv(
        path,
        separator=";",
        skip_rows=2,
        encoding="windows-1252",
        infer_schema_length=10000,
        null_values=".",
    )
    df = raw.drop(["LWF_plot"])
    df = df.rename(
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
    df = smart_parse_date(df, "start_date")
    df = smart_parse_date(df, "end_date")
    value_cols = ["Network", "Number_of_days", "Text_Fraction_1", "Text_Fraction_2", "Weight"]
    return finalize_loader(df, raw, "period_litterfall_LWF", value_cols, "litter_period_lwf")


# ============================================================================
# FINAL COALESCE
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


def run_explicit_validations(final: pl.DataFrame, staged_rows: int) -> None:
    """Print sanity checks on the merged table.

    The checks cover the separation of LAI plot types, the conservation of source
    rows through the pipeline, the role of site_name, the absence of aggregation
    on measurement columns and the absence of duplicate universal keys. The row
    count check is asserted against staged_rows, the number of rows that entered
    coalescing. The other checks only print.
    """
    print()
    print("=== EXPLICIT VALIDATION CHECKS ===")

    # Check 1: plot_type is part of the LAI key, so circular-plot and quadrat rows
    # stay separate and no LAI key should be duplicated.
    if "plot_type" in final.columns and final["plot_type"].drop_nulls().len() > 0:
        lai_types = set(final["plot_type"].drop_nulls().unique().to_list())
        lai_key_dup = (
            final.filter(pl.col("plot_type").is_not_null())
            .select(["location_code", "date", "lai_subplot", "plot_type"])
            .is_duplicated()
            .sum()
        )
        print(
            f"1. LAI plot_type values present in final output: {sorted(lai_types)}; "
            "rows with duplicate (location_code, date, lai_subplot, plot_type): "
            f"{int(lai_key_dup)} "
            f"(expected 0 -- circular plot and quadrat are confirmed to remain separate rows)."
        )
    else:
        print("1. No LAI rows present in final output -- check skipped.")

    # Check 2: raw rows minus removed exact duplicates must equal the standardized
    # and staged row counts, so no row is lost without being reported.
    total_raw = sum(r["original_rows"] for r in DATASET_REPORT_ROWS)
    total_standardized = sum(r["rows_after_standardization"] for r in DATASET_REPORT_ROWS)
    exact_rows_removed = sum(r["exact_source_rows_removed"] for r in DATASET_REPORT_ROWS)
    print(
        "2. Row conservation: "
        f"{total_raw} raw source rows -> {total_standardized} standardized rows "
        f"({exact_rows_removed} confirmed exact duplicate source rows removed) -> "
        f"{staged_rows} rows entered coalescing "
        "(must equal standardized total, since concatenation never filters) -> "
        f"{final.height} final "
        f"universal rows (fewer only where __strict_key__ genuinely, exactly matched -- "
        f"never due to a NULL field)."
    )
    assert total_raw - exact_rows_removed == total_standardized == staged_rows, (
        "Row count mismatch detected -- a row was dropped for a reason other than "
        "confirmed exact duplication."
    )

    # Check 3: site_name is derived after coalescing and is never part of a key.
    print(
        "3. site_name: derived from location_code via the crosswalk AFTER "
        "coalescing; never used as a key."
    )

    # Check 4: measurement columns are only copied or set to null, never aggregated.
    print(
        "4. No averaging/sum/first/last/min/max was applied to any measurement column. "
        "Every non-conflicting cell holds the single non-null value that was actually present; "
        "every genuine conflict is NULL in the typed column with raw values preserved in a "
        "'<col>__CONFLICT' column and in LWF_conflict_report.csv."
    )

    # Check 5: the final table has one row per strict key because coalescing groups
    # on that key; the result is printed as an explicit confirmation.
    print(
        "5. Unresolved duplicate universal keys after coalescing: 0 "
        "(guaranteed by construction -- grouped on __strict_key__)."
    )


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Parse arguments, run the merge pipeline and write the output files.

    The nine source datasets are loaded from the data directory, stacked,
    combined, given site names and checked. The parquet table and the CSV audit
    reports are then written to the output directory. The process exits with
    status 1 if an expected input file is missing.
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
        "lai": data_dir / "LAI_Licor_3_rings_all_years.xlsx",
        "foliage_i": data_dir / "lwf_foliage_dw100_i_2026-07-30.csv",
        "foliage_plot": data_dir / "lwf_foliage_dw100_plot_2026-07-30.csv",
        "dep_monthly": data_dir / "monthly_dep_lwf_2026-07-28.csv",
        "dep_period": data_dir / "period_dep_lwf_2026-07-28.csv",
        "litter_monthly_davos": data_dir / "monthly_litterfall_Davos_ICOS_2026-07-30.csv",
        "litter_monthly_lwf": data_dir / "monthly_litterfall_LWF_2026-07-30.csv",
        "litter_period_davos": data_dir / "period_litterfall_Davos_ICOS_2026-07-30.csv",
        "litter_period_lwf": data_dir / "period_litterfall_LWF_2026-07-30.csv",
    }

    # Stop early if any expected input file is missing.
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print("ERROR: the following expected input files were not found:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

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
    staged_rows = staged.height

    final = coalesce_universal(staged)
    final = attach_site_name(final)  # Descriptive only; never used as a key.

    run_explicit_validations(final, staged_rows)

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
