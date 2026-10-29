"""Reusable ICOS helpers: Level 2 download and ICOS ETC L2 -> 3-PG preprocessing functions.

``download(arg)`` is the downloader (formerly ICOS_download.py). Every other function was moved unchanged from the
former notebook icos_to_3pg.py (apart from the optional ``site_ids`` filter of ``discover_sites``) and is an ordinary
function: call them in this order (see icos_preprocessing.py)

    download -> discover_sites -> collect_station_metadata -> build_outputs

MISSING-DATA THRESHOLD (MAX_MISSING_PCT, one pipeline-wide constant, strictly-greater-than rule):
    MissingPct = 100 * missing expected observations / expected observations of one measurement in one site-month.
    A month with MissingPct > MAX_MISSING_PCT gets NO value for that measurement (decided per measurement, before any
    aggregation, so no join / pivot can bring it back); other measurements of the month are untouched.
    Applied   : climate_3pg (TA -> tmp_min/tmp_max/tmp_ave/frost_days, P -> prcp, SW_IN -> srad, VPD -> vpd_day,
                CO2_DRY -> co2) and observed_gpp (GPP_DT_VUT_REF -> GPP_DT and GPP_NT_VUT_REF -> GPP_NT from
                the half-hourly FLUXNET_HH file). GPP_total is calculated only after those independent aggregates.
                Denominator = days in the calendar month x records per day at the file's own cadence
                (modal timestamp step); timestamps absent from the file count as missing, -9999 / NaN / null / inf too,
                0 is valid. ICOS timestamps are local standard time (no DST), so every day has 24 h.
    Derived   : site_3pg from/to are the first / last month with a valid climate value AFTER thresholding; latitude /
                altitude are validated (finite, |lat| <= 90) instead of thresholded.
    Not applicable (no defensible expected-observation count in the source; nothing is invented): species_3pg,
                observed_lai, observed_total_biomass and the ICOS-aggregate part of observed_dbh (one self-contained
                BADM record per date: valid or absent, ambiguity rules already null conflicting records) and the LWF
                individual-tree DBH (a tree list without an expected-measurement status).

The environment variables ICOS_USERNAME and ICOS_PASSWORD (a .env file is read) must hold the credentials of an ICOS
account (https://cpauth.icos-cp.eu).
"""

# ruff: noqa: D205, D401, E501

import json
import logging
import math
import os
import re
import urllib.parse
import urllib.request
import warnings
from collections.abc import Iterable
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from trunx.config import icos_raw_data_folder
from trunx.datasets.icos import (
    download_ICOS_L2,
    get_ICOS_clients,
    list_L2_station_ids,
)

# ---- download ----
PROXIMITY_FILE = "examples/Sites_Proximity_5km.parquet"  # only these ICOS sites are considered in an all-sites run

# ---- source data folder (download target; ICOSETC_* files and the LWF file live here) ----
DATA_DIR = Path(icos_raw_data_folder)

SENTINEL = -9999  # ICOS missing-value code (time series and ANCILLARY DATAVALUE)
# Pipeline-wide missing-data threshold [%]: a site-month of a measurement is discarded when its missing share is
# STRICTLY greater than this value (a month at exactly the threshold is kept). Valid range 0..100; 100 disables it.
MAX_MISSING_PCT = 50.0
SRAD_FACTOR = 0.0864  # W m-2 (daily mean) -> MJ m-2 d-1 : 86400 s d-1 * 1e-6 MJ J-1
ICOS_SPARQL_ENDPOINT = "https://meta.icos-cp.eu/sparql"
REQUIRED_PRODUCTS = ("METEO", "FLUXES", "ANCILLARY")  # FLUXNET_HH is optional: only source of GPP
# ICOSETC_<SITE_ID>_<PRODUCT>_<RELEASE>_L2...csv (case-insensitive; a trailing "(1)" is ignored)
FILE_RE = re.compile(
    r"^ICOSETC_(?P<cc>[A-Za-z]{2})-(?P<rest>[A-Za-z0-9]+)_(?P<product>METEO|FLUXES|ANCILLARY|FLUXNET_HH)"
    r"_(?P<release>[A-Za-z]+)_L2.*\.csv$",
    re.IGNORECASE,
)

# ---- output contract: exact column names and order of the seven final tables ----
CLIMATE_COLS = [
    "site_id",
    "year",
    "month",
    "tmp_min",
    "tmp_max",
    "tmp_ave",
    "prcp",
    "srad",
    "frost_days",
    "vpd_day",
    "co2",
    "d13catm",
]
SITE_COLS = ["site_id", "latitude", "altitude", "from", "to"]
COMPONENT_COLS = ["biom_stem", "biom_root", "biom_foliage"]
SPECIES_COLS = [
    "site_id",
    "observation_date",
    "year",
    "month",
    "species",
    "planted",
    "fertility",
    "stems_n",
    *COMPONENT_COLS,
]
OBSERVED_COLS = {
    "observed_gpp": [
        "plot_area_ha",
        "site_id",
        "observation_date",
        "year",
        "month",
        "GPP_DT",
        "GPP_NT",
        "GPP_total",
    ],
    "observed_lai": ["plot_area_ha", "site_id", "observation_date", "year", "month", "LAI"],
    "observed_total_biomass": [
        "plot_area_ha",
        "site_id",
        "observation_date",
        "year",
        "month",
        "species",
        "total_biomass",
        "biomass_ha",
    ],
    "observed_dbh": [
        "plot_area_ha",
        "area_site",
        "site",
        "date",
        "tree_id",
        "species",
        "dbh",
        "dbh_source",
    ],
}
# Columns that count as a measurement when deciding whether a row is metadata-only. Dates, years, months, ids,
# species names and plot_area_ha never qualify.
MEASUREMENT_COLS = {
    "climate_3pg": [
        "tmp_min",
        "tmp_max",
        "tmp_ave",
        "prcp",
        "srad",
        "frost_days",
        "vpd_day",
        "co2",
        "d13catm",
    ],
    "site_3pg": ["latitude", "altitude"],
    "species_3pg": ["planted", "fertility", "stems_n", *COMPONENT_COLS],
    "observed_gpp": ["GPP_DT", "GPP_NT", "GPP_total"],
    "observed_lai": ["LAI"],
    "observed_total_biomass": ["total_biomass"],
    "observed_dbh": ["dbh"],
}

# ---- ANCILLARY (BADM) ----
NUMERIC_VARS = {
    "TREES_NUM",
    "BIOMASS",
    "DBH",
    "LAI",
}  # measured values; every other variable is a descriptor
# Exact organ labels mapped to r3PG pools; anything else (e.g. "Total aboveground") is never split into pools.
ORGAN_MAP = {
    "stem": "biom_stem",
    "stems": "biom_stem",
    "stem wood": "biom_stem",
    "stemwood": "biom_stem",
    "foliage": "biom_foliage",
    "leaf": "biom_foliage",
    "leaves": "biom_foliage",
    "root": "biom_root",
    "roots": "biom_root",
    "total root": "biom_root",
    "total roots": "biom_root",
}
TOTAL_ORGAN_LABELS = ["total aboveground"]
# Area-based dry-matter units (lower-case, single spaces) -> t DM ha-1. Exact conversions only; carbon units
# (gC m-2 ...) are not dry matter and are deliberately absent, so they give a null biomass_ha.
DM_AREA_UNIT_TO_T_HA = {"kgdm m-2": 10.0, "gdm m-2": 0.01, "tdm ha-1": 1.0, "kgdm ha-1": 0.001}
KGDM_M2_TO_T_HA = DM_AREA_UNIT_TO_T_HA["kgdm m-2"]  # 1 kg DM m-2 = 10 t DM ha-1

# ---- GPP (FLUXNET_HH, half-hourly) ----
# The two VUT reference GPP fields are daytime- and nighttime-partitioning estimates of gross primary production,
# respectively. They are alternative estimates, not independent daytime/nighttime flux components.
GPP_SOURCE_COLUMNS = {"GPP_DT_VUT_REF": "GPP_DT", "GPP_NT_VUT_REF": "GPP_NT"}
GPP_OUTPUT_COLUMNS = list(GPP_SOURCE_COLUMNS.values())

# ---- LWF individual-tree DBH for CH-Dav (located by file name inside the data directory) ----
LWF_ICOS_SITE_ID = (
    "CH-Dav"  # ICOS station whose aggregated DBH is replaced by the LWF individual trees
)
LWF_SITENAME = "Davos"  # exact value of the LWF `sitename` column
LWF_FILENAME = "Givi_LWFdata_Jun26_flagged.csv"
LWF_REQUIRED_COLS = [
    "sitename",
    "parcelname",
    "year",
    "date_observation",
    "invnr",
    "banr",
    "banreti",
    "tree_species",
    "dbh",
    "height",
    "dbh_corrected",
]
LWF_NUMERIC_COLS = ["banr", "dbh", "dbh_corrected"]
LWF_MISSING_MARKERS = ["", "NA", "N/A", "NaN", "nan", "NULL", "null"]

# Site area (ha), entered manually from the DEIMS-SDR pages (deims.org); sites not listed get null.
#   CH-Dav: "Davos Seehornwald", ca. 1.10 ha.  SE-Svb: "Svartberget ICOS Station", experimental forest ca. 1520 ha.
# This is descriptive metadata only: it is NOT the area over which any biomass was measured.
PLOT_AREA_HA = {"CH-Dav": 1.10, "SE-Svb": 1520.00}
PLOT_AREA_BY_SITE_LABEL = {**PLOT_AREA_HA, LWF_SITENAME: PLOT_AREA_HA[LWF_ICOS_SITE_ID]}

LWF_PATH = (
    DATA_DIR / LWF_FILENAME
)  # LWF individual-tree file (not an ICOS product; must be placed in DATA_DIR)


def download(arg):
    """Download Level 2 products for one requested site or the available sites in the proximity Parquet file.

    arg : a site id (e.g. "CH-Dav", "BE-Bra") -> only that site is considered, provided it is in the ICOS Level 2
          station listing; None -> read ``PROXIMITY_FILE`` and consider only rows where ``network_b == "ICOS"``.

    In all-sites mode, the proximity-site list is intersected with the station IDs returned by
    ``list_L2_station_ids(meta_client)``. Only sites present in BOTH sources are downloaded. Proximity sites without
    an available Level 2 archive are logged and skipped. Products available for a site are downloaded
    (``products=None``, as in the former ICOS_download.py). A failed site's download is recorded and processing
    continues. Returns the requested, downloaded, unavailable and failed site IDs, with failure reasons for failures.
    """
    if arg is not None and (not isinstance(arg, str) or not arg.strip()):
        raise ValueError(f"download() expects a site id such as 'CH-Dav' or None, got {arg!r}")

    # Get credentials from .env file and initialize clients.
    load_dotenv()
    try:
        username = os.environ["ICOS_USERNAME"]
        password = os.environ["ICOS_PASSWORD"]
    except KeyError as e:
        raise RuntimeError(f"Environment variable {e} must be defined.") from None

    meta_client, data_client = get_ICOS_clients(username, password)

    # The ICOS metadata service is authoritative for which stations have a Level 2 archive.
    station_ids = list_L2_station_ids(meta_client)

    # Map IDs case-insensitively to the canonical ICOS station ID.
    station_id_map = {station_id.upper(): station_id for station_id in station_ids}

    if arg is None:
        # In all-sites mode, only consider ICOS sites listed in the proximity Parquet, then let the loop below
        # skip any of those that do not currently have a Level 2 archive.
        proximity_path = Path(PROXIMITY_FILE)
        if not proximity_path.is_file():
            raise FileNotFoundError(
                f"Proximity-site Parquet not found: {proximity_path.resolve()}. "
                "Run the downloader from the project root or update PROXIMITY_FILE."
            )

        proximity_sites = pl.read_parquet(proximity_path)
        required_proximity_cols = {"network_b", "site_id_b"}
        missing_proximity_cols = required_proximity_cols - set(proximity_sites.columns)
        if missing_proximity_cols:
            raise ValueError(
                f"{proximity_path}: missing required column(s) {sorted(missing_proximity_cols)}; "
                f"found columns {proximity_sites.columns}"
            )

        requested = (
            proximity_sites.select(
                pl.col("site_id_b").cast(pl.Utf8).str.strip_chars().alias("site_id_b")
            )
            .filter(pl.col("site_id_b").is_not_null() & (pl.col("site_id_b").str.len_chars() > 0))
            .unique()
            .sort("site_id_b")
            .get_column("site_id_b")
            .to_list()
        )
        logging.info(
            "Proximity filter: %d unique ICOS site(s) from %s; only sites with a Level 2 archive will be downloaded.",
            len(requested),
            proximity_path,
        )
    else:
        # An explicit site argument intentionally bypasses the proximity subset, but still must exist in the
        # authoritative Level 2 station listing (e.g. to test an archived station directly).
        requested = [arg.strip()]

    unavailable_sites = []
    failed_sites = []
    downloaded_sites = []

    for site_id in requested:
        logging.info("Processing ICOS site: %s", site_id)

        # Match the requested site ID case-insensitively.
        canonical_site_id = station_id_map.get(site_id.upper())

        if canonical_site_id is None:
            logging.warning(
                "ICOS site %s is not present in the available Level 2 station list. Skipping it.",
                site_id,
            )
            unavailable_sites.append(site_id)
            continue

        logging.info("Matched %s to ICOS Level 2 station %s", site_id, canonical_site_id)

        try:
            download_ICOS_L2(
                canonical_site_id,
                products=None,
                folder=icos_raw_data_folder,
                meta=meta_client,
                data=data_client,
            )
            downloaded_sites.append(site_id)
            logging.info("Finished ICOS site: %s", site_id)

        except Exception as e:
            logging.warning(
                "Could not download ICOS site %s. Continuing with the next site. Reason: %s",
                site_id,
                e,
            )
            failed_sites.append((site_id, str(e)))

    logging.info("ICOS download summary")
    logging.info("Successfully processed: %d", len(downloaded_sites))
    logging.info("Unavailable Level 2 sites: %d", len(unavailable_sites))
    logging.info("Failed downloads: %d", len(failed_sites))

    if unavailable_sites:
        logging.info("Unavailable sites: %s", ", ".join(unavailable_sites))

    if failed_sites:
        logging.info("Sites with download errors:")
        for site_id, error in failed_sites:
            logging.info("  %s: %s", site_id, error)

    return {
        "requested": requested,
        "downloaded": downloaded_sites,
        "unavailable": unavailable_sites,
        "failed": failed_sites,
    }


def finite_or_null(col: pl.Expr, strict: bool = False) -> pl.Expr:
    """Float64 value where it is a finite number; null / NaN / +-inf become null. Zero is a valid value."""
    x = col.cast(pl.Float64, strict=strict)
    return pl.when(x.is_finite()).then(x)


def drop_metadata_only_rows(df: pl.DataFrame, measurement_cols: list[str]) -> pl.DataFrame:
    """Keep rows with at least one valid number in `measurement_cols`. Rows are removed, columns never are."""
    return df.filter(
        pl.any_horizontal([finite_or_null(pl.col(c)).is_not_null() for c in measurement_cols])
    )


def norm_text(col: str) -> pl.Expr:
    """Trimmed, lower-case text for case-insensitive comparisons."""
    return pl.col(col).str.strip_chars().str.to_lowercase()


def clean_text(col: str) -> pl.Expr:
    """Trimmed text, blank -> null. Used for species names; never inferred or filled."""
    s = pl.col(col).str.strip_chars()
    return pl.when(s.str.len_chars() > 0).then(s).alias(col)


def ensure_cols(df: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    """Add all-null text columns for descriptors that a site's ANCILLARY file does not contain."""
    return df.with_columns(
        [pl.lit(None, dtype=pl.Utf8).alias(n) for n in names if n not in df.columns]
    )


def with_site_and_period(df: pl.DataFrame, site_id: str) -> pl.DataFrame:
    """Add site_id, year and month (from observation_date)."""
    return df.with_columns(
        pl.lit(site_id).alias("site_id"),
        pl.col("observation_date").dt.year().alias("year"),
        pl.col("observation_date").dt.month().alias("month"),
    )


def empty_observed(value: str) -> pl.DataFrame:
    """Typed empty frame for an observed variable with the common key columns."""
    schema = {
        "site_id": pl.Utf8,
        "observation_date": pl.Date,
        "year": pl.Int32,
        "month": pl.Int32,
        value: pl.Float64,
    }
    return pl.DataFrame(schema=schema)


def empty_gpp() -> pl.DataFrame:
    """Typed empty observed_gpp frame, retaining all three GPP measurement columns."""
    return pl.DataFrame(
        schema={
            "site_id": pl.Utf8,
            "observation_date": pl.Date,
            "year": pl.Int32,
            "month": pl.Int32,
            **dict.fromkeys([*GPP_OUTPUT_COLUMNS, "GPP_total"], pl.Float64),
        }
    )


def read_csv_header(path: Path) -> list[str]:
    """Column names of a CSV without any type inference."""
    return pl.read_csv(path, n_rows=1, infer_schema_length=0).columns


def read_csv_columns(
    path: Path, columns: list[str], text_columns: list[str] | tuple[str, ...] = ()
) -> pl.DataFrame:
    """Read selected columns without relying on CSV type inference.

    All selected columns are initially read as strings so an early integer-looking value cannot make a measurement
    column integer-typed when later rows contain valid decimals (for example, TA values 3 and 3.98). Explicit text
    columns remain strings; other columns are safely cast to Float64. Unparseable, blank, sentinel, NaN and infinite
    measurement values become null without discarding otherwise valid rows.
    """
    header = read_csv_header(path)
    missing = [c for c in columns if c not in header]
    if missing:
        raise ValueError(f"{Path(path).name}: missing column(s) {missing}")

    # Force strings at CSV-read time. Polars' inference can otherwise lock a numeric column to Int64 based on the
    # sampled values and fail when a later row contains a decimal. Coercion is handled explicitly below.
    df = pl.read_csv(
        path,
        columns=columns,
        infer_schema_length=0,
        schema_overrides=dict.fromkeys(columns, pl.Utf8),
    )

    text_set = set(text_columns)

    def clean(c: str) -> pl.Expr:
        x = pl.col(c).str.strip_chars().cast(pl.Float64, strict=False)
        return pl.when(x.is_finite() & (x != SENTINEL)).then(x).otherwise(None).alias(c)

    return df.with_columns([clean(c) for c in columns if c not in text_set])


def read_timeseries(
    path: Path,
    value_columns: list[str],
    keep_timestamp: bool = False,
    timestamp_column: str = "TIMESTAMP_START",
) -> pl.DataFrame:
    """30-min ICOS series: `date` (day of TIMESTAMP_START) plus the value columns as Float64 (-9999 -> null).
    keep_timestamp=True also returns the parsed TIMESTAMP_START as `timestamp` (first column), needed by the
    missing-data threshold to count expected records.
    """
    df = read_csv_columns(
        path, [timestamp_column, *value_columns], text_columns=[timestamp_column]
    )
    parsed = df.with_columns(
        pl.col(timestamp_column)
        .str.strptime(pl.Datetime("us"), "%Y%m%d%H%M", strict=False)
        .alias("_ts")
    )
    bad = parsed["_ts"].null_count()
    if bad:
        raise ValueError(f"{Path(path).name}: {bad} unparsable {timestamp_column} values")
    leading = [pl.col("_ts").alias("timestamp")] if keep_timestamp else []
    return parsed.select(*leading, pl.col("_ts").dt.date().alias("date"), *value_columns)


def resolve_max_missing_pct(max_missing_pct: float | None = None) -> float:
    """Threshold in use: the argument, else the module-wide MAX_MISSING_PCT (read at call time). Must be in [0, 100]."""
    pct = MAX_MISSING_PCT if max_missing_pct is None else max_missing_pct
    if (
        isinstance(pct, bool)
        or not isinstance(pct, int | float)
        or not math.isfinite(pct)
        or not 0 <= pct <= 100
    ):
        raise ValueError(f"max_missing_pct must be a number between 0 and 100, got {pct!r}")
    return float(pct)


def exceeds_missing_threshold(
    n_missing: pl.Expr, n_expected: pl.Expr, max_missing_pct: float
) -> pl.Expr:
    """True where 100 * n_missing / n_expected is STRICTLY greater than max_missing_pct. Cross-multiplied so a month
    sitting exactly on the threshold is never lost to floating-point division.
    """
    return n_missing.cast(pl.Float64) * 100.0 > n_expected.cast(pl.Float64) * max_missing_pct


def infer_step_minutes(timestamps: pl.Series) -> int:
    """Sampling interval of a regular series = the most frequent gap between distinct timestamps (minutes), so a
    file with gaps still reports its nominal cadence. Must divide a day evenly (30 and 60 min are the ICOS cases).
    """
    ts = timestamps.drop_nulls().unique().sort()
    if ts.len() < 2:
        raise ValueError("cannot infer the sampling interval from fewer than two timestamps")
    step = int(ts.diff().drop_nulls().dt.total_minutes().mode().min())
    if step <= 0 or 1440 % step:
        raise ValueError(
            f"unsupported sampling interval of {step} min (it must divide a day evenly)"
        )
    return step


def monthly_completeness(
    ts: pl.DataFrame, value_columns: list[str], max_missing_pct: float | None = None
) -> pl.DataFrame:
    """Per measurement column and calendar month of a regular series (needs a `timestamp` column; columns already
    cleaned: -9999 / NaN / inf are null): n_expected, n_valid, n_missing, missing_pct, keep.
    n_expected = days in the month x records per day (month length and leap years from the calendar; a month is
    counted by the day of TIMESTAMP_START, as in the climate aggregation). Records absent from the file are missing;
    null / invalid values are missing; 0 is valid. n_valid counts DISTINCT valid timestamps (duplicates and
    off-grid stamps cannot inflate it beyond n_expected). keep = NOT (missing_pct > threshold).
    """
    limit = resolve_max_missing_pct(max_missing_pct)
    per_day = 1440 // infer_step_minutes(ts["timestamp"])
    t = ts.with_columns(
        pl.col("timestamp").dt.year().alias("year"), pl.col("timestamp").dt.month().alias("month")
    )
    months = (
        t.select("year", "month")
        .unique()
        .with_columns(
            (pl.date("year", "month", 1).dt.month_end().dt.day().cast(pl.Int64) * per_day).alias(
                "n_expected"
            )
        )
    )
    frames = []
    for c in value_columns:
        valid = (
            t.filter(finite_or_null(pl.col(c)).is_not_null())
            .group_by("year", "month")
            .agg(pl.col("timestamp").n_unique().cast(pl.Int64).alias("n_valid"))
        )
        frames.append(
            months.join(valid, on=["year", "month"], how="left").with_columns(
                pl.lit(c).alias("column"), pl.col("n_valid").fill_null(0)
            )
        )
    return (
        pl.concat(frames)
        .with_columns(pl.min_horizontal("n_valid", "n_expected").alias("n_valid"))
        .with_columns((pl.col("n_expected") - pl.col("n_valid")).alias("n_missing"))
        .with_columns(
            (100.0 * pl.col("n_missing") / pl.col("n_expected")).alias("missing_pct"),
            (~exceeds_missing_threshold(pl.col("n_missing"), pl.col("n_expected"), limit)).alias(
                "keep"
            ),
        )
        .select(
            "column", "year", "month", "n_expected", "n_valid", "n_missing", "missing_pct", "keep"
        )
        .sort("column", "year", "month")
    )


def mask_incomplete_months(
    ts: pl.DataFrame, value_columns: list[str], max_missing_pct: float | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Set every value of a measurement to null in the months where THAT measurement fails the threshold, BEFORE any
    aggregation. Columns are decided independently; rows, other columns and row order are untouched. Returns the
    masked frame and the monthly_completeness report.
    """
    report = monthly_completeness(ts, value_columns, max_missing_pct)
    out = ts.with_row_index("_row").with_columns(
        pl.col("timestamp").dt.year().alias("year"), pl.col("timestamp").dt.month().alias("month")
    )
    for c in value_columns:
        rejected = (
            report.filter((pl.col("column") == c) & ~pl.col("keep"))
            .select("year", "month")
            .with_columns(pl.lit(True).alias("_rejected"))
        )
        out = (
            out.join(rejected, on=["year", "month"], how="left")
            .with_columns(pl.when(pl.col("_rejected").is_null()).then(pl.col(c)).alias(c))
            .drop("_rejected")
        )
    return out.sort("_row").drop("_row", "year", "month"), report


def warn_discarded_months(
    report: pl.DataFrame, site_id: str, max_missing_pct: float | None = None
) -> None:
    """One warning per measurement that lost months to the threshold (reported, never silent)."""
    limit = resolve_max_missing_pct(max_missing_pct)
    lost = (
        report.filter(~pl.col("keep"))
        .group_by("column", maintain_order=True)
        .agg(pl.len().alias("n"), pl.col("missing_pct").max().alias("worst"))
    )
    for row in lost.iter_rows(named=True):
        warnings.warn(
            f"{site_id}: {row['n']} month(s) of {row['column']} discarded (missing > {limit:g}%, worst "
            f"{row['worst']:.1f}%)",
            stacklevel=2,
        )


def validate_site_metadata(meta: dict, site_id: str) -> dict[str, float]:
    """Required site_3pg metadata: finite latitude (|lat| <= 90) and altitude. Site metadata is a property of the
    station, not a monthly measurement, so the missing-data threshold does not apply; invalid metadata raises.
    """
    out = {}
    for key in ("latitude", "altitude"):
        value = meta.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"{site_id}: site metadata {key!r} is missing or not a finite number: {value!r}"
            )
        out[key] = float(value)
    if not -90.0 <= out["latitude"] <= 90.0:
        raise ValueError(f"{site_id}: latitude {out['latitude']} is outside [-90, 90]")
    return out


def parse_badm_dates(df: pl.DataFrame, date_col: str | None) -> pl.DataFrame:
    """Add year, month, observation_date from a BADM date string (YYYY, YYYYMM, YYYYMMDD or YYYYMMDDHHMM).
    Only the precision that exists is used: observation_date needs a full day; nothing is completed or invented.
    """
    if date_col is None:
        return df.with_columns(
            pl.lit(None, dtype=pl.Int32).alias("year"),
            pl.lit(None, dtype=pl.Int32).alias("month"),
            pl.lit(None, dtype=pl.Date).alias("observation_date"),
        )
    d = pl.col(date_col).str.replace(r"\.0$", "").str.strip_chars()
    n = d.str.len_chars()
    return df.with_columns(
        pl.when(n >= 4).then(d.str.slice(0, 4).cast(pl.Int32, strict=False)).alias("year"),
        pl.when(n >= 6).then(d.str.slice(4, 2).cast(pl.Int32, strict=False)).alias("month"),
        pl.when(n >= 8)
        .then(d.str.slice(0, 8).str.strptime(pl.Date, "%Y%m%d", strict=False))
        .alias("observation_date"),
    )


def discover_sites(
    data_dir: Path, site_ids: Iterable[str] | None = None
) -> dict[str, dict[str, Path]]:
    """{site_id: {product: path}} for sites with exactly one METEO, FLUXES and ANCILLARY file (FLUXNET_HH optional).
    Incomplete or duplicated sites are skipped with a warning. `site_ids` (case-insensitive) restricts discovery to
    those sites; None keeps every site found in the folder.
    """
    wanted = None if site_ids is None else {s.upper() for s in site_ids}
    found: dict[str, list[tuple[str, str, Path]]] = {}
    for path in sorted(Path(data_dir).glob("ICOSETC_*.csv")):
        m = FILE_RE.match(path.name)
        if m:
            site_id = f"{m['cc'].upper()}-{m['rest']}"
            if wanted is not None and site_id.upper() not in wanted:
                continue
            found.setdefault(site_id.upper(), []).append((site_id, m["product"].upper(), path))
    if not found:
        scope = "" if wanted is None else f" for site(s) {sorted(wanted)}"
        raise FileNotFoundError(
            f"No ICOSETC_<SITE>_<PRODUCT>_*_L2*.csv files{scope} in {Path(data_dir).resolve()}"
        )

    sites = {}
    for records in found.values():
        site_id = min(r[0] for r in records)  # canonical spelling if filenames differ only by case
        by_product: dict[str, list[Path]] = {}
        for _, product, path in records:
            by_product.setdefault(product, []).append(path)
        duplicated = [p for p, paths in by_product.items() if len(paths) > 1]
        missing = [p for p in REQUIRED_PRODUCTS if p not in by_product]
        if duplicated or missing:
            warnings.warn(
                f"{site_id} skipped: duplicate files {duplicated}, missing files {missing}",
                stacklevel=2,
            )
            continue
        sites[site_id] = {product: paths[0] for product, paths in by_product.items()}
    return dict(sorted(sites.items()))


def fetch_icos_station_metadata(site_id: str) -> dict:
    """Latitude / elevation from the ICOS Carbon Portal metadata service (needs network). Raises, never guesses."""
    query = f"""
    prefix cpmeta: <http://meta.icos-cp.eu/ontologies/cpmeta/>
    select ?lat ?elev where {{
      ?station cpmeta:hasStationId "{site_id}" .
      optional {{ ?station cpmeta:hasLatitude ?lat }}
      optional {{ ?station cpmeta:hasElevation ?elev }}
    }}
    """
    req = urllib.request.Request(
        ICOS_SPARQL_ENDPOINT,
        data=urllib.parse.urlencode({"query": query}).encode(),
        headers={"Accept": "application/sparql-results+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as exc:  # network, HTTP, JSON ...
        raise RuntimeError(
            f"ICOS metadata lookup failed for {site_id}: {exc}. No coordinates were guessed."
        ) from exc
    for row in payload.get("results", {}).get("bindings", []):
        if "lat" in row and "elev" in row:
            return {
                "latitude": float(row["lat"]["value"]),
                "altitude": float(row["elev"]["value"]),
            }
    raise LookupError(f"No ICOS metadata with latitude AND elevation for station {site_id}.")


def collect_station_metadata(site_ids: Iterable[str]) -> dict[str, dict]:
    """{site_id: {"latitude", "altitude"}} from the ICOS metadata service (network). A site whose lookup fails is left
    out with a warning (build_outputs then skips it); coordinates are never guessed.
    """
    station_meta = {}
    for site_id in site_ids:
        try:
            station_meta[site_id] = fetch_icos_station_metadata(site_id)
        except (RuntimeError, LookupError) as exc:
            warnings.warn(f"{site_id} skipped: {exc}", stacklevel=2)
    return station_meta


def read_ancillary(path: Path, site_id: str) -> pl.DataFrame:
    """Long-format BADM table with DATAVALUE trimmed; blank and -9999 values become null."""
    anc = pl.read_csv(path, infer_schema_length=0)
    needed = {"SITE_ID", "GROUP_ID", "VARIABLE_GROUP", "VARIABLE", "DATAVALUE"}
    if not needed <= set(anc.columns):
        raise ValueError(f"ANCILLARY columns missing: {sorted(needed - set(anc.columns))}")
    file_sites = {s.strip().upper() for s in anc["SITE_ID"].drop_nulls().unique()}
    if file_sites != {site_id.upper()}:
        raise ValueError(
            f"ANCILLARY SITE_ID {sorted(file_sites)} does not match filename site {site_id}"
        )
    value = pl.col("DATAVALUE").str.strip_chars()
    return anc.with_columns(
        pl.col("GROUP_ID").cast(pl.Int64, strict=False),
        pl.when((value.str.len_chars() > 0) & (value != str(SENTINEL)))
        .then(value)
        .alias("DATAVALUE"),
    )


def read_ancillary_group(anc: pl.DataFrame, group: str) -> tuple[pl.DataFrame, list[str]] | None:
    """One BADM variable group as a wide table (one row per GROUP_ID) with numeric variables as finite Float64 and
    year / month / observation_date parsed from its *_DATE column. Also returns the descriptor columns.
    None if the site has no such group.
    """
    g = anc.filter(pl.col("VARIABLE_GROUP") == group)
    if g.is_empty():
        return None
    wide = g.pivot(on="VARIABLE", index="GROUP_ID", values="DATAVALUE", aggregate_function="first")
    date_col = next((c for c in wide.columns if re.search(r"_DATE$", c)), None)
    numeric = [c for c in wide.columns if c in NUMERIC_VARS]
    descriptors = [c for c in wide.columns if c not in ("GROUP_ID", date_col, *numeric)]
    wide = wide.with_columns([finite_or_null(pl.col(c)).alias(c) for c in numeric])
    return parse_badm_dates(wide, date_col), descriptors


def monthly_climate(meteo: pl.DataFrame, fluxes: pl.DataFrame, site_id: str) -> pl.DataFrame:
    """climate_3pg rows of one site (unfiltered). Every 30-min value belongs to the calendar day of TIMESTAMP_START.
    tmp_min/tmp_max/tmp_ave : monthly mean of the daily min / max / mean of TA                  [degC]
    prcp                    : daily sums of P summed over the month (gaps are skipped, not filled) [mm month-1]
    srad                    : monthly mean of SW_IN [W m-2] * 0.0864                               [MJ m-2 d-1]
    frost_days              : number of days with daily-min TA < 0 degC                            [days month-1]
    vpd_day                 : monthly mean of daily mean VPD (hPa == mbar, numerically unchanged)  [mbar]
    co2                     : monthly mean of daily mean CO2_DRY (umol mol-1 == ppm)               [ppm]
    d13catm                 : not available in the ICOS data -> null.
    """
    daily = (
        meteo.group_by("date")
        .agg(
            pl.col("TA").min().alias("ta_min"),
            pl.col("TA").max().alias("ta_max"),
            pl.col("TA").mean().alias("ta_mean"),
            pl.col("VPD").mean().alias("vpd_mean"),
            pl.col("SW_IN").mean().alias("sw_in_mean"),
            # sum() of an all-null day is 0 in Polars -> keep such days null
            pl.when(pl.col("P").count() > 0).then(pl.col("P").sum()).alias("p_day"),
        )
        .with_columns(
            pl.col("date").dt.year().alias("year"), pl.col("date").dt.month().alias("month")
        )
    )
    monthly = daily.group_by("year", "month").agg(
        pl.col("ta_min").mean().alias("tmp_min"),
        pl.col("ta_max").mean().alias("tmp_max"),
        pl.col("ta_mean").mean().alias("tmp_ave"),
        pl.when(pl.col("p_day").count() > 0).then(pl.col("p_day").sum()).alias("prcp"),
        (pl.col("sw_in_mean").mean() * SRAD_FACTOR).alias("srad"),
        pl.when(pl.col("ta_min").count() > 0)
        .then((pl.col("ta_min") < 0).sum().cast(pl.Int32))
        .alias("frost_days"),
        pl.col("vpd_mean").mean().alias("vpd_day"),
    )
    co2 = (
        fluxes.group_by("date")
        .agg(pl.col("CO2_DRY").mean().alias("co2_day"))
        .with_columns(
            pl.col("date").dt.year().alias("year"), pl.col("date").dt.month().alias("month")
        )
        .group_by("year", "month")
        .agg(pl.col("co2_day").mean().alias("co2"))
    )
    return (
        monthly.join(co2, on=["year", "month"], how="full", coalesce=True)
        .with_columns(
            pl.lit(site_id).alias("site_id"), pl.lit(None, dtype=pl.Float64).alias("d13catm")
        )
        .select(CLIMATE_COLS)
        .sort("year", "month")
    )


def extract_stems(records: pl.DataFrame) -> pl.DataFrame:
    """Species-level stems_n (trees ha-1) from GRP_TREES_NUM: (observation_date, species, stems_n).
    Only species-specific records are used, so a stand-level count is never assigned to a species. Filters: Tree,
    Mature or life stage not stated, Alive, Mean. Species-specific records carry no TREES_NUM_ORGAN in the ICOS
    source (only the stand total states "Total AG"), so the organ must be "Total AG" OR NOT STATED.
    More than one record for the same date + species is ambiguous -> stems_n null, never averaged.
    """
    descriptors = [
        "TREES_NUM_SPP",
        "TREES_NUM_VEGTYPE",
        "TREES_NUM_LIFESTAGE",
        "TREES_NUM_VEG_STATUS",
        "TREES_NUM_ORGAN",
        "TREES_NUM_STATISTIC",
    ]
    r = ensure_cols(records, descriptors).with_columns(
        clean_text("TREES_NUM_SPP"), clean_text("TREES_NUM_ORGAN")
    )
    sel = r.filter(
        pl.col("TREES_NUM").is_not_null()
        & pl.col("observation_date").is_not_null()
        & pl.col("TREES_NUM_SPP").is_not_null()
        & (norm_text("TREES_NUM_VEGTYPE") == "tree")
        & (
            pl.col("TREES_NUM_LIFESTAGE").is_null()
            | (norm_text("TREES_NUM_LIFESTAGE") == "mature")
        )
        & (pl.col("TREES_NUM_ORGAN").is_null() | (norm_text("TREES_NUM_ORGAN") == "total ag"))
        & (norm_text("TREES_NUM_VEG_STATUS") == "alive")
        & (norm_text("TREES_NUM_STATISTIC") == "mean")
    )
    return (
        sel.group_by("observation_date", pl.col("TREES_NUM_SPP").alias("species"))
        .agg(pl.col("TREES_NUM").first().alias("_v"), pl.len().alias("_n"))
        .select(
            "observation_date",
            "species",
            pl.when(pl.col("_n") == 1).then(pl.col("_v")).alias("stems_n"),
        )
    )


def extract_biomass_components(records: pl.DataFrame) -> pl.DataFrame:
    """Explicit stem / root / foliage biomass (t DM ha-1) from GRP_BIOMASS: (observation_date, species, biom_*).
    Alive, Mean, kgDM m-2 only (x10 -> t DM ha-1). Total aboveground biomass is never split into pools.
    """
    r = ensure_cols(
        records,
        ["BIOMASS_ORGAN", "BIOMASS_UNIT", "BIOMASS_STATUS", "BIOMASS_STATISTIC", "BIOMASS_SPP"],
    ).with_columns(clean_text("BIOMASS_SPP"))
    sel = r.with_columns(
        norm_text("BIOMASS_ORGAN").replace_strict(ORGAN_MAP, default=None).alias("target")
    ).filter(
        pl.col("BIOMASS").is_not_null()
        & pl.col("observation_date").is_not_null()
        & pl.col("target").is_not_null()
        & (norm_text("BIOMASS_STATUS") == "alive")
        & (norm_text("BIOMASS_STATISTIC") == "mean")
        & (norm_text("BIOMASS_UNIT") == "kgdm m-2")
    )
    empty = pl.DataFrame(
        schema={
            "observation_date": pl.Date,
            "species": pl.Utf8,
            **dict.fromkeys(COMPONENT_COLS, pl.Float64),
        }
    )
    if sel.is_empty():
        return empty
    out = (
        sel.group_by("observation_date", pl.col("BIOMASS_SPP").alias("species"), "target")
        .agg(pl.col("BIOMASS").first().alias("_v"), pl.len().alias("_n"))
        .with_columns(
            pl.when(pl.col("_n") == 1).then(pl.col("_v") * KGDM_M2_TO_T_HA).alias("value")
        )
        .pivot(
            on="target",
            index=["observation_date", "species"],
            values="value",
            aggregate_function="first",
        )
    )
    out = out.with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(c) for c in COMPONENT_COLS if c not in out.columns]
    )
    return out.select("observation_date", "species", *COMPONENT_COLS)


def extract_total_biomass(records: pl.DataFrame, site_id: str) -> pl.DataFrame:
    """Species-specific total aboveground biomass from GRP_BIOMASS: one row per date + species + source unit.
    Never summed or averaged across species. Selection: dated, unit stated, organ = Total aboveground (or organ not
    stated for a species-specific Tree record), Alive, Mean. BIOMASS_SPP is kept verbatim (null = none stated).
    Conflicting values for one date + species + unit -> null.

    total_biomass : value exactly as observed, in the source unit (kgDM m-2 at CH-Dav; unit not stored in the table).
    biomass_ha    : t DM ha-1 = total_biomass * factor(unit), factor from DM_AREA_UNIT_TO_T_HA (kgDM m-2 -> x10).
                    The source value is already per unit area, so the table's plot_area_ha (a site-level area, not
                    the sampled area) is NOT used. Unknown / non-dry-matter units or invalid values -> null.
    """
    descriptors = [
        "BIOMASS_ORGAN",
        "BIOMASS_UNIT",
        "BIOMASS_STATUS",
        "BIOMASS_STATISTIC",
        "BIOMASS_SPP",
        "BIOMASS_VEGTYPE",
    ]
    r = ensure_cols(records, descriptors).with_columns(clean_text("BIOMASS_SPP"))
    eligible = r.filter(
        pl.col("BIOMASS").is_not_null()
        & pl.col("observation_date").is_not_null()
        & pl.col("BIOMASS_UNIT").is_not_null()
        & (
            norm_text("BIOMASS_ORGAN").is_in(TOTAL_ORGAN_LABELS)
            | (
                pl.col("BIOMASS_ORGAN").is_null()
                & pl.col("BIOMASS_SPP").is_not_null()
                & (norm_text("BIOMASS_VEGTYPE") == "tree")
            )
        )
        & (norm_text("BIOMASS_STATUS") == "alive")
        & (norm_text("BIOMASS_STATISTIC") == "mean")
    )
    factor = (
        pl.col("unit")
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
        .replace_strict(DM_AREA_UNIT_TO_T_HA, default=None, return_dtype=pl.Float64)
    )
    out = (
        eligible.group_by(
            "observation_date",
            pl.col("BIOMASS_SPP").alias("species"),
            pl.col("BIOMASS_UNIT").alias("unit"),
        )
        .agg(
            pl.col("BIOMASS").n_unique().alias("_n_values"), pl.col("BIOMASS").first().alias("_v")
        )
        .with_columns(pl.when(pl.col("_n_values") == 1).then(pl.col("_v")).alias("total_biomass"))
        .with_columns(finite_or_null(pl.col("total_biomass") * factor).alias("biomass_ha"))
        .sort("observation_date", "species", "unit", nulls_last=True)
    )
    return with_site_and_period(out, site_id).select(
        "site_id", "observation_date", "year", "month", "species", "total_biomass", "biomass_ha"
    )


def assemble_species(site_id: str, stems: pl.DataFrame, comps: pl.DataFrame) -> pl.DataFrame:
    """species_3pg: one row per observation_date + species present in either source. planted / fertility are not
    available in the ICOS data and stay null; nothing is forward-filled.
    """
    key = ["observation_date", "_k"]

    def keyed(df):
        return df.with_columns(
            pl.col("species").fill_null("").alias("_k")
        )  # lets null species join on null species

    keys = pl.concat([d.select("observation_date", "species") for d in (stems, comps)]).unique()
    out = (
        keyed(keys)
        .join(keyed(stems).drop("species"), on=key, how="left")
        .join(keyed(comps).drop("species"), on=key, how="left")
    )
    out = with_site_and_period(out, site_id).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("planted"),
        pl.lit(None, dtype=pl.Float64).alias("fertility"),
    )
    return out.select(SPECIES_COLS).sort("observation_date", "species", nulls_last=True)


def extract_gpp(
    path: Path | None, site_id: str, max_missing_pct: float | None = None
) -> pl.DataFrame:
    """Monthly GPP estimates from FLUXNET_HH (µmol CO2 m-2 s-1 rates; no unit conversion; -9999 -> null).

    ``GPP_DT_VUT_REF`` and ``GPP_NT_VUT_REF`` are independently thresholded against expected half-hourly stamps,
    then each is the monthly mean of valid daily means. ``GPP_total = GPP_DT + GPP_NT`` is deliberately calculated
    only after aggregation and remains null if either estimate is null. Despite its requested name, it is a sum of
    daytime- and nighttime-partitioning *estimates* of the same underlying GPP, not necessarily physical total GPP.
    TIMESTAMP_START is used, falling back to TIMESTAMP. Month = day of that timestamp.
    """
    if path is None:
        return empty_gpp()
    limit = resolve_max_missing_pct(max_missing_pct)
    try:
        header = read_csv_header(path)
        time_column = "TIMESTAMP_START" if "TIMESTAMP_START" in header else "TIMESTAMP"
        source_columns = [c for c in GPP_SOURCE_COLUMNS if c in header]
        missing = [c for c in GPP_SOURCE_COLUMNS if c not in header]
        if missing:
            warnings.warn(
                f"{site_id}: FLUXNET_HH missing GPP source column(s) {missing}", stacklevel=2
            )
        if not source_columns:
            warnings.warn(
                f"{site_id}: GPP left empty (no supported GPP source columns)", stacklevel=2
            )
            return empty_gpp()
        ts = read_timeseries(
            path, source_columns, keep_timestamp=True, timestamp_column=time_column
        )
        masked, report = mask_incomplete_months(ts, source_columns, limit)
    except ValueError as exc:
        warnings.warn(f"{site_id}: GPP left empty ({exc})", stacklevel=2)
        return empty_gpp()
    warn_discarded_months(report, site_id, limit)
    # count() > 0 guards prevent Polars 1.x producing NaN for all-null groups.
    daily = masked.group_by("date").agg(
        [
            pl.when(pl.col(c).count() > 0).then(pl.col(c).mean()).alias(GPP_SOURCE_COLUMNS[c])
            for c in source_columns
        ]
    )
    monthly = (
        daily.with_columns(
            pl.col("date").dt.year().cast(pl.Int32).alias("year"),
            pl.col("date").dt.month().cast(pl.Int32).alias("month"),
        )
        .group_by("year", "month")
        .agg(
            [
                pl.when(pl.col(c).count() > 0).then(pl.col(c).mean()).alias(c)
                for c in GPP_OUTPUT_COLUMNS
                if c in daily
            ]
        )
    )
    monthly = monthly.with_columns(
        [
            pl.lit(None, dtype=pl.Float64).alias(c)
            for c in GPP_OUTPUT_COLUMNS
            if c not in monthly.columns
        ]
    ).with_columns(
        # This is the requested arithmetic sum of two alternative partitioning estimates, not physical total GPP.
        (pl.col("GPP_DT") + pl.col("GPP_NT")).alias("GPP_total")
    )
    return monthly.select(
        pl.lit(site_id).alias("site_id"),
        pl.date("year", "month", 1).alias("observation_date"),
        "year",
        "month",
        *GPP_OUTPUT_COLUMNS,
        "GPP_total",
    ).sort("observation_date")


def extract_lai(group: tuple[pl.DataFrame, list[str]] | None, site_id: str) -> pl.DataFrame:
    """Observed LAI (m2 m-2, as supplied) from GRP_LAI, placed by the record's own year-month (observation_date is
    null when the source gives no day). Only LAI_STATISTIC = Mean records are used. Several definitions (different
    descriptors) or conflicting values in one date-month are ambiguous -> null. No averaging, interpolation or fill.
    """
    if group is None or "LAI" not in group[0].columns:
        return empty_observed("LAI")
    records, descriptors = group
    stat_cols = [c for c in descriptors if c.endswith("_STATISTIC")]
    r = records.filter(
        pl.col("LAI").is_not_null()
        & pl.col("year").is_not_null()
        & pl.col("month").is_between(1, 12)
        & pl.all_horizontal([pl.lit(True), *[norm_text(c) == "mean" for c in stat_cols]])
    )
    n_definitions = pl.struct(descriptors).n_unique() if descriptors else pl.lit(1)
    return (
        r.group_by("observation_date", "year", "month")
        .agg(
            n_definitions.alias("_n_defs"),
            pl.col("LAI").n_unique().alias("_n_vals"),
            pl.col("LAI").first().alias("_v"),
        )
        .select(
            pl.lit(site_id).alias("site_id"),
            "observation_date",
            "year",
            "month",
            pl.when((pl.col("_n_defs") == 1) & (pl.col("_n_vals") == 1))
            .then(pl.col("_v"))
            .alias("LAI"),
        )
        .sort("observation_date", "year", "month", nulls_last=True)
    )


def extract_dbh_aggregate(
    group: tuple[pl.DataFrame, list[str]] | None, site_id: str
) -> pl.DataFrame:
    """Stand-aggregate DBH (cm) from GRP_DBH: explicit Mean, Total AG, Alive, no species, approach "all species".
    Species-level records are not turned into a stand value; conflicting values on one date -> null.
    """
    if group is None or "DBH" not in group[0].columns:
        return empty_observed("DBH")
    r = ensure_cols(
        group[0], ["DBH_SPP", "DBH_ORGAN", "DBH_VEG_STATUS", "DBH_STATISTIC", "DBH_APPROACH"]
    )
    agg = r.filter(
        pl.col("DBH").is_not_null()
        & pl.col("observation_date").is_not_null()
        & pl.col("DBH_SPP").is_null()
        & (norm_text("DBH_STATISTIC") == "mean")
        & (norm_text("DBH_ORGAN") == "total ag")
        & (norm_text("DBH_VEG_STATUS") == "alive")
        & norm_text("DBH_APPROACH").str.contains("all species")
    )
    return (
        agg.group_by("observation_date", "year", "month")
        .agg(pl.col("DBH").n_unique().alias("_n"), pl.col("DBH").first().alias("_v"))
        .select(
            pl.lit(site_id).alias("site_id"),
            "observation_date",
            "year",
            "month",
            pl.when(pl.col("_n") == 1).then(pl.col("_v")).alias("DBH"),
        )
        .sort("observation_date")
    )


def load_lwf_dbh(path: Path) -> pl.DataFrame:
    """Individual-tree DBH (cm) of the LWF stand `sitename == "Davos"` (CH-Dav), one row per tree observation:
    area_site = parcelname, site = sitename, date = date_observation, tree_id = banr (source identifier),
    species = tree_species, dbh = dbh_corrected where available else raw dbh, dbh_source says which.
    Raises if the file, its columns or the Davos rows are unavailable: no fallback to ICOS aggregates.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: CH-Dav DBH needs the LWF individual-tree file."
        )
    with open(path, "rb") as fh:
        head = fh.readline().decode("utf-8-sig", errors="replace")
    raw = pl.read_csv(
        path,
        separator=max([",", ";", "\t"], key=head.count),
        encoding="utf8-lossy",
        infer_schema_length=0,
    )
    raw = raw.rename({c: c.strip().lstrip("\ufeff") for c in raw.columns})
    missing = [c for c in LWF_REQUIRED_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} lacks documented columns {missing}; found {raw.columns}")
    trees = raw.filter(pl.col("sitename") == LWF_SITENAME)  # exact LWF name, not the ICOS id
    if trees.is_empty():
        raise ValueError(f"{path.name}: no rows with sitename == {LWF_SITENAME!r}")

    def text_or_null(c):  # textual missing markers -> null before numeric / date casting
        s = pl.col(c).str.strip_chars()
        return pl.when(s.is_in(LWF_MISSING_MARKERS)).then(None).otherwise(s).alias(c)

    trees = trees.with_columns([text_or_null(c) for c in [*LWF_NUMERIC_COLS, "date_observation"]])
    corrected = finite_or_null(pl.col("dbh_corrected"), strict=True)
    measured = finite_or_null(pl.col("dbh"), strict=True)
    return trees.select(
        pl.col("parcelname").alias("area_site"),
        pl.col("sitename").alias("site"),
        pl.col("date_observation").str.slice(0, 10).str.to_date("%Y-%m-%d").alias("date"),
        pl.col("banr").cast(pl.Int64).alias("tree_id"),
        pl.col("tree_species").alias("species"),
        pl.coalesce(corrected, measured).alias("dbh"),
        pl.when(corrected.is_not_null())
        .then(pl.lit("dbh_corrected"))
        .when(measured.is_not_null())
        .then(pl.lit("dbh"))
        .alias("dbh_source"),
    )


def process_site(
    site_id: str, paths: dict[str, Path], meta: dict, max_missing_pct: float | None = None
) -> dict[str, pl.DataFrame]:
    """All extractions for ONE site (unfiltered). Nothing here can see another site's data.
    max_missing_pct : missing-data threshold (None = MAX_MISSING_PCT); applied to the 30-min climate and GPP inputs
                      before aggregation, see the module docstring for the tables it cannot apply to.
    """
    limit = resolve_max_missing_pct(max_missing_pct)
    site_meta = validate_site_metadata(meta, site_id)

    meteo_cols = ["TA", "P", "SW_IN", "VPD"]
    meteo, meteo_report = mask_incomplete_months(
        read_timeseries(paths["METEO"], meteo_cols, keep_timestamp=True), meteo_cols, limit
    )
    # dry-air CO2 only; wet / sensor channels not substituted
    fluxes, fluxes_report = mask_incomplete_months(
        read_timeseries(paths["FLUXES"], ["CO2_DRY"], keep_timestamp=True), ["CO2_DRY"], limit
    )
    warn_discarded_months(pl.concat([meteo_report, fluxes_report]), site_id, limit)
    climate = monthly_climate(meteo, fluxes, site_id)

    # site_3pg from/to = first / last year-month with at least one valid derived climate value (after thresholding)
    valid = drop_metadata_only_rows(climate, MEASUREMENT_COLS["climate_3pg"]).sort("year", "month")
    if valid.is_empty():
        raise ValueError(f"{site_id}: no valid derived climate values; cannot define from/to")
    first, last = valid.row(0, named=True), valid.row(valid.height - 1, named=True)
    site = pl.DataFrame(
        {
            "site_id": [site_id],
            "latitude": [site_meta["latitude"]],
            "altitude": [site_meta["altitude"]],
            "from": [f"{first['year']:04d}-{first['month']:02d}"],
            "to": [f"{last['year']:04d}-{last['month']:02d}"],
        },
        schema={
            "site_id": pl.Utf8,
            "latitude": pl.Float64,
            "altitude": pl.Float64,
            "from": pl.Utf8,
            "to": pl.Utf8,
        },
    )

    anc = read_ancillary(paths["ANCILLARY"], site_id)
    trees_num, biomass = (
        read_ancillary_group(anc, "GRP_TREES_NUM"),
        read_ancillary_group(anc, "GRP_BIOMASS"),
    )
    species_schema = {"observation_date": pl.Date, "species": pl.Utf8}
    stems = (
        extract_stems(trees_num[0])
        if trees_num
        else pl.DataFrame(schema={**species_schema, "stems_n": pl.Float64})
    )
    comps = (
        extract_biomass_components(biomass[0])
        if biomass
        else pl.DataFrame(schema={**species_schema, **dict.fromkeys(COMPONENT_COLS, pl.Float64)})
    )
    total_biomass = (
        extract_total_biomass(biomass[0], site_id)
        if biomass
        else pl.DataFrame(
            schema={
                "site_id": pl.Utf8,
                "observation_date": pl.Date,
                "year": pl.Int32,
                "month": pl.Int32,
                "species": pl.Utf8,
                "total_biomass": pl.Float64,
                "biomass_ha": pl.Float64,
            }
        )
    )
    return {
        "climate_3pg": climate,
        "site_3pg": site,
        "species_3pg": assemble_species(site_id, stems, comps),
        "observed_gpp": extract_gpp(paths.get("FLUXNET_HH"), site_id, limit),
        "observed_lai": extract_lai(read_ancillary_group(anc, "GRP_LAI"), site_id),
        "observed_total_biomass": total_biomass,
        "observed_dbh": extract_dbh_aggregate(read_ancillary_group(anc, "GRP_DBH"), site_id),
    }


def build_outputs(
    sites: dict[str, dict[str, Path]],
    station_meta: dict[str, dict],
    lwf_path: Path | None = None,
    max_missing_pct: float | None = None,
) -> dict[str, pl.DataFrame]:
    """Run every site independently and return exactly the seven final tables (metadata-only rows removed, no column
    dropped). Skipped sites and empty tables are reported with warnings.

    sites        : {site_id: {product: path}} from discover_sites
    station_meta : {site_id: {"latitude": float, "altitude": float}}
    lwf_path     : LWF individual-tree file; required when CH-Dav is processed.
    max_missing_pct : missing-data threshold for every site (None = MAX_MISSING_PCT).
    """
    per_site = {}
    skipped_reasons: list[str] = []
    for site_id, paths in sites.items():
        if site_id not in station_meta:
            reason = f"{site_id}: no ICOS station metadata returned (check the metadata lookup warnings above)"
            skipped_reasons.append(reason)
            warnings.warn(f"{site_id} skipped: no ICOS station metadata", stacklevel=2)
            continue
        try:
            per_site[site_id] = process_site(
                site_id, paths, station_meta[site_id], max_missing_pct
            )
        except (ValueError, KeyError, OSError, pl.exceptions.PolarsError) as exc:
            reason = f"{site_id}: {type(exc).__name__}: {exc}"
            skipped_reasons.append(reason)
            warnings.warn(f"{site_id} skipped: {type(exc).__name__}: {exc}", stacklevel=2)
    if not per_site:
        details = (
            "\n  - ".join(skipped_reasons)
            if skipped_reasons
            else "No candidate sites were discovered."
        )
        raise RuntimeError(f"No site could be processed. Site-level reasons:\n  - {details}")

    def stack(name: str) -> pl.DataFrame:
        return pl.concat([r[name] for r in per_site.values()], how="diagonal_relaxed")

    def with_plot_area(df: pl.DataFrame, site_col: str) -> pl.DataFrame:
        area = pl.col(site_col).replace_strict(
            PLOT_AREA_BY_SITE_LABEL, default=None, return_dtype=pl.Float64
        )
        return df.select(area.alias("plot_area_ha"), pl.all())

    # observed_dbh: CH-Dav = LWF individual trees (no fallback to ICOS aggregates); other sites keep their ICOS
    # aggregated DBH in the same schema with area_site, tree_id, species and dbh_source null.
    davos = [s for s in per_site if s.upper() == LWF_ICOS_SITE_ID.upper()]
    icos_dbh = (
        stack("observed_dbh")
        .filter(pl.col("site_id").str.to_uppercase() != LWF_ICOS_SITE_ID.upper())
        .select(
            pl.lit(None, dtype=pl.Utf8).alias("area_site"),
            pl.col("site_id").alias("site"),
            pl.col("observation_date").alias("date"),
            pl.lit(None, dtype=pl.Int64).alias("tree_id"),
            pl.lit(None, dtype=pl.Utf8).alias("species"),
            pl.col("DBH").alias("dbh"),
            pl.lit(None, dtype=pl.Utf8).alias("dbh_source"),
        )
    )
    if davos:
        if lwf_path is None:
            raise ValueError(f"{LWF_ICOS_SITE_ID} was processed but no LWF file path was given")
        icos_dbh = pl.concat([icos_dbh, load_lwf_dbh(lwf_path)], how="vertical_relaxed")
    observed_dbh = icos_dbh.sort("site", "date", "area_site", "tree_id", nulls_last=True)

    tables = {
        "climate_3pg": stack("climate_3pg").select(CLIMATE_COLS),
        "site_3pg": stack("site_3pg").select(SITE_COLS),
        "species_3pg": stack("species_3pg").select(SPECIES_COLS),
        "observed_gpp": with_plot_area(stack("observed_gpp"), "site_id"),
        "observed_lai": with_plot_area(stack("observed_lai"), "site_id"),
        "observed_total_biomass": with_plot_area(stack("observed_total_biomass"), "site_id"),
        "observed_dbh": with_plot_area(observed_dbh, "site"),
    }
    outputs = {}
    for name, df in tables.items():
        if name in OBSERVED_COLS:
            df = df.select(OBSERVED_COLS[name])
        outputs[name] = drop_metadata_only_rows(df, MEASUREMENT_COLS[name])
        if outputs[name].is_empty():
            warnings.warn(f"{name} is empty: no row contains a valid measurement", stacklevel=2)
    return outputs
