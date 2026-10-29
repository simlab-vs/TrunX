"""ICOS ETC L2 -> 3-PG preprocessing: download, preprocess and save the seven 3-PG tables.

Usage:
    python icos_preprocessing.py             # all available sites  (same as download(None))
    python icos_preprocessing.py None        # all available sites
    python icos_preprocessing.py CH-Dav      # one site

Each site is processed on its own from its METEO / FLUXES / ANCILLARY files (plus the optional FLUXNET_HH file, the
only GPP source); the tables are concatenated afterwards. -9999, NaN and infinite values become null before any
computation. Rows without a valid measurement are removed, columns are never dropped; nothing is interpolated or
forward-filled. ICOS timestamps are local standard time; a 30-min value belongs to the day of TIMESTAMP_START.

The seven tables (exact names): climate_3pg, site_3pg, species_3pg, observed_gpp, observed_lai,
observed_total_biomass, observed_dbh. They are written as Parquet files to PROCESSED_DIR (all sites) or to
PROCESSED_DIR/<SITE>/ (a single site, so a one-site run never overwrites the all-sites tables).
"""

# ruff: noqa: E501

import argparse
import logging
import re
import warnings
from pathlib import Path

import polars as pl
from icos_helper import (
    DATA_DIR,
    LWF_PATH,
    MAX_MISSING_PCT,
    build_outputs,
    collect_station_metadata,
    discover_sites,
    download,
)

PROCESSED_DIR = Path("data/processed/ICOS")  # mirrors the raw-data layout data/raw/ICOS
OUTPUT_NAMES = (
    "climate_3pg",
    "site_3pg",
    "species_3pg",
    "observed_gpp",
    "observed_lai",
    "observed_total_biomass",
    "observed_dbh",
)


def parse_site_argument(value: str | None) -> str | None:
    """Command-line site -> site id, or None (all available sites) for no argument or the literal string 'None'."""
    if value is None or value.strip().lower() == "none":
        return None
    return value.strip()


def save_outputs(outputs: dict[str, pl.DataFrame], site: str | None) -> Path:
    """Write the seven tables as <name>.parquet; a single-site run gets its own sub-folder."""
    folder = (
        PROCESSED_DIR
        if site is None
        else PROCESSED_DIR / re.sub(r"[^A-Za-z0-9_-]", "_", site.upper())
    )
    folder.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        outputs[name].write_parquet(folder / f"{name}.parquet")
    return folder


def run(site: str | None = None) -> dict[str, pl.DataFrame]:
    """Download (download(site)), preprocess in dependency order and save. Returns the seven tables by name."""
    result = download(site)
    logging.info("Missing-data threshold: MAX_MISSING_PCT = %s%%", MAX_MISSING_PCT)

    # Warnings (skipped sites, empty tables) are collected and reported once at the end.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Source files of the requested sites (files already in DATA_DIR are used even if a download failed).
        sites = discover_sites(DATA_DIR, result["requested"])
        station_meta = collect_station_metadata(sites)
        try:
            outputs = build_outputs(sites, station_meta, LWF_PATH)
        except Exception:
            # Do not hide the warnings that explain why every candidate site was skipped.
            notes = sorted({str(w.message) for w in caught})
            for note in notes:
                logging.error("Preprocessing diagnostic: %s", note)
            raise
    notes = sorted({str(w.message) for w in caught})
    logging.info("Notes: %s", "; ".join(notes) if notes else "none")

    folder = save_outputs(outputs, site)
    for name in OUTPUT_NAMES:
        logging.info("%s: %d rows x %d columns", name, outputs[name].height, outputs[name].width)
    logging.info("Saved the seven tables to %s", folder)
    return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Download and preprocess ICOS ETC L2 data for 3-PG."
    )
    parser.add_argument(
        "site",
        nargs="?",
        default=None,
        help="ICOS site id such as CH-Dav; omit or pass 'None' to process all available sites.",
    )
    args = parser.parse_args()

    outputs = run(parse_site_argument(args.site))

    climate_3pg = outputs["climate_3pg"]
    site_3pg = outputs["site_3pg"]
    species_3pg = outputs["species_3pg"]
    observed_gpp = outputs["observed_gpp"]
    observed_lai = outputs["observed_lai"]
    observed_total_biomass = outputs["observed_total_biomass"]
    observed_dbh = outputs["observed_dbh"]
