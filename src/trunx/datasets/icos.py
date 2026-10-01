"""ICOS dataset related utilities.

All Level 2 CSVs of an ICOS ecosystem station (FLUXES, METEO, FLUXNET_DD,
VARINFO_*, ...) are packed into a single data object of type "ETC L2 ARCHIVE"
(e.g. ``ICOSETC_CH-Dav_ARCHIVE_INTERIM_L2.zip``). Level 2 files are thus fetched
by downloading that archive and extracting the wanted members. The available
products are read from the archive content, so they are not hardcoded here.

References
----------
- ICOS Carbon Portal: https://data.icos-cp.eu/portal/
- ICOS data levels: https://www.icos-cp.eu/data-services/data-collection/data-levels-quality
- "ETC L2 ARCHIVE" data type: http://meta.icos-cp.eu/resources/cpmeta/etcArchiveProduct
- icoscp_core library: https://pypi.org/project/icoscp_core/
- Product descriptions: ``ICOSETC_Level2_products_description.pdf``, shipped in
  each archive.
"""

import logging
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import polars as pl
from icoscp_core.dataclient import DataClient
from icoscp_core.icos import bootstrap
from icoscp_core.metaclient import MetadataClient

logger = logging.getLogger(__name__)

L2_ARCHIVE_DATATYPE = "http://meta.icos-cp.eu/resources/cpmeta/etcArchiveProduct"


def get_ICOS_clients(token: str) -> tuple[MetadataClient, DataClient]:
    """Initialize the metadata and data clients for the ICOS data portal.

    Parameters
    ----------
    token: str
        Authentication token for the ICOS portal.
        Can be fetched from https://cpauth.icos-cp.eu. Token are only valid for 27 hours.

    Returns
    -------
    tuple[MetadataClient, DataClient]
        Metadata client (for searching) and data client (for downloading).
    """
    try:
        return bootstrap.fromCookieToken(token)

    # icoscp_core raises bare exceptions, we thus have to catch-all and re-raise
    except Exception as e:
        logger.error("Potential error in ICOS client initialization (token not shown).")
        logger.error(f"Original exception: {e}")
        raise


def download_ICOS_object(dobj_uri: str, folder: Path, data: DataClient) -> Path:
    """Download a given ICOS object.

    Parameters
    ----------
    dobj_uri: str
        URI to the object to download.
    folder: Path
        Path to the folder where the file will be saved.
    data: DataClient
        Initialized data client for the ICOS portal. Use `get_ICOS_clients`.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    folder.mkdir(parents=True, exist_ok=True)
    filename = data.save_to_folder(dobj_uri, str(folder))
    logger.info(f"Successfuly downloaded {filename}")
    return folder / filename


def list_L2_station_ids(meta: MetadataClient) -> list[str]:
    """List the ids of the stations that have a Level 2 archive.

    Parameters
    ----------
    meta: MetadataClient
        Initialized metadata client. Use `get_ICOS_clients`.

    Returns
    -------
    list[str]
        Sorted station ids, e.g. ["BE-Bra", ..., "CH-Dav", ...].
    """
    archives = meta.list_data_objects(datatype=L2_ARCHIVE_DATATYPE, limit=10_000)
    station_uris = {a.station_uri for a in archives}
    station_ids = sorted(s.id for s in meta.list_stations() if s.uri in station_uris)
    logger.info(f"{len(station_ids)} stations with a Level 2 archive: {station_ids}")
    return station_ids


def list_L2_stations_in_country(country: str, meta: MetadataClient) -> pl.DataFrame:
    """List the stations with a Level 2 archive in a given country.

    Parameters
    ----------
    country: str
        Two-letter country code, e.g. "CH".
    meta: MetadataClient
        Initialized metadata client. Use `get_ICOS_clients`.

    Returns
    -------
    pl.DataFrame
        One row per station with its id, name, lat, lon and elevation.
    """
    station_ids = list_L2_station_ids(meta)
    return pl.DataFrame(
        [
            (s.id, s.name, s.lat, s.lon, s.elevation)
            for s in meta.list_stations()
            if s.id in station_ids and s.country_code == country.upper()
        ],
        schema=["id", "name", "lat", "lon", "elevation"],
        orient="row",
    )


def find_L2_archive_uri(site: str, meta: MetadataClient) -> str:
    """Find the URI of the latest Level 2 archive of a station.

    Parameters
    ----------
    site: str
        ICOS station id, e.g. "CH-Dav".
    meta: MetadataClient
        Initialized metadata client. Use `get_ICOS_clients`.

    Returns
    -------
    str
        URI of the most recently submitted, non-deprecated archive.
    """
    station_uris = [s.uri for s in meta.list_stations() if s.id == site]
    if not station_uris:
        raise ValueError(f"No ICOS station with id {site!r}.")

    # Objects are sorted by submission time, most recent first.
    archives = meta.list_data_objects(
        datatype=L2_ARCHIVE_DATATYPE,
        station=station_uris,
        order_by={"prop": "submTime", "descending": True},
    )

    if not archives:
        raise FileNotFoundError(f"No Level 2 archive found for {site}.")
    if len(archives) > 1:
        logger.warning(f"{len(archives)} archives found for {site}, using the latest.")
    return archives[0].uri


def L2_product_name(filename: str, site: str) -> str | None:
    """Extract the product name from a Level 2 CSV file name.

    Parameters
    ----------
    filename: str
        File name, e.g. "ICOSETC_CH-Dav_FLUXNET_DD_INTERIM_L2.csv".
    site: str
        ICOS station id, e.g. "CH-Dav".

    Returns
    -------
    str | None
        Product name, e.g. "FLUXNET_DD", or None if the file is not a station
        Level 2 CSV.
    """
    match = re.fullmatch(rf"ICOSETC_{re.escape(site)}_(.+?)(?:_INTERIM)?_L2\.csv", filename)
    return match.group(1) if match else None


def download_ICOS_L2(
    site: str,
    folder: Path,
    meta: MetadataClient,
    data: DataClient,
    products: list[str] | None = None,
) -> list[Path]:
    """Download Level 2 CSVs of a station, skipping products already present.

    Parameters
    ----------
    site: str
        ICOS station id, e.g. "CH-Dav".
    folder: Path
        Path to the folder where the files will be saved.
    meta: MetadataClient
        Initialized metadata client. Use `get_ICOS_clients`.
    data: DataClient
        Initialized data client. Use `get_ICOS_clients`.
    products: list[str] | None
        Products to extract, named as in the archive files, e.g. "FLUXES",
        "FLUXNET_DD" or "VARINFO_METEO". If None (default) or empty, all
        products of the archive are extracted.

    Returns
    -------
    list[Path]
        Paths to the extracted files, in the order of `products` (sorted by
        product name if `products` is None or empty).
    """
    existing = {
        product: p for p in folder.glob("*.csv") if (product := L2_product_name(p.name, site))
    }
    if products and all(product in existing for product in products):
        logger.info(f"All products already present in {folder}, skipping download.")
        return [existing[product] for product in products]

    archive_uri = find_L2_archive_uri(site, meta)
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        archive_path = download_ICOS_object(archive_uri, Path(tmp), data)
        with zipfile.ZipFile(archive_path) as zf:
            # Members are matched on their base name, ignoring folders in the archive.
            members = {PurePosixPath(n).name: n for n in zf.namelist()}
            available = {
                product: name for name in members if (product := L2_product_name(name, site))
            }
            logger.info(f"Products available for {site}: {sorted(available)}")
            products = products or sorted(available)
            missing = [product for product in products if product not in available]
            if missing:
                raise FileNotFoundError(
                    f"Products {missing} not in archive. Available: {sorted(available)}."
                )

            paths = []
            for product in products:
                target = folder / available[product]
                target.write_bytes(zf.read(members[available[product]]))
                logger.info(f"Extracted {target}")
                paths.append(target)
    return paths
