"""Sync project data from the S3 bucket into a local data folder.

Only objects that are missing or outdated locally are transferred, so re-running this
on a machine that already holds the data is cheap. Freshness is decided on size plus
modification time: a downloaded file gets the S3 `LastModified` stamped onto it, so an
unchanged object compares equal on the next run.

Objects keep their full key below the destination, so pulling `threepg_inputs` into
`~/shared_sim_lab/trunx` writes `~/shared_sim_lab/trunx/data/threepg_inputs/...`, which
is the layout `trunx.config` expects from a base directory.

Examples
--------
Refresh the 3PG inputs in the project data folder::

    uv run scripts/pull_data.py threepg_inputs

Preview a larger sync into a shared folder::

    uv run scripts/pull_data.py clean raw --dest ~/shared_sim_lab/trunx --dry-run
"""

import argparse
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from simlab_tools.storage import download_file, get_s3_client, list_bucket_contents

from trunx.config import (
    base_dir,
    s3_bucket,
    s3_datasets,
    s3_endpoint_url,
    s3_profile,
)

# Editor lock files and macOS folder metadata are noise in the bucket, never inputs.
IGNORED_NAMES = frozenset({".DS_Store"})

# Filesystems with coarse timestamps can round a stamped mtime down by up to a second,
# which would otherwise make an unchanged file look outdated on every run.
MTIME_TOLERANCE_S = 1.0


def is_ignored(key: str) -> bool:
    """Tell whether an object key is a folder marker or an editor artefact."""
    name = PurePosixPath(key).name
    return key.endswith("/") or name in IGNORED_NAMES or name.startswith("~$")


def resolve_prefix(dataset: str) -> str:
    """Resolve a dataset name to its bucket prefix, passing raw prefixes through.

    Parameters
    ----------
    dataset : str
        A name from the `s3.datasets` config section, or any value containing a "/",
        which is taken to be a bucket prefix already.

    Raises
    ------
    KeyError
        If the name is unknown and is not a raw prefix.
    """
    if "/" in dataset:
        return dataset
    if dataset not in s3_datasets:
        known = ", ".join(sorted(s3_datasets))
        raise KeyError(f"Unknown dataset {dataset!r}. Known datasets: {known}")
    return s3_datasets[dataset]


def is_up_to_date(local_file: Path, obj: dict[str, Any]) -> bool:
    """Tell whether a local file already matches the size and age of an S3 object."""
    if not local_file.exists():
        return False
    stat = local_file.stat()
    if stat.st_size != obj["Size"]:
        return False
    return stat.st_mtime + MTIME_TOLERANCE_S >= obj["LastModified"].timestamp()


def select_outdated(objects: list[dict[str, Any]], dest: Path) -> tuple[list[dict[str, Any]], int]:
    """Split listed objects into those needing a download and a count of current ones."""
    outdated = []
    up_to_date = 0
    for obj in objects:
        if is_ignored(obj["Key"]):
            continue
        if is_up_to_date(dest / obj["Key"], obj):
            up_to_date += 1
        else:
            outdated.append(obj)
    return outdated, up_to_date


def pull_object(client: Any, bucket: str, obj: dict[str, Any], dest: Path) -> None:
    """Download one object and stamp the remote modification time onto the local file."""
    local_file = dest / obj["Key"]
    download_file(client, bucket, obj["Key"], local_file)
    last_modified = obj["LastModified"].timestamp()
    os.utime(local_file, (last_modified, last_modified))


def format_size(num_bytes: float) -> str:
    """Render a byte count in the largest unit that keeps it readable."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(num_bytes) < 1024 or unit == "GiB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GiB"


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Sync project data from the S3 bucket into a local data folder."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        metavar="DATASET",
        help=f"Datasets to pull. Known names: {', '.join(sorted(s3_datasets))}. "
        "A value containing a '/' is used as a bucket prefix directly.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(base_dir),
        help="Base directory to sync into; objects keep their full key below it "
        "(default: the project data root, %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what is outdated without downloading anything.",
    )
    args = parser.parse_args()
    if not args.datasets:
        parser.error(f"no dataset given. Known datasets: {', '.join(sorted(s3_datasets))}")
    return args


def main() -> int:
    """Sync the requested datasets and report what was transferred."""
    args = parse_args()

    try:
        prefixes = [resolve_prefix(dataset) for dataset in args.datasets]
    except KeyError as error:
        print(error.args[0], file=sys.stderr)
        return 2

    dest = args.dest.expanduser().resolve()
    client = get_s3_client(s3_endpoint_url, profile=s3_profile)
    print(f"Bucket:      s3://{s3_bucket} ({s3_endpoint_url})")
    print(f"Destination: {dest}\n")

    outdated: list[dict[str, Any]] = []
    up_to_date = 0
    for prefix in prefixes:
        objects = list_bucket_contents(client, s3_bucket, prefix=prefix)
        stale, current = select_outdated(objects, dest)
        outdated += stale
        up_to_date += current
        print(f"{prefix}: {len(stale)} to pull, {current} up to date")

    total_bytes = sum(obj["Size"] for obj in outdated)
    if not outdated:
        print(f"\nEverything is up to date ({up_to_date} files).")
        return 0

    print(f"\n{len(outdated)} file(s) to pull, {format_size(total_bytes)}\n")
    if args.dry_run:
        for obj in outdated:
            print(f"  {obj['Key']} ({format_size(obj['Size'])})")
        return 0

    failed = []
    for index, obj in enumerate(outdated, start=1):
        print(f"[{index}/{len(outdated)}] {obj['Key']}")
        try:
            pull_object(client, s3_bucket, obj, dest)
        except Exception as error:  # noqa: BLE001 - report and continue with the rest
            print(f"  failed: {error}", file=sys.stderr)
            failed.append(obj["Key"])

    print(
        f"\nPulled {len(outdated) - len(failed)}/{len(outdated)} file(s), "
        f"{up_to_date} already up to date."
    )
    if failed:
        print(f"{len(failed)} file(s) failed:", file=sys.stderr)
        for key in failed:
            print(f"  {key}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
