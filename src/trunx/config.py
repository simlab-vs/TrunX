"""Configuration variables for the project.

Used for centralizing paths definitions, among other things.
"""

import os
from pathlib import Path
from typing import Literal

import yaml

# Root of the repository.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def load_config(project_root):
    """Load project configuration file."""
    with open(os.path.join(project_root, "config.yaml")) as p:
        config = yaml.safe_load(p)
    return config


config = load_config(project_root)
mode = config.get("mode", "local")

# Determine base directory based on mode. TRUNX_BASE_DIR wins over both modes, so a
# cluster job can point every data folder at a shared directory without editing
# the checked-in config.
env_base_dir = os.environ.get("TRUNX_BASE_DIR")
if env_base_dir:
    base_dir = env_base_dir
elif mode == "server":
    base_dir = config["server_base_dir"]
elif mode == "local":
    base_dir = project_root
else:
    raise ValueError(f"Invalid mode '{mode}'. Must be 'local' or 'server'.")

data_folder = Path(os.path.join(base_dir, "data/"))
raw_data_folder = Path(os.path.join(base_dir, "data/raw/"))
clean_data_folder = Path(os.path.join(base_dir, "data/clean/"))
threepg_data_folder = Path(os.path.join(base_dir, "data/threepg_inputs/"))
intermediate_data_folder = Path(os.path.join(base_dir, "data/intermediate/"))
results_data_folder = Path(os.path.join(base_dir, "data/results/"))
era5_data_folder = Path(os.path.join(base_dir, "data/raw/ERA5/"))

icos_raw_data_folder = raw_data_folder / "ICOS"
icp_raw_data_folder = raw_data_folder / "ICP"

# Remote object storage holding the project data (see scripts/pull_data.py).
s3_config = config.get("s3", {})
s3_endpoint_url = s3_config.get("endpoint_url", "")
s3_bucket = s3_config.get("bucket", "")
s3_profile = s3_config.get("profile")
s3_datasets: dict[str, str] = s3_config.get("datasets", {})

# Define key constants
Species = Literal["spruce", "pine", "beech", "oak"]
Levels = Literal["tree", "plot"]

SPECIES_INDICES = {
    "Picea abies": 1,
    "Pinus sylvestris": 2,
    "Fagus sylvatica": 3,
    "Quercus robur": 4,
    "Quercus petraea": 5,
}
