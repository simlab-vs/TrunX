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

# Determine base directory based on mode
if mode == "server":
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
images_folder = Path(os.path.join(base_dir, "images/"))

icos_raw_data_folder = raw_data_folder / "ICOS"
icp_raw_data_folder = raw_data_folder / "ICP"

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
