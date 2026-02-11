"""
Script to download ICP Forests Level II datasets.

1. Station information
2. Tree species distribution.

"""

import requests


def icp_download_level_two(url: str, output_file: str):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {output_file}")
    else:
        print(f"Failed to download: HTTP {response.status_code}")


if __name__ == "__main__":
    # ICP Forests Level II plots
    url = "https://icp-forests.org/open_data/level_ii/gpd/gpd_level_ii.csv"
    output_file = "./data/raw/ICP/gpd_level_ii.csv"
    icp_download_level_two(url, output_file)

    # Tree species distribution
    url = "https://icp-forests.org/open_data/level_ii/gpd/ts_level_ii.csv"
    output_file = "./data/raw/ICP/tree_species_distribution.csv"
    icp_download_level_two(url, output_file)
