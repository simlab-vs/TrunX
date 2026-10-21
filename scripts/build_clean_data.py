"""Build the clean data used across the project, in `clean_data_folder`, from the raw data.

Steps, in dependency order:

1. ICP monthly deposition (`icp_monthly_deposition.parquet`), joined to the ICP trees.
2. Tree-level data of NFI, EFM, LWF and ICP and their combination
   (`*_tree_data.parquet`, `trunx_tree_level_data.parquet`).
3. ICP inventory (`icp_inventory_data.parquet`), which needs the ICP trees.
4. Plot-level data of NFI, EFM, LWF and ICP and their combination
   (`*_cleaned.parquet`, `icp_plot_data.parquet`, `trunx_plot_level_data.parquet`), which
   needs the ICP inventory.
5. With `--weather`: monthly weather of every plot (`trunx_plot_weather.parquet`), which
   needs the combined trees and the ICP and ERA5 weather files.

The cached 3-PG input files (`threepg_inputs/*_plots/`) are not rebuilt; delete them so
they are created again from the new data.
"""

import argparse
import logging

from trunx.datasets.combined_data import get_combined_plot_data, get_combined_tree_data
from trunx.datasets.combined_weather_data import build_tree_plot_weather
from trunx.datasets.icp_level2_data import prepare_icp_inventory_data
from trunx.datasets.icp_process_monthly_depositions import process_monthly_depositions

logger = logging.getLogger(__name__)


def build_clean_data(weather: bool = False) -> None:
    """Build the project's clean data from the raw data.

    Parameters
    ----------
    weather : bool
        Whether to also rebuild the plots' monthly weather.
    """
    logger.info("1/5 ICP monthly deposition")
    process_monthly_depositions()
    logger.info("2/5 Tree-level data")
    get_combined_tree_data()
    logger.info("3/5 ICP inventory")
    prepare_icp_inventory_data()
    logger.info("4/5 Plot-level data")
    get_combined_plot_data()
    if weather:
        logger.info("5/5 Plot weather")
        build_tree_plot_weather()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--weather", action="store_true", help="also rebuild trunx_plot_weather.parquet"
    )
    build_clean_data(weather=parser.parse_args().weather)
