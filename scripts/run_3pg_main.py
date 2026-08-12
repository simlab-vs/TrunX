"""Run the 3PG model end-to-end on the Solling reference dataset."""

import os

from trunx.config import threepg_data_folder
from trunx.gp3.PG3_model_impl import run_threepg_main

if __name__ == "__main__":
    file_path = os.path.join(threepg_data_folder, "solling_data.xlsx")
    fig, outputs = run_threepg_main(
        file_path, observed_data=None, plot_output=True, r_comparison=False
    )
