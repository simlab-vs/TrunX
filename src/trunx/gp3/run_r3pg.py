"""Run r3PG implementation in Python."""

import os
from typing import Optional

import polars as pl
import rpy2.robjects as ro
from rpy2.rinterface_lib import callbacks
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# # Disable rpy2 callbacks
# def silent_callback(x: str) -> None:
#     """Silence rpy2 console output."""
#     pass

# callbacks.consolewrite_print = silent_callback
# callbacks.consolewrite_warnerror = lambda s: None

callbacks.consolewrite_print = lambda s: None  # type: ignore
callbacks.consolewrite_warnerror = lambda s: None  # type: ignore


def run_comparison_r(file_path: str | None = None) -> pl.DataFrame | None:
    """
    Run R comparison script with optional file path.

    Parameters
    ----------
    file_path (str, optional): Path to Excel file. If None, uses built-in data.

    Returns
    -------
    polars.DataFrame: out_3PG results
    """
    print("File path passed:", file_path)

    # Path to your R script
    r_script = "/Users/glory/Documents/Research/TrunkX/models/r3PG/Trunx_comp.R"

    # Method A: Pass file_path as an argument to the R script
    if file_path:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"Using custom file: {file_path}")

        # Source the script and pass file_path as an argument
        # This creates a variable 'file_path' in the R environment
        ro.globalenv["file_path"] = file_path
        ro.globalenv["use_builtin"] = False
        ro.r(f'source("{r_script}")')
    else:
        print("Using built-in example data")
        # Source the script without setting file_path
        # The R script should check if file_path exists
        ro.r(f'source("{r_script}")')

    with localconverter(ro.default_converter + pandas2ri.converter):
        # Convert to pandas first, then to polars
        df = pl.DataFrame(ro.conversion.rpy2py(ro.r("out_3PG")))
        # df = pl.from_pandas(pandas_df)

    return df


# Example usage:
if __name__ == "__main__":
    # Test with built-in data
    print("Built-in data")
    df1 = run_comparison_r()
    if df1 is not None:
        print(df1.shape)

    # Test with custom Excel file
    print("Custom Excel file")
    file_path = "/Users/glory/Documents/Research/TrunkX/data/data.inputonespecies.xlsx"
    df2 = run_comparison_r(file_path)
    if df2 is not None:
        print(df2.shape)
