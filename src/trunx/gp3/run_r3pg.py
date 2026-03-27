"""Run r3PG implementation in Python."""

import os
from typing import Optional

import polars as pl
import rpy2.robjects as ro
from rpy2.rinterface_lib import callbacks
from rpy2.robjects import default_converter, pandas2ri
from rpy2.robjects.conversion import localconverter

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

    r_script = "/Users/glory/Documents/Research/TrunkX/models/r3PG/Trunx_comp.R"

    with localconverter(ro.default_converter + pandas2ri.converter):
        if file_path:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            print(f"Using custom file: {file_path}")
            ro.globalenv["file_path"] = ro.StrVector([file_path])
            ro.globalenv["use_builtin"] = False
        else:
            print("Using built-in example data")
            ro.globalenv["use_builtin"] = False

        ro.r(f'source("{r_script}")')
        r_out = ro.r("out_3PG")
        df = pl.DataFrame(ro.conversion.rpy2py(r_out))
        return df


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
