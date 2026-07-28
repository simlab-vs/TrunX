"""Literature data processing for 3PG model implementation."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import pandas as pd
    import pyreadr

    from trunx.config import threepg_data_folder

    return os, pd, pyreadr, threepg_data_folder


@app.cell
def _(pyreadr):
    data = pyreadr.read_r("./models/r3PG/vignettes_build/vignette_data/solling.rda")
    data.keys()
    return


@app.cell
def _(os, pd, threepg_data_folder):
    from docx import Document

    param_default = pd.read_excel(os.path.join(threepg_data_folder, "data.default.xlsx"))

    def parse_prior_posterior_table(table):
        """Parse a Table S3/S4-style docx table (two-row header) into a DataFrame."""
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        columns = [
            "parameter",
            "prior_min",
            "prior_max",
            "posterior_2.5%",
            "posterior_50%",
            "posterior_97.5%",
            "reduction_pct",
            "conv_point_est",
            "conv_upper_ci",
        ]
        # The first two rows are the (merged) header; data starts on row 2.
        return pd.DataFrame(rows[2:], columns=columns)

    file_path = os.path.join("./literature/", "gcb15011-sup-0001-supinfo.docx")
    doc = Document(file_path)
    piab_table = parse_prior_posterior_table(doc.tables[2])
    fasy_table = parse_prior_posterior_table(doc.tables[3])
    return fasy_table, param_default, piab_table


@app.cell
def _(param_default, pd, piab_table):
    # Species: P. abies
    piab_df = piab_table[["parameter", "prior_min", "posterior_50%", "prior_max"]].rename(
        columns={
            "prior_min": "min",
            "posterior_50%": "Picea abies",
            "prior_max": "max",
        }
    )
    piab_df = pd.merge(param_default, piab_df, on="parameter", how="left")
    piab_df["Picea abies"] = piab_df["Picea abies"].fillna(piab_df["default"])
    piab_df = piab_df[["parameter", "min", "Picea abies", "max"]]

    return


@app.cell
def _(fasy_table, param_default, pd):
    # Species: F. sylvatica
    fasy_df = fasy_table[["parameter", "prior_min", "posterior_50%", "prior_max"]].rename(
        columns={
            "prior_min": "min",
            "posterior_50%": "Fagus sylvatica",
            "prior_max": "max",
        }
    )
    fasy_df = pd.merge(param_default, fasy_df, on="parameter", how="left")
    fasy_df["fasy"] = fasy_df["Fagus sylvatica"].fillna(fasy_df["default"])
    fasy_df = fasy_df[["parameter", "min", "Fagus sylvatica", "max"]]

    return


if __name__ == "__main__":
    app.run()
