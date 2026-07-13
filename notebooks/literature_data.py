"""Literature data processing for 3PG model implementation."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import pandas as pd
    import pyreadr

    return pd, pyreadr


@app.cell
def _(pyreadr):
    data = pyreadr.read_r("./models/r3PG/vignettes_build/vignette_data/solling.rda")
    data.keys()
    return


@app.cell
def _(pd):
    from docx import Document

    param_default = pd.read_excel("../data/data.default.xlsx")

    def docx_tables_to_dfs(file_path):
        """Extract all tables from a .docx file and return a list of DataFrames."""
        doc = Document(file_path)
        all_dfs = []

        for table in doc.tables:
            data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                data.append(row_data)

            # Convert to DataFrame
            if data:
                # Assume first row is header
                df = pd.DataFrame(data[1:], columns=data[0])
                all_dfs.append(df)

        return all_dfs

    file_path = "../data/gcb15011-sup-0001-supinfo-2.docx"
    dfs = docx_tables_to_dfs(file_path)

    # Species: P. abies
    param_df = dfs[2]
    param_df.head()
    parameter_df = pd.DataFrame(param_df.values[1:], columns=param_df.values[0])[
        ["Parameter", "50.00%"]
    ]
    parameter_df.rename(columns={"50.00%": "piab", "Parameter": "parameter"}, inplace=True)

    parameter_df = pd.merge(param_default, parameter_df, on="parameter", how="left")
    parameter_df["piab"] = parameter_df["piab"].fillna(parameter_df["default"])

    parameter_df = parameter_df[["parameter", "piab"]]
    return


if __name__ == "__main__":
    app.run()
