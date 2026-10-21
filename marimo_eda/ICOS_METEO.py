import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    return Path, mo, pl


@app.cell
def _(Path, pl):
    file_path = Path("data/raw/ICOS/ICOSETC_CH-Dav_METEO_INTERIM_L2.csv")

    meteo = pl.read_csv(file_path, infer_schema_length=100000, null_values="-9999")
    pl.Config.set_tbl_cols(-1)
    return (meteo,)


@app.cell
def _(meteo):
    meteo
    return


@app.cell
def _(meteo):
    meteo.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Link for Column name Metadata: https://www.swissfluxnet.ethz.ch/index.php/data/variables/variable-abbreviations/
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    
    """)
    return


if __name__ == "__main__":
    app.run()
