import marimo

__generated_with = "0.24.0"
app = marimo.App(app_title="LFW EDA")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    from typing import Any, Callable

    return Any, Callable, os


@app.cell
def _():
    import polars as pl
    import polars.selectors as cs

    return cs, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### LFW EDA
    The following notebook helps with the Exploratory Data Analysis (EDA) of the LFW-issued data.
    """)
    return


@app.cell
def _(os):
    RAW_DATA_PATH: str = os.path.join(os.getcwd(), "data", "raw")
    return (RAW_DATA_PATH,)


@app.cell(hide_code=True)
def _(RAW_DATA_PATH: str, mo):
    mo.md(rf"""
    Before proceeding further, we may inspect the content of the `.xlsx` files at {RAW_DATA_PATH}.
    """)
    return


@app.cell(hide_code=True)
def _(Any, Callable):
    def callback(new: Any, fn: Callable[[Any], None]) -> None:
        if new is not None:
            fn(new)

    return (callback,)


@app.cell(hide_code=True)
def _(mo):
    get_selected_xlsx, set_selected_xlsx = mo.state(None)
    return get_selected_xlsx, set_selected_xlsx


@app.cell(hide_code=True)
def _(RAW_DATA_PATH: str, callback, mo, os, set_selected_xlsx):
    _xlsx_file_browser: mo.ui.file_browser = mo.ui.file_browser(
        initial_path=RAW_DATA_PATH,
        filetypes=[".xlsx"],
        label="Select a file to inspect its raw content. Cells below will rerun accordingly.",
        multiple=False,
        on_change=lambda filepath: callback(
            os.path.join(RAW_DATA_PATH, filepath[0].name), set_selected_xlsx
        ),
    )
    _xlsx_file_browser
    return


@app.cell(hide_code=True)
def _(get_selected_xlsx, pl):
    _dfs: tuple[pl.DataFrame] | None = None
    _selected_xlsx_filename: str | None = get_selected_xlsx()
    if _selected_xlsx_filename is not None:
        _dfs = pl.read_excel(_selected_xlsx_filename, sheet_id=0)
    _dfs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Going through those files, we deduct that:
    - `LAI_Licor_3_rings_all_years.xlsx` states the estimated Leaf Area Index (LAI) for plots, and subplots/periods, using the Miller and Norman & Campbell methods.
    - `legend_deposition_2026-07-28.xlsx` gives a detailed description, and unit (when available), of the short-form variable acronyms for deposition (dep) data, as well as its data semantic.
    - `legend_foliage_dry_weight.xlsx` serves the same purpose as `legend_deposition_2026-07-28.xlsx`, but for foliage data.

    With that in mind, we can now explore the data itself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    get_selected_csv, set_selected_csv = mo.state(None)
    return get_selected_csv, set_selected_csv


@app.cell(hide_code=True)
def _(RAW_DATA_PATH: str, callback, mo, os, set_selected_csv):
    _csv_file_browser: mo.ui.file_browser = mo.ui.file_browser(
        initial_path=RAW_DATA_PATH,
        filetypes=[".csv"],
        label="Select one file to inspect its data. Cells below will rerun accordingly.",
        multiple=False,
        on_change=lambda filepath: callback(
            os.path.join(RAW_DATA_PATH, filepath[0].name), set_selected_csv
        ),
    )
    _csv_file_browser
    return


@app.cell(hide_code=True)
def _(get_selected_csv, pl):
    df: pl.DataFrame | None = None
    _selected_csv_filename: str | None = get_selected_csv()
    if _selected_csv_filename is not None:
        df = pl.read_csv(
            _selected_csv_filename,
            infer_schema=False,
            separator=";",
            **(
                {
                    "null_values": [".", " "],
                    "encoding": "windows-1252",
                    "skip_lines": 2,
                }
                if "litterfal" in _selected_csv_filename
                else {"null_values": "NA"}
            ),
        )
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Please keep in mind that, for now, `infer_schema` is set to `False` as it is the engine won't infer types properly because of particular values in the dataset. On top of that, `litterfall` data require special instructions when loading to 1) skip unnecessary lines (which contain important legends to understand the data), and 2) handle invalid utf-8 sequences. Nulls inside these files are marked by the `.` character. The equivalent for all other files is the string `NA`.

    The different `csv` files can be understood as:
    - `lwf_foliage_dw100_i_2026-07-30.csv` provides foliage dry weight (in grams, at 65°C reference temperature) for individual trees, identified by `sample_id`.
    - `lwf_foliage_dw100_plot_2026-07-30.csv` provides averaged foliage dry weight (in grams, at 65°C reference temperature, again) for different species (`species` column), as measured on pooled samples.
    - `monthly_*` give monthly deposition (`dep`) and foliage (`weight`) data, averaged (is it ?) at monthly intervals.
    - The same states for `period_*` data, at set periods.

    We can now take a closer look at the schema, and associated units of the data, when available. For that, we first need to parse the Excel legend/definitions into a map.
    """)
    return


@app.cell
def _():
    DEPOSITION_LEGEND_FILENAME: str = "legend_deposition_2026-07-28.xlsx"
    return (DEPOSITION_LEGEND_FILENAME,)


@app.cell(hide_code=True)
def _(DEPOSITION_LEGEND_FILENAME: str, RAW_DATA_PATH: str, cs, os, pl):
    columns_map: dict[str, str] = {
        k: v[0]
        for k, v in pl.read_excel(
            os.path.join(RAW_DATA_PATH, DEPOSITION_LEGEND_FILENAME),
            read_options={"skip_rows": 3},
        )
        .head(-2)
        .select(cs.by_index(0, 1))
        .rows_by_key(
            cs.by_index(0), named=False, include_key=False, unique=True
        )
        .items()
    }
    columns_map
    return (columns_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Considering this, we can now rename the `DataFrame` to replace the short-form parameter names by their unit-augmented one.

    Let's assess the null proportion, across variables of the dataset.
    """)
    return


@app.cell(hide_code=True)
def _(columns_map: dict[str, str], df: "pl.DataFrame | None", pl):
    df_renamed: pl.DataFrame = df.rename(columns_map, strict=False)
    return (df_renamed,)


@app.cell(hide_code=True)
def _(df_renamed: "pl.DataFrame", mo):
    mo.md(
        "\n".join(
            f"- **{k}**: {(v[0] * 100):.2f}% of nulls"
            for k, v in sorted(
                (df_renamed.null_count() / df_renamed.height)
                .to_dict(as_series=False)
                .items(),
                key=lambda v: v[1][0],
                reverse=True,
            )
        )
    )
    return


if __name__ == "__main__":
    app.run()
