"""Master Marimo EDA application for the LWF datasets."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    import marimo as mo

    dataset_options = {
        "LI-COR LAI": "LAI_Licor.py",
        "Individual foliage dry weight": "Individual_LWF_gew100.py",
        "Plot-level foliage dry weight": "Plot_LWF_gew100.py",
        "Monthly deposition": "Monthly_dep_LWF.py",
        "Periodic deposition": "Periodic_dep_LWF.py",
        "Monthly Davos ICOS litterfall": "Monthly_Litterfall_Davos.py",
        "Periodic Davos ICOS litterfall": "Periodic_Litterfall_Davos.py",
        "Monthly LWF litterfall": "Monthly_Litterfall_LWF.py",
        "Periodic LWF litterfall": "Periodic_Litterfall_LWF.py",
        "LWF site metadata": "LWF_site_metadata_table.py",
        "LWF sites": "sites.py",
        "DBH": "dbh_flagged.py",
    }

    dataset_selector = mo.ui.dropdown(
        options=dataset_options,
        value="LI-COR LAI",
        label="Dataset",
    )

    mo.vstack(
        [
            mo.md(
                """
                # LWF Exploratory Data Analysis

                Select the LWF dataset you want to explore.
                """
            ),
            dataset_selector,
        ]
    )
    return (
        Path,
        dataset_options,
        dataset_selector,
        mo,
        module_from_spec,
        spec_from_file_location,
    )


@app.cell
def _(Path, dataset_selector, module_from_spec, spec_from_file_location):
    notebook_dir = Path(__file__).resolve().parent
    notebook_path = notebook_dir / dataset_selector.value

    module_name = f"_lwf_eda_{notebook_path.stem}"

    spec = spec_from_file_location(module_name, notebook_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load notebook: {notebook_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    selected_app = module.app
    return (selected_app,)


@app.cell
async def _(dataset_options, dataset_selector, mo, selected_app):
    result = await selected_app.embed()

    selected_name = next(
        name
        for name, path in dataset_options.items()
        if path == dataset_selector.value
    )

    mo.vstack(
        [
            mo.md(f"## {selected_name}"),
            result.output,
        ]
    )

@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()