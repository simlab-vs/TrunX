"""Create 3PG input files from the combined plot-level and weather tables.

This is the single builder of 3PG plot inputs for the project. Works for any plot
in `trunx_plot_level_data.parquet` (NFI, EFM, LWF, ICP), using its monthly weather
from `trunx_plot_weather.parquet`. Deposition is only available for ICP plots;
other sources get missing (null) deposition.
"""

import logging
import os

import numpy as np
import pandas as pd
import polars as pl

from trunx.config import SPECIES_INDICES, clean_data_folder, threepg_data_folder
from trunx.gp3.age_regression import fit_models, predict_age_from_dbh
from trunx.gp3.prepare_deposition import load_monthly_deposition

logger = logging.getLogger(__name__)

_CLIMATE_COLUMNS = ["year", "month", "tmp_ave", "tmp_min", "tmp_max", "frost_days", "prcp", "srad"]
# Needed at the first survey: the initial state, plus mean DBH to estimate the stand age
_FIRST_SURVEY_COLUMNS = ["n_stems", "biom_stem", "biom_root", "biom_foliage", "dbh_mean"]
_DEP_COLUMNS = ["dep_n_tot", "dep_s_so4"]
# Deposition columns of a plot or month without deposition data
_MISSING_DEPOSITION = [pl.lit(None, dtype=pl.Float64).alias(c) for c in _DEP_COLUMNS]
# Years of a plot's deposition record averaged to fill months it does not cover
_DEP_FILL_YEARS = 5


_LITERATURE_SOURCES = {
    "Forrester": "literature_params_forrester_forrester.parquet",
    "Forrester_default": "literature_params_forrester_default.parquet",
    "Trotsiuk": "literature_params_trotsiuk.parquet",
}


def _load_species_param_bound(
    species_name: str, literature_source: str = "Forrester"
) -> pd.DataFrame:
    """Build a param_bound table for one species from the literature parquet.

    Parameters
    ----------
    species_name : str
        Species name as it appears in the literature parquet (e.g. "Picea abies").
    literature_source : str
        Which literature table to load bounds from — one of `_LITERATURE_SOURCES`
        (`"Forrester"`, `"Forrester_default"`, `"Trotsiuk"`).

    Returns
    -------
    pd.DataFrame
        Columns: param_name, default, min, max.
    """
    if literature_source not in _LITERATURE_SOURCES:
        raise ValueError(f"Unknown literature source: {literature_source}")
    literature_path = os.path.join(threepg_data_folder, _LITERATURE_SOURCES[literature_source])
    literature_bound = pd.read_parquet(literature_path)
    literature_bound = literature_bound[literature_bound["species"] == species_name]
    if literature_bound.empty:
        raise ValueError(f"No literature parameter bounds found for species: {species_name}")
    param_bound = literature_bound.rename(columns={"parameter": "param_name"})

    return param_bound[["param_name", "default", "min", "max"]]


def parameters_from_param_bound(param_bound: pd.DataFrame, species_name: str) -> pd.DataFrame:
    """Build the `parameters` sheet from `param_bound`'s defaults, so both sheets agree.

    Parameters
    ----------
    param_bound : pd.DataFrame
        Output of `_load_species_param_bound`.
    species_name : str
        Name of the species column.

    Returns
    -------
    pd.DataFrame
        Columns: parameter, <species_name>.
    """
    return param_bound[["param_name", "default"]].rename(
        columns={"param_name": "parameter", "default": species_name}
    )


def _plot_observations(plot_data: pl.DataFrame, plot_id: str, source: str) -> pl.DataFrame:
    """Select one plot's surveys of the species that have 3PG parameters."""
    required = {"basal_area", "height", "lai", "lat", "altitude", "dbh_qmd", "specie", "stand_age"}
    missing = sorted((required | set(_FIRST_SURVEY_COLUMNS)) - set(plot_data.columns))
    if missing:
        raise ValueError(
            f"Plot data is missing columns {missing}; regenerate it with "
            "`combined_data.get_combined_plot_data()`"
        )
    plot_df = plot_data.filter((pl.col("plot_id") == plot_id) & (pl.col("source") == source))
    if plot_df.is_empty():
        raise ValueError(f"No plot data for plot_id={plot_id!r}, source={source!r}")

    unsupported = sorted(set(plot_df["specie"]) - set(SPECIES_INDICES))
    if unsupported:
        logger.warning("Dropping species without 3PG parameters from %s: %s", plot_id, unsupported)
        plot_df = plot_df.filter(pl.col("specie").is_in(list(SPECIES_INDICES)))
    if plot_df.is_empty():
        raise ValueError(f"No species of plot {plot_id} has 3PG parameters: {unsupported}")

    return plot_df.with_columns(
        pl.col("date").dt.year().alias("year"), pl.col("date").dt.month().alias("month")
    ).sort("date", "specie")


def _plot_weather(weather_data: pl.DataFrame, plot_id: str, source: str) -> pl.DataFrame:
    """Select one plot's monthly weather in 3PG climate-sheet columns."""
    weather_df = (
        weather_data.filter((pl.col("plot_id") == plot_id) & (pl.col("source") == source))
        .select(_CLIMATE_COLUMNS)
        .with_columns(pl.col("year").cast(pl.Int64), pl.col("month").cast(pl.Int64))
        .sort("year", "month")
    )
    if weather_df.is_empty():
        raise ValueError(f"No weather data for plot_id={plot_id!r}, source={source!r}")
    return weather_df


def _month_index(year: pl.Expr, month: pl.Expr) -> pl.Expr:
    """Consecutive month number, used to compare and count year-month pairs."""
    return year * 12 + month - 1


def _planting_year(
    survey: dict, start_year: int, age_models: dict[str, tuple[float, float]]
) -> int:
    """Planting year from the measured stand age, otherwise from the age-vs-DBH model."""
    age = survey["stand_age"]
    if age is None:
        if survey["specie"] not in age_models:
            raise ValueError(f"No age model for species {survey['specie']!r}")
        a, b = age_models[survey["specie"]]
        age = predict_age_from_dbh(np.array([survey["dbh_mean"]]), a, b)
    return int(start_year - round(age))


def _species_sheet(
    first_surveys: pl.DataFrame, start_year: int, age_models: dict[str, tuple[float, float]]
) -> pl.DataFrame:
    """Build the species sheet, one row per species, with the initial state of the first survey."""
    return first_surveys.select(
        pl.col("specie").alias("species"),
        pl.Series(
            "planted",
            [
                f"{_planting_year(survey, start_year, age_models)}-01"
                for survey in first_surveys.iter_rows(named=True)
            ],
        ),
        pl.lit(0.5).alias("fertility"),
        pl.col("n_stems").alias("stems_n"),
        "biom_stem",
        "biom_root",
        "biom_foliage",
    )


def _site_sheet(first_survey: dict, last_survey: dict) -> pl.DataFrame:
    """Build the site sheet from the template, simulating from first to last survey."""
    return pl.read_excel(
        os.path.join(threepg_data_folder, "data.input.xlsx"), sheet_name="site"
    ).with_columns(
        pl.lit(first_survey["lat"]).alias("latitude"),
        pl.lit(first_survey["altitude"]).alias("altitude"),
        pl.lit(f"{first_survey['year']}-{first_survey['month']:02d}").alias("from"),
        pl.lit(f"{last_survey['year']}-{last_survey['month']:02d}").alias("to"),
    )


def _plot_gpp(
    plot_id: str,
    source: str,
    gpp: pl.DataFrame | None,
    first_survey: dict,
    last_survey: dict,
) -> pl.DataFrame:
    """Select an ICP plot's monthly GOSIF GPP within the simulated period; empty otherwise."""
    schema = {"year": pl.Int32, "month": pl.Int8, "GPP": pl.Float64}
    if source != "ICP":
        return pl.DataFrame(schema=schema)
    if gpp is None:
        gpp = pl.read_csv(os.path.join(clean_data_folder, "GOSIF_GPP_icp.csv"))
    month = _month_index(pl.col("year"), pl.col("month"))
    return gpp.filter(
        (pl.col("plot_id") == float(plot_id))
        & month.is_between(
            _month_index(pl.lit(first_survey["year"]), pl.lit(first_survey["month"])),
            _month_index(pl.lit(last_survey["year"]), pl.lit(last_survey["month"])),
        )
    ).select(pl.col(name).cast(dtype) for name, dtype in schema.items())


def _observed_sheet(surveys: pl.DataFrame) -> pl.DataFrame:
    """Build the observed sheet, one row per survey and species; missing values stay null."""
    return surveys.select(
        "specie",
        "month",
        "year",
        pl.col("date").dt.month_end().alias("date"),
        pl.col("dbh_qmd").alias("DBH"),
        pl.col("biom_stem").alias("WS"),
        pl.col("biom_foliage").alias("WF"),
        pl.col("biom_root").alias("WR"),
        pl.col("basal_area").alias("BA"),
        pl.col("height").alias("Height"),
        pl.col("n_stems").alias("N"),
        pl.col("lai").alias("LAI"),
    )


def _all_observed_sheet(observed: pl.DataFrame, gpp: pl.DataFrame) -> pl.DataFrame:
    """Add monthly GPP rows to the observed sheet, one row per survey or GPP month and species.

    Stand-level GPP is repeated for every species; missing values stay null.
    """
    gpp_obs = observed.select("specie").unique().join(gpp, how="cross")
    return (
        observed.join(gpp_obs, on=["specie", "year", "month"], how="full", coalesce=True)
        .with_columns(pl.date(pl.col("year"), pl.col("month"), 1).dt.month_end().alias("date"))
        .sort("date", "specie")
    )


def _fill_from_nearest_years(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Fill nulls of `col` with its same-calendar-month mean over the nearest observed years."""
    observed = df.filter(pl.col(col).is_not_null()).select(
        "month", pl.col("year").alias("obs_year"), pl.col(col).alias("obs")
    )
    fill = (
        df.filter(pl.col(col).is_null())
        .select("year", "month")
        .join(observed, on="month")
        .group_by("year", "month")
        .agg(
            pl.col("obs")
            .sort_by((pl.col("obs_year") - pl.col("year")).abs(), "obs_year")
            .head(_DEP_FILL_YEARS)
            .mean()
            .alias("fill")
        )
    )
    return (
        df.join(fill, on=["year", "month"], how="left")
        .with_columns(pl.coalesce(col, "fill").alias(col))
        .drop("fill")
    )


def add_deposition_to_weather(
    weather_df: pl.DataFrame,
    plot_id: str,
    deposition: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Join an ICP plot's monthly deposition onto its weather.

    Months without deposition, e.g. before or after monitoring, get the same-calendar-month
    mean of the `_DEP_FILL_YEARS` nearest years of the plot's record. Months this can't
    fill, and every month of plots without any deposition, stay missing (null).

    Parameters
    ----------
    weather_df : pl.DataFrame
        Monthly weather with ``year`` and ``month`` columns.
    plot_id : str
        ICP plot identifier.
    deposition : pl.DataFrame | None
        Pre-loaded monthly ICP deposition; loaded from disk when None.

    Returns
    -------
    pl.DataFrame
        ``weather_df`` with ``dep_n_tot`` and ``dep_s_so4`` columns.
    """
    if deposition is None:
        deposition = load_monthly_deposition()
    record = deposition.filter(pl.col("plot_id") == plot_id).select(
        pl.col("year").cast(pl.Int64), pl.col("month").cast(pl.Int64), *_DEP_COLUMNS
    )
    if record.is_empty():
        logger.warning("No deposition for plot_id %s — leaving it missing", plot_id)
        return weather_df.with_columns(_MISSING_DEPOSITION)

    months = weather_df.select(pl.col("year", "month").cast(pl.Int64))
    dep = months.join(record, on=["year", "month"], how="full", coalesce=True)
    n_missing = months.join(dep, on=["year", "month"]).select(
        pl.any_horizontal(pl.col(_DEP_COLUMNS).is_null()).sum()
    )
    for col in _DEP_COLUMNS:
        dep = _fill_from_nearest_years(dep, col)
    if n_missing.item():
        logger.info(
            "Filled %d month(s) of deposition for plot_id %s from the nearest %d years",
            n_missing.item(),
            plot_id,
            _DEP_FILL_YEARS,
        )

    return weather_df.join(dep, on=["year", "month"], how="left")


def load_plot_tables(plot_id: str, source: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read one plot's rows of the combined plot-level and weather tables.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Plot-level data and monthly weather, ready for `create_plot_input_file`.
    """
    plot_filter = (pl.col("plot_id") == plot_id) & (pl.col("source") == source)

    def read(name: str) -> pl.DataFrame:
        """Read this plot's rows of one combined table."""
        return pl.scan_parquet(os.path.join(clean_data_folder, name)).filter(plot_filter).collect()

    return read("trunx_plot_level_data.parquet"), read("trunx_plot_weather.parquet")


def build_plot_input_sheets(
    plot_id: str,
    source: str,
    plot_data: pl.DataFrame,
    weather_data: pl.DataFrame,
    literature_source: str = "Forrester",
    age_models: dict[str, tuple[float, float]] | None = None,
    deposition: pl.DataFrame | None = None,
    gpp: pl.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build the 3PG input sheets for one plot of the combined datasets.

    Mixed plots give one species row, `parameters` column and set of `param_bound`
    and `observed` rows per species; species without 3PG parameters are dropped.
    The simulation starts at the first survey inside the weather period where every
    recorded species has stem number, biomass and mean DBH (the initial state), and
    ends at the last survey inside it. Species first recorded later are dropped.

    Parameters
    ----------
    plot_id : str
        Plot identifier, as in `plot_data`.
    source : str
        Dataset of the plot: "NFI", "EFM", "LWF" or "ICP".
    plot_data : pl.DataFrame
        Combined plot-level table (`combined_data.get_combined_plot_data`).
    weather_data : pl.DataFrame
        Combined monthly weather (`combined_weather_data.build_tree_plot_weather`).
    literature_source : str
        Literature table the `param_bound` and `parameters` sheets are built from.
    age_models : dict[str, tuple[float, float]] | None
        Per-species age-vs-DBH models, used for the planting year when the plot
        has no measured stand age; fitted from ICP data when None. Pass them in
        when creating many plots.
    deposition : pl.DataFrame | None
        Pre-loaded monthly ICP deposition; loaded from disk when None.
    gpp : pl.DataFrame | None
        Pre-loaded monthly GOSIF GPP of ICP plots, added to the `all_observed` sheet
        (the `observed` sheet used for calibration has survey rows only); loaded from
        disk when None.

    Returns
    -------
    dict[str, pd.DataFrame]
        Sheet name to table, in 3PG input file order.
    """
    plot_df = _plot_observations(plot_data, plot_id, source)
    weather_df = _plot_weather(weather_data, plot_id, source)

    weather_months = weather_df.select(_month_index(pl.col("year"), pl.col("month")))
    first_month, last_month = weather_months.min().item(), weather_months.max().item()
    if weather_df.height != last_month - first_month + 1:
        raise ValueError(f"Weather for plot {plot_id} has missing months")

    survey_month = _month_index(pl.col("year"), pl.col("month"))
    surveys = plot_df.filter(survey_month.is_between(first_month, last_month))
    complete_dates = (
        surveys.group_by("date")
        .agg(
            pl.all_horizontal(pl.col(_FIRST_SURVEY_COLUMNS).is_not_null()).all().alias("complete")
        )
        .filter("complete")
    )
    if complete_dates.is_empty():
        raise ValueError(f"No survey of plot {plot_id} with an initial state inside the weather")
    start_date = complete_dates["date"].min()
    first_surveys = surveys.filter(pl.col("date") == start_date)
    species = first_surveys["specie"].to_list()
    late_species = sorted(set(surveys["specie"]) - set(species))
    if late_species:
        logger.warning(
            "Dropping species first recorded after the start of %s: %s", plot_id, late_species
        )
    surveys = surveys.filter((pl.col("date") >= start_date) & pl.col("specie").is_in(species))
    first_survey = first_surveys.row(0, named=True)
    last_survey = surveys.sort("date").row(-1, named=True)

    if age_models is None:
        age_models = fit_models()
    species_df = _species_sheet(first_surveys, first_survey["year"], age_models)
    site_df = _site_sheet(first_survey, last_survey)

    climate_df = weather_df.filter(
        _month_index(pl.col("year"), pl.col("month")).is_between(
            _month_index(pl.lit(first_survey["year"]), pl.lit(first_survey["month"])),
            _month_index(pl.lit(last_survey["year"]), pl.lit(last_survey["month"])),
        )
    )
    if source == "ICP":
        climate_df = add_deposition_to_weather(climate_df, plot_id, deposition=deposition)
    else:
        climate_df = climate_df.with_columns(_MISSING_DEPOSITION)

    param_bounds = {
        name: _load_species_param_bound(name, literature_source=literature_source)
        for name in species
    }
    parameters = pd.concat(
        [
            parameters_from_param_bound(bound, name).set_index("parameter")
            for name, bound in param_bounds.items()
        ],
        axis=1,
    ).reset_index()
    param_bound = pd.concat(
        [bound.assign(species=name) for name, bound in param_bounds.items()], ignore_index=True
    )
    error_param = pd.read_excel(
        os.path.join(threepg_data_folder, "solling_data.xlsx"), sheet_name="error_param"
    )
    observed_df = _observed_sheet(surveys)
    all_observed_df = _all_observed_sheet(
        observed_df, _plot_gpp(plot_id, source, gpp, first_survey, last_survey)
    )

    return {
        "climate": climate_df.to_pandas(),
        "parameters": parameters,
        "species": species_df.to_pandas(),
        "site": site_df.to_pandas(),
        "thinning": pd.DataFrame(),
        "sizeDist": pd.DataFrame(),
        "observed": observed_df.to_pandas(),
        "all_observed": all_observed_df.to_pandas(),
        "param_bound": param_bound,
        "error_param": error_param,
    }


def create_plot_input_file(
    plot_id: str,
    source: str,
    output_file: str,
    plot_data: pl.DataFrame,
    weather_data: pl.DataFrame,
    literature_source: str = "Forrester",
    age_models: dict[str, tuple[float, float]] | None = None,
    deposition: pl.DataFrame | None = None,
    gpp: pl.DataFrame | None = None,
) -> str:
    """Create a 3PG input Excel file for one plot of the combined datasets.

    The sheets are built by `build_plot_input_sheets`, see there for the parameters.

    Parameters
    ----------
    output_file : str
        Path of the Excel file to write.

    Returns
    -------
    str
        `output_file`.
    """
    sheets = build_plot_input_sheets(
        plot_id,
        source,
        plot_data,
        weather_data,
        literature_source=literature_source,
        age_models=age_models,
        deposition=deposition,
        gpp=gpp,
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for name, sheet in sheets.items():
            sheet.to_excel(writer, sheet_name=name, index=False)

    logger.info("Wrote 3PG input for %s plot %s to %s", source, plot_id, output_file)
    return output_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    literature_source = "Forrester"
    plot_data = pl.read_parquet(os.path.join(clean_data_folder, "trunx_plot_level_data.parquet"))
    weather_data = pl.read_parquet(os.path.join(clean_data_folder, "trunx_plot_weather.parquet"))
    age_models = fit_models()

    output_dir = os.path.join(threepg_data_folder, "combined_plots", literature_source)
    for plot_id, source in [("04.1402", "ICP")]:
        create_plot_input_file(
            plot_id,
            source,
            os.path.join(output_dir, f"{source}_{plot_id}_data.xlsx"),
            plot_data,
            weather_data,
            literature_source=literature_source,
            age_models=age_models,
        )
