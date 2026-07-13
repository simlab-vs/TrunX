"""Create data input for 3PG model using ICP data.

TODO
-------
Adjust the per-hectare scaling to account for the actual plot area
(currently assumed to be 1 ha). Following it adjust the biomass and LAI calculations
accordingly.

"""

import logging
import os

import jax.numpy as jnp
import pandas as pd
import polars as pl

from trunx.config import clean_data_folder, threepg_data_folder
from trunx.datasets.ICP_weather_data import prepare_icp_weather_data
from trunx.gp3.age_regression import fit_models, predict_age_from_dbh
from trunx.gp3.allometrics import add_allometric_columns, load_forrester_eq3

logger = logging.getLogger(__name__)

# Forrester et al. (2017) eq. 3 coefficients for all species with complete data.
# Loaded once at import time to avoid repeated Excel reads.
_FORRESTER_COEFFS = load_forrester_eq3()


def dms_to_decimal(dms):
    """Convert DMS packed as ±DDMMSS or ±DDDMMSS to decimal degrees."""
    sign = -1 if str(dms).startswith("-") else 1
    dms = abs(int(dms))

    degrees = dms // 10000
    minutes = (dms % 10000) // 100
    seconds = dms % 100

    val = sign * (degrees + minutes / 60 + seconds / 3600)
    return val


def create_weather_input(df, plot_id):
    """Create weather data input for 3PG model."""
    plot_df = df.filter(pl.col("plot_id") == plot_id)

    temp_df = plot_df.filter(pl.col("code_variable") == "AT")

    prcp_df = plot_df.filter(pl.col("code_variable") == "PR")

    srad_df = plot_df.filter(pl.col("code_variable") == "SR")
    # Convert from W/m² to MJ/m² (1 W/m² = 0.0864 MJ/m²)
    for col in srad_df.schema:
        if col.startswith("daily_m"):
            srad_df = srad_df.with_columns((pl.col(col) * 0.0864).alias(col))

    mtemp_df = (
        temp_df.group_by(["month_year"])
        .agg(
            pl.col("daily_mean").mean().alias("tmp_ave"),
            pl.col("daily_min").mean().alias("tmp_min"),
            pl.col("daily_max").mean().alias("tmp_max"),
            (pl.col("daily_min") < 0).sum().alias("frost_days"),
        )
        .with_columns(
            pl.when(pl.col("tmp_ave") < pl.col("tmp_min"))
            .then(pl.col("tmp_ave"))
            .otherwise(pl.col("tmp_min"))
            .alias("tmp_min"),
            pl.when(pl.col("tmp_ave") > pl.col("tmp_max"))
            .then(pl.col("tmp_ave"))
            .otherwise(pl.col("tmp_max"))
            .alias("tmp_max"),
        )
    )

    mprcp_df = prcp_df.group_by("month_year").agg(pl.col("daily_mean").sum().alias("prcp"))

    msrad_df = srad_df.group_by("month_year").agg(pl.col("daily_mean").mean().alias("srad"))

    empty_dfs = [
        name
        for name, df in [
            ("temperature", mtemp_df),
            ("precipitation", mprcp_df),
            ("solar radiation", msrad_df),
        ]
        if df.height == 0
    ]

    if empty_dfs:
        raise ValueError(
            f"Empty DataFrame(s) for plot_id {plot_id}: {', '.join(empty_dfs)}. "
            "Please check the input data."
        )

    weather_df = mtemp_df.join(mprcp_df, on="month_year").join(msrad_df, on="month_year")

    weather_df = (
        weather_df.with_columns(
            [pl.col("month_year").str.strptime(pl.Date, "%m-%Y").alias("month_year")]
        )
        .with_columns(
            [
                pl.col("month_year").dt.year().alias("year"),
                pl.col("month_year").dt.month().alias("month"),
            ]
        )
        .select(["year", "month", "tmp_ave", "tmp_min", "tmp_max", "frost_days", "prcp", "srad"])
        .sort(by=["year", "month"])
        .drop_nulls()
    )

    # Check for gaps
    weather_pl = weather_df.with_columns(pl.date(pl.col("year"), pl.col("month"), 1).alias("date"))
    min_date = weather_pl.select(pl.col("date").min()).item()
    max_date = weather_pl.select(pl.col("date").max()).item()
    all_months = pl.date_range(start=min_date, end=max_date, interval="1mo", eager=True)
    existing_months = set(weather_pl.select("date").to_series().to_list())
    miss_months = sorted(set(all_months) - existing_months)

    return miss_months, weather_df


def weather_summary(weather_df):
    """Summarize weather data."""
    summary_stats = {
        "Metric": [
            "Total records (months)",
            "Start date",
            "End date",
            "Expected months (based on date range)",
            "Actual months present",
            "Missing months",
            "Completeness (%)",
            "Has gaps",
            "Number of gaps",
            "Largest gap (months)",
            "Average gap size (months)",
            "Unique years",
            "Unique months",
        ],
        "Value": [],
    }

    # Calculate basic metrics
    total_records = weather_df.height
    start_date = (
        weather_df.select(pl.col("year").min()).item(),
        weather_df.select(pl.col("month").min()).item(),
    )
    end_date = (
        weather_df.select(pl.col("year").max()).item(),
        weather_df.select(pl.col("month").max()).item(),
    )

    # Calculate expected months if continuous
    start_year, start_month = start_date
    end_year, end_month = end_date
    expected_months = (end_year - start_year) * 12 + (end_month - start_month) + 1

    # Check for gaps
    weather_pl = weather_df.with_columns(pl.date(pl.col("year"), pl.col("month"), 1).alias("date"))
    min_date = weather_pl.select(pl.col("date").min()).item()
    max_date = weather_pl.select(pl.col("date").max()).item()
    all_months = pl.date_range(start=min_date, end=max_date, interval="1mo", eager=True)
    existing_months = set(weather_pl.select("date").to_series().to_list())
    miss_months = sorted(set(all_months) - existing_months)

    # Calculate gap statistics
    gaps = []
    if miss_months:
        # Find consecutive missing months (gaps)
        missing_series = pl.Series(miss_months)
        if len(missing_series) > 0:
            diff = missing_series.diff().dt.total_days()
            gap_starts = missing_series.filter((diff.is_null()) | (diff > 31)).to_list()
            gap_lengths = []

            if len(gap_starts) > 0:
                for i, gap_start in enumerate(gap_starts):
                    if i < len(gap_starts) - 1:
                        gap_end = gap_starts[i + 1]
                        gap_months = len(
                            missing_series.filter(
                                (missing_series >= gap_start) & (missing_series < gap_end)
                            )
                        )
                    else:
                        gap_months = len(missing_series.filter(missing_series >= gap_start))
                    gap_lengths.append(gap_months)
            else:
                gap_lengths = [len(missing_series)]

            gaps = gap_lengths
        else:
            gaps = [len(miss_months)]

    # Count years with complete data
    weather_df = weather_df.with_columns(
        (pl.col("year").cast(pl.Utf8) + "-" + pl.col("month").cast(pl.Utf8).str.zfill(2)).alias(
            "year_month"
        )
    )
    # Populate summary values
    summary_stats["Value"].extend(
        [
            total_records,
            f"{start_date[0]}-{start_date[1]:02d}",
            f"{end_date[0]}-{end_date[1]:02d}",
            expected_months,
            total_records,
            len(miss_months),
            f"{(total_records / expected_months * 100):.1f}%",
            "Yes" if miss_months else "No",
            len(gaps),
            max(gaps) if gaps else 0,
            f"{sum(gaps) / len(gaps):.1f}" if gaps else "N/A",
            weather_df.select(pl.col("year").n_unique()).item(),
            weather_df.select(pl.col("month").n_unique()).item(),
        ]
    )

    if len(miss_months) > 0:
        print("The data is not complete. \n")

    summary_df = pd.DataFrame(summary_stats)

    # Detailed missing months analysis

    print("MISSING MONTHS ANALYSIS")

    if miss_months:
        # Group missing months by year
        missing_df = pl.DataFrame({"missing_date": miss_months})
        missing_df = missing_df.with_columns(
            [
                pl.col("missing_date").dt.year().alias("year"),
                pl.col("missing_date").dt.month().alias("month"),
            ]
        )
        missing_by_year = missing_df.group_by("year").agg(pl.len()).sort("year")

        print("\nMissing months by year:")
        print(missing_by_year)

        print("\nFirst 10 missing months:")
        for month in miss_months[:10]:
            print(f"  - {month.strftime('%B %Y')}")

        if len(miss_months) > 10:
            print(f"  ... and {len(miss_months) - 10} more")

        # Check if missing months are at the beginning or end
        if miss_months[0] == all_months[0]:
            print("\n Missing data at the beginning of the time series")
        if miss_months[-1] == all_months[-1]:
            print("\n Missing data at the end of the time series")
    else:
        print("\n No missing months detected. The dataset is perfectly continuous!")

    return summary_df


def create_input_params(icp_df):
    """Create parameter data input for 3PG model."""
    input_params = {}
    params_df = pl.read_excel(
        os.path.join(threepg_data_folder, "data.input.xlsx"), sheet_name="parameters"
    )
    input_params["parameter"] = params_df["parameter"].to_list()
    for specie in icp_df["specie"].unique().to_list():
        if specie in params_df.schema:
            input_params[specie] = params_df[specie]

    input_params_df = pl.DataFrame(input_params)

    if len(input_params_df.schema) < 2:
        raise ValueError("No parameter values found for species in this location")

    return input_params_df


def create_species_data(icp_df):
    """Create species data input for 3PG model."""
    species_df = pl.read_excel(
        os.path.join(threepg_data_folder, "data.input.xlsx"), sheet_name="species"
    )

    print(
        "\n Species found in this location: ",
        icp_df.select("specie").unique().to_series().to_list(),
    )
    species_df = species_df.filter(
        pl.col("species").is_in(icp_df.select("specie").unique().to_series().to_list())
    )

    return species_df


def create_site_data(icp_df, weather_df):
    """Create site data input for 3PG model."""
    site_df = pl.read_excel(
        os.path.join(threepg_data_folder, "data.input.xlsx"), sheet_name="site"
    )

    latitude_value = icp_df["Lat"][0]
    altitude_value = icp_df["plot_altitude"][0]

    # Add as constant columns
    site_df = site_df.with_columns(
        [pl.lit(latitude_value).alias("latitude"), pl.lit(altitude_value).alias("altitude")]
    )

    min_year = weather_df["year"].min()
    min_month = weather_df.filter(pl.col("year") == min_year).select("month").min().item()
    site_df = site_df.with_columns([pl.lit(f"{min_year}-{min_month:02d}").alias("from")])

    max_year = weather_df["year"].max()
    max_month = weather_df.filter(pl.col("year") == max_year).select("month").max().item()
    site_df = site_df.with_columns([pl.lit(f"{max_year}-{max_month:02d}").alias("to")])

    return site_df


def update_species_data(
    icp_df: pl.DataFrame,
    species_df: pl.DataFrame,
    input_params_df: pl.DataFrame,
    models: dict[str, tuple[float, float]] | None = None,
) -> tuple[pl.DataFrame, int]:
    """Create species data for 3PG model for a specific site.

    Parameters
    ----------
    icp_df : pl.DataFrame
        ICP level2 data filtered to the target plot.
    species_df : pl.DataFrame
        Species template data from the 3PG input file.
    input_params_df : pl.DataFrame
        Parameter data for the species present.
    models : dict[str, tuple[float, float]] | None
        Per-species power-law coefficients ``(a, b)`` from
        :func:`trunx.gp3.age_regression.fit_models`. When provided, the
        ``planted`` date is estimated from DBH at the first common survey
        year rather than taken from the template.

    Returns
    -------
    tuple[pl.DataFrame, int]
        Updated species DataFrame and the first common survey year.
    """
    avg_diameter = icp_df.group_by(["specie", "period_end"]).agg(
        pl.col("diameter_end").mean().alias("DBH")
    )
    avg_diameter = avg_diameter.with_columns(pl.col("period_end").dt.year().alias("year"))
    avg_diameter = avg_diameter.filter(
        pl.col("specie").is_in(species_df.select("species").to_series().to_list())
    )

    common_start = (
        avg_diameter.group_by("specie")
        .agg([pl.col("year").min().alias("min_year")])
        .select([pl.col("min_year").max().alias("start_year")])
        .sort(["start_year"])
    )

    start_year = common_start["start_year"].item()

    avg_diameter = avg_diameter.filter(pl.col("year") == start_year)

    if models is not None:
        for specie in avg_diameter["specie"].unique().to_list():
            _raw_age = (
                icp_df.filter(
                    (pl.col("period_end").dt.year() == start_year) & (pl.col("specie") == specie)
                )
                .select(pl.col("soph_avg_age").cast(pl.Float64).mean())
                .item()
            )
            direct_age: float | None = float(_raw_age) if _raw_age is not None else None

            if direct_age is not None:
                planting_year = int(start_year - round(direct_age))
                species_df = species_df.with_columns(
                    pl.when(pl.col("species") == specie)
                    .then(pl.lit(f"{planting_year}-01"))
                    .otherwise(pl.col("planted"))
                    .alias("planted")
                )
                logger.info(
                    "Direct age for '%s': %d (age %.1f yr at %d)",
                    specie,
                    planting_year,
                    direct_age,
                    start_year,
                )
                continue

            if specie not in models:
                logger.warning("No age model for '%s', keeping template planted date", specie)
                continue
            dbh = avg_diameter.filter(pl.col("specie") == specie)["DBH"].to_numpy()
            a, b = models[specie]
            estimated_age = predict_age_from_dbh(dbh, a, b)
            planting_year = int(start_year - round(estimated_age))
            species_df = species_df.with_columns(
                pl.when(pl.col("species") == specie)
                .then(pl.lit(f"{planting_year}-01"))
                .otherwise(pl.col("planted"))
                .alias("planted")
            )
            logger.info(
                "Estimated planting year for '%s': %d (age %.1f yr at %d)",
                specie,
                planting_year,
                estimated_age,
                start_year,
            )

    num_trees = (
        icp_df.filter(pl.col("specie").is_in(species_df.select("species").to_series().to_list()))
        .group_by(["specie", "period_end"])
        .agg(pl.len().alias("num_trees"))
        .with_columns(pl.col("period_end").dt.year().alias("year"))
        .filter(pl.col("year") == start_year)
        .rename({"specie": "species"})
    )

    species_df = (
        species_df.join(num_trees, on=["species"], how="inner")
        .select(
            [
                "species",
                "planted",
                "fertility",
                "num_trees",
                "biom_stem",
                "biom_root",
                "biom_foliage",
            ]
        )
        .rename({"num_trees": "stems_n"})
    )

    sp_inp_params = input_params_df.filter(
        pl.col("parameter").is_in(["aWS", "nWS", "pFS2", "pFS20", "pRn", "pRx"])
    )

    params_lookup = {}
    for species in sp_inp_params.columns:
        if species != "parameter":
            aWS = sp_inp_params.filter(pl.col("parameter") == "aWS").select(species).item()
            nWS = sp_inp_params.filter(pl.col("parameter") == "nWS").select(species).item()
            pFS2 = sp_inp_params.filter(pl.col("parameter") == "pFS2").select(species).item()
            pFS20 = sp_inp_params.filter(pl.col("parameter") == "pFS20").select(species).item()
            pRn = sp_inp_params.filter(pl.col("parameter") == "pRn").select(species).item()
            pRx = sp_inp_params.filter(pl.col("parameter") == "pRx").select(species).item()

            params_lookup[species] = {
                "aWS": aWS,
                "nWS": nWS,
                "pFS2": pFS2,
                "pFS20": pFS20,
                "pRn": pRn,
                "pRx": pRx,
            }

    # Create a lookup for DBH values
    dbh_lookup = {}
    for row in avg_diameter.iter_rows():
        species = row[0]
        dbh = row[2]
        dbh_lookup[species] = dbh

    # Update biom_stem for each species in species_df
    for species in species_df["species"].unique().to_list():
        if species in params_lookup and species in dbh_lookup:
            aWS = params_lookup[species]["aWS"]
            nWS = params_lookup[species]["nWS"]
            dbh = dbh_lookup[species]
            pFS2 = params_lookup[species]["pFS2"]
            pFS20 = params_lookup[species]["pFS20"]
            pRn = params_lookup[species]["pRn"]
            pRx = params_lookup[species]["pRx"]
            stems_n = species_df.filter(pl.col("species") == species).select("stems_n").item()

            calculated_biom_stem = (aWS * (dbh**nWS) * stems_n) / 1000.0
            pfsPower = jnp.log(pFS20 / (pFS2 + 1e-8)) / jnp.log(10.0)
            pfsConst = pFS2 / 2.0**pfsPower
            pFS = pfsConst * (dbh**pfsPower)
            calculated_biom_foliage = calculated_biom_stem * pFS
            pRS = pRn + (pRx - pRn) * 0.5  # Assuming water stress factor of 0.5 for simplicity,
            # this can be adjusted based on actual conditions
            calculated_biom_root = (calculated_biom_stem + calculated_biom_foliage) * pRS

            species_df = (
                species_df.with_columns(
                    pl.when(pl.col("species") == species)
                    .then(pl.lit(calculated_biom_stem))
                    .otherwise(pl.col("biom_stem"))
                    .alias("biom_stem")
                )
                .with_columns(
                    pl.when(pl.col("species") == species)
                    .then(pl.lit(calculated_biom_foliage))
                    .otherwise(pl.col("biom_foliage"))
                    .alias("biom_foliage")
                )
                .with_columns(
                    pl.when(pl.col("species") == species)
                    .then(pl.lit(calculated_biom_root))
                    .otherwise(pl.col("biom_root"))
                    .alias("biom_root")
                )
            )

    return species_df, start_year


def _allometric_observations(icp_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-tree Forrester biomass to survey-date observations.

    Parameters
    ----------
    icp_df : pl.DataFrame
        Tree-level ICP data with ``diameter_end`` (cm) and ``specie`` columns.

    Returns
    -------
    pl.DataFrame
        One row per (specie, month_year) with columns ``WS``, ``WF``, ``WR``
        (t ha⁻¹) and ``LAI`` (m² m⁻²).  Rows whose species has no Forrester
        equation 3 coefficients carry null values.

    Notes
    -----
    Per-hectare scaling assumes the ICP plot represents 1 ha.  WS/WF/WR are
    the sum of per-tree biomass (kg) divided by 1 000.  LAI is the sum of
    per-tree leaf area (m²) divided by 10 000.
    """
    return (
        add_allometric_columns(
            icp_df,
            _FORRESTER_COEFFS,
            dbh_col="diameter_end",
            species_col="specie",
        )
        .group_by(["specie", "period_end"])
        .agg(
            (pl.col("allo_sb_kg").sum() / 1000.0).alias("WS"),
            (pl.col("allo_fb_kg").sum() / 1000.0).alias("WF"),
            (pl.col("allo_rb_kg").sum() / 1000.0).alias("WR"),
            (pl.col("allo_la_m2").sum() / 10000.0).alias("LAI"),
        )
        .with_columns(
            pl.col("period_end").dt.strftime("%m-%Y").alias("month_year"),
        )
        .drop("period_end")
    )


def create_observation_data(plot_id, icp_df, weather_df, start_year):
    """Create observed data input for 3PG model."""
    dbh_df = (
        icp_df.group_by(["specie", "period_end"])
        .agg(pl.col("diameter_end").mean().alias("DBH"))
        .sort("period_end")
    ).with_columns(
        pl.col("period_end").dt.year().cast(pl.Int64).alias("year"),
        pl.col("period_end").dt.month().cast(pl.Int64).alias("month"),
        pl.col("period_end").dt.strftime("%m-%Y").alias("month_year"),
    )

    gpp_df = pl.read_csv(os.path.join(clean_data_folder, "GOSIF_GPP_icp.csv"))

    gpp_df = gpp_df.filter(pl.col("plot_id") == float(plot_id))

    gpp_df = (
        gpp_df.with_columns(pl.datetime(pl.col("year"), pl.col("month"), 1).alias("Date"))
        .sort(["year", "month"])
        .with_columns(pl.col("Date").dt.month_end())
        .with_columns(pl.col("Date").dt.strftime("%m-%Y").alias("month_year"))
        .drop(["plot_id"])
        .filter(pl.col("Date").dt.year() >= start_year)
    )

    specie_gpp = []
    for specie in dbh_df["specie"].unique():
        tdf = gpp_df.with_columns(pl.lit(specie).alias("specie"))
        specie_gpp.append(tdf)

    specie_gpp = pl.concat(specie_gpp)

    biom_df = _allometric_observations(icp_df)

    observed_data = (
        specie_gpp.join(dbh_df, on=["specie", "month_year"], how="full")
        .join(biom_df, on=["specie", "month_year"], how="left")
        .select(["specie", "month", "year", "Date", "GPP", "DBH", "WS", "WF", "WR", "LAI"])
        .join(weather_df, on=["month", "year"], how="full")
        .select("specie", "month", "year", "Date", "GPP", "DBH", "WS", "WF", "WR", "LAI")
    )

    return observed_data


def create_input_data(input_data_file, plot_id):
    """Create input data for 3PG model."""
    # ICP weather data
    # raw_file_path = os.path.join(icp_raw_data_folder, "595_mm_20260227091917/mm_mem.csv")
    # processor = prepare_icp_weather_data(raw_file_path)
    # df = processor.clean_data()

    df = pl.read_parquet(os.path.join(clean_data_folder, "ICP_weather_data.parquet"))

    # ICP data — load full dataset first for cross-plot age model fitting
    icp_full = pl.read_parquet(os.path.join(clean_data_folder, "icp_level2_cleaned.parquet"))
    age_models = fit_models(icp_full)
    icp_df = icp_full.filter(pl.col("plot_id") == plot_id)
    icp_df = icp_df.filter(
        pl.col("specie").is_in(
            list(
                [
                    "Picea abies",
                    "Pinus sylvestris",
                    "Fagus sylvatica",
                    "Quercus robur",
                    "Quercus petraea",
                ]
            )
        )
    )

    icp_df = icp_df.with_columns(
        pl.col("plot_latitude").map_elements(dms_to_decimal, return_dtype=pl.Float64).alias("Lat"),
        pl.col("plot_longitude")
        .map_elements(dms_to_decimal, return_dtype=pl.Float64)
        .alias("Lon"),
    )

    logging.info("Pre-processed ICP data for plot_id: %s", plot_id)
    # Weather data
    miss_months, weather_df = create_weather_input(df, plot_id=plot_id)
    logging.info("Pre-processed weather data for plot_id: %s", plot_id)

    # Species data
    input_species_df = create_species_data(icp_df)
    logging.info("Created species data for plot_id: %s", plot_id)

    # Parameter data
    input_params_df = create_input_params(icp_df)
    logging.info("Created parameter data for plot_id: %s", plot_id)

    input_species_df, start_year = update_species_data(
        icp_df, input_species_df, input_params_df, models=age_models
    )

    weather_df = weather_df.filter(pl.col("year") >= start_year)

    # Site data
    input_site_df = create_site_data(icp_df, weather_df)

    icp_df = icp_df.filter(pl.col("specie").is_in(input_species_df["species"]))

    observed_data = create_observation_data(plot_id, icp_df, weather_df, start_year)

    if len(miss_months) == 0:
        with pd.ExcelWriter(input_data_file, engine="openpyxl") as writer:
            weather_df.to_pandas().to_excel(writer, sheet_name="climate", index=False)
            input_params_df.to_pandas().to_excel(writer, sheet_name="parameters", index=False)
            input_species_df.to_pandas().to_excel(writer, sheet_name="species", index=False)
            input_site_df.to_pandas().to_excel(writer, sheet_name="site", index=False)
            pd.DataFrame().to_excel(writer, sheet_name="thinning", index=False)
            pd.DataFrame().to_excel(writer, sheet_name="sizeDist", index=False)
            observed_data.to_pandas().to_excel(writer, sheet_name="observed", index=False)
    else:
        print("The weather data is not complete and need processing to fill missing data.")
        summary_wdf = weather_summary(weather_df)
        print(summary_wdf)

    return miss_months, observed_data.to_pandas()
