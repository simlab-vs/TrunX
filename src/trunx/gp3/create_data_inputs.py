"""Create data input for 3PG model using ICP data."""

import os

import pandas as pd
import polars as pl

from trunx.config import icp_raw_data_folder
from trunx.datasets.ICP_weather_data import prepare_icp_weather_data


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

    mtemp_df = temp_df.group_by(["month_year"]).agg(
        pl.col("daily_mean").mean().alias("tmp_ave"),
        pl.col("daily_min").mean().alias("tmp_min"),
        pl.col("daily_max").mean().alias("tmp_max"),
        (pl.col("daily_min") < 0).sum().alias("frost_days"),
    )

    mprcp_df = prcp_df.group_by("month_year").agg(pl.col("daily_mean").sum().alias("prcp"))

    msrad_df = srad_df.group_by("month_year").agg(pl.col("daily_mean").mean().alias("srad"))

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
    params_df = pl.read_excel("./data/data.input.xlsx", sheet_name="parameters")
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
    species_df = pl.read_excel("./data/data.input.xlsx", sheet_name="species")
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
    site_df = pl.read_excel("./data/data.input.xlsx", sheet_name="site")

    latitude_value = icp_df["Lat"][0]  # or use .mean(), .first(), etc.
    altitude_value = icp_df["plot_altitude"][0]

    # Add as constant columns
    site_df = site_df.with_columns(
        [pl.lit(latitude_value).alias("latitude"), pl.lit(altitude_value).alias("altitude")]
    )

    min_year = weather_df["year"].min()
    min_month = weather_df["month"].min()
    site_df = site_df.with_columns([pl.lit(f"{min_year}-{min_month:02d}").alias("from")])

    max_year = weather_df["year"].max()
    max_month = weather_df["month"].max()
    site_df = site_df.with_columns([pl.lit(f"{max_year}-{max_month:02d}").alias("to")])

    return site_df


def create_input_data(input_data_file, plot_id):
    """Create input data for 3PG model."""
    # ICP weather data
    raw_file_path = os.path.join(icp_raw_data_folder, "595_mm_20260227091917/mm_mem.csv")
    processor = prepare_icp_weather_data(raw_file_path)
    df = processor.clean_data()

    # ICP data
    icp_df = pl.read_parquet("./data/clean/icp_level2_cleaned.parquet")
    icp_df = icp_df.filter(pl.col("plot_id") == plot_id)
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

    # Weather data
    miss_months, weather_df = create_weather_input(df, plot_id=plot_id)

    # Species data
    input_species_df = create_species_data(icp_df)

    # Parameter data
    input_params_df = create_input_params(icp_df)

    # Site data
    input_site_df = create_site_data(icp_df, weather_df)

    icp_df = icp_df.filter(pl.col("specie").is_in(input_species_df["species"]))
    observed_data = icp_df.group_by(["specie", "period_end"]).agg(
        pl.col("diameter_end").mean().alias("DBH")
    )

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
