"""Geographic locations of ICP plots."""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import polars as pl

    sys.path.append(str(Path(__file__).parent.parent))
    from scripts.support_utils import load_prepare_data
    from trunx.config import icp_raw_data_folder
    from trunx.datasets.ICP_weather_data import prepare_icp_weather_data
    from trunx.plot_utils import plot_geographic_location_species

    return (
        icp_raw_data_folder,
        load_prepare_data,
        os,
        pl,
        plot_geographic_location_species,
        prepare_icp_weather_data,
    )


@app.cell
def _(load_prepare_data, pl, plot_geographic_location_species):
    _, df = load_prepare_data()

    # Single species locations
    df = df.join(
        df.group_by(["Lat", "Lon"]).agg(species_count=pl.col("Species").unique().count()),
        on=["Lat", "Lon"],
    ).filter(pl.col("species_count") == 1)

    plot_geographic_location_species(df, selected_species=["Spruce", "Beech", "Pine", "Oak"])
    return (df,)


@app.cell
def _(df, pl):
    df.group_by(["code_plot", "Lat", "Lon", "plot_id"], maintain_order=True).agg(
        num_trees=pl.col("tree_id").n_unique(),
        specie=pl.col("specie").unique(),
        age=pl.col("soph_avg_age").unique(),
    )
    return


@app.cell
def _(df, pl):
    # Plot ids with maximum number of growth periods
    max_number_periods_df = (
        df.group_by(["code_plot", "Lat", "Lon", "plot_id", "tree_id"], maintain_order=True)
        .agg(
            num_period_group=pl.len(),
            specie=pl.col("specie").unique(),
            age=pl.col("soph_avg_age").unique(),
        )
        .filter(pl.col("num_period_group") == pl.col("num_period_group").max())
        .group_by(["code_plot", "Lat", "Lon", "plot_id"])
        .agg(
            num_trees=pl.col("tree_id").unique().len(),
            species_list=pl.col("specie").unique(),
            age_values=pl.col("age").unique(),
        )
        .sort("num_trees", descending=True)
    )

    max_number_periods_plot_ids = sorted(max_number_periods_df["plot_id"].to_list())
    print("Plot ids with maximum number of growth periods are: \n", max_number_periods_plot_ids)

    print(max_number_periods_df)
    return


@app.cell
def _(icp_raw_data_folder, os, prepare_icp_weather_data):

    raw_file_path = os.path.join(icp_raw_data_folder, "595_mm_20260227091917/mm_mem.csv")
    processor = prepare_icp_weather_data(raw_file_path)
    clean_wdf = processor.clean_data()
    return (clean_wdf,)


@app.cell
def _(clean_wdf, df, pl):
    import matplotlib.pyplot as plt

    def check_missing_weather(df, plot_id, metric):
        plot_df = df.filter(pl.col("plot_id") == plot_id)
        metric_df = plot_df.filter(pl.col("code_variable") == metric)
        mavg_df = metric_df.group_by("month_year").agg(pl.col("daily_mean").mean().alias("avg"))
        mavg_df = (
            mavg_df.with_columns(
                [pl.col("month_year").str.strptime(pl.Date, "%m-%Y").alias("month_year")]
            )
            .with_columns(
                [
                    pl.col("month_year").dt.year().alias("year"),
                    pl.col("month_year").dt.month().alias("month"),
                ]
            )
            .select(["year", "month", "avg"])
            .sort(by=["year", "month"])
            .drop_nulls()
        )

        metric_pl = mavg_df.with_columns(pl.date(pl.col("year"), pl.col("month"), 1).alias("date"))

        if metric_pl.height > 0:
            min_date = metric_pl.select(pl.col("date").min()).item()
            max_date = metric_pl.select(pl.col("date").max()).item()
            all_months = pl.date_range(start=min_date, end=max_date, interval="1mo", eager=True)
            existing_months = set(metric_pl.select("date").to_series().to_list())
            miss_months = len(sorted(set(all_months) - existing_months)) / len(all_months)
        else:
            miss_months = -0.25

        return miss_months

    num_missing_months = {}

    for plot_id in df["plot_id"].unique().sort():
        miss_months = check_missing_weather(clean_wdf, plot_id, "AT")
        num_missing_months[plot_id] = miss_months

    missing_df = pl.DataFrame(
        {
            "plot_id": list(num_missing_months.keys()),
            "missing_months": list(num_missing_months.values()),
        }
    ).sort("plot_id")
    return missing_df, plt


@app.cell
def _(missing_df, plt):
    # Create bar plot
    fig, ax = plt.subplots(figsize=(45, 8))

    # Bar plot (since plot_ids are categorical)
    ax.bar(
        missing_df["plot_id"].to_list(),
        missing_df["missing_months"].to_list(),
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )

    ax.set_xlabel("Plot ID", fontsize=12)
    ax.set_ylabel("Percentage of Missing Months", fontsize=8)
    ax.margins(x=0)
    ax.grid(True, alpha=0.3, axis="y")

    # Rotate x-axis labels if many plots
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("./images/missing_weather_data.png")
    plt.show()
    return


@app.cell
def _(mo):

    # Complete country mapping with integer codes (based on your data)
    country_codes = {
        # Western Europe
        "🇫🇷 France (FR, 1)": 1,
        "🇧🇪 Belgium (BE, 2)": 2,
        "🇳🇱 Netherlands (NL, 3)": 3,
        "🇩🇪 Germany (DE, 4)": 4,
        "🇮🇹 Italy (IT, 5)": 5,
        "🇬🇧 United Kingdom (UK, 6)": 6,
        "🇮🇪 Ireland (IE, 7)": 7,
        "🇩🇰 Denmark (DK, 8)": 8,
        "🇬🇷 Greece (GR, 9)": 9,
        "🇵🇹 Portugal (PT, 10)": 10,
        "🇪🇸 Spain (ES, 11)": 11,
        "🇱🇺 Luxembourg (LU, 12)": 12,
        "🇸🇪 Sweden (SE, 13)": 13,
        "🇦🇹 Austria (AT, 14)": 14,
        "🇫🇮 Finland (FI, 15)": 15,
        # Central & Eastern Europe
        "🇨🇭 Switzerland (CH, 50)": 50,
        "🇭🇺 Hungary (HU, 51)": 51,
        "🇷🇴 Romania (RO, 52)": 52,
        "🇵🇱 Poland (PL, 53)": 53,
        "🇸🇰 Slovak Republic (SK, 54)": 54,
        "🇳🇴 Norway (NO, 55)": 55,
        "🇱🇹 Lithuania (LT, 56)": 56,
        "🇭🇷 Croatia (HR, 57)": 57,
        "🇨🇿 Czech Republic (CZ, 58)": 58,
        "🇪🇪 Estonia (EE, 59)": 59,
        "🇸🇮 Slovenia (SI, 60)": 60,
        "🇲🇩 Republic of Moldova (MD, 61)": 61,
        "🇷🇺 Russia (RU, 62)": 62,
        "🇧🇬 Bulgaria (BG, 63)": 63,
        "🇱🇻 Latvia (LV, 64)": 64,
        "🇧🇾 Belarus (BY, 65)": 65,
        "🇨🇾 Cyprus (CY, 66)": 66,
        "🇷🇸 Serbia (RS, 67)": 67,
        "🇦🇩 Andorra (AD, 68)": 68,
        "🇹🇷 Türkiye (TR, 72)": 72,
        "🇲🇪 Montenegro (ME, 80)": 80,
    }

    # Country selector
    country_ui = mo.ui.dropdown(
        options=country_codes.keys(), label="🌍 Select Country", value=None
    )
    return country_codes, country_ui


@app.cell
def _(country_codes, country_ui, df, mo, pl):
    # Get integer code
    selected_country_code = country_codes.get(country_ui.value) if country_ui.value else None

    options_list = (
        df.filter(pl.col("code_country") == selected_country_code)["plot_id"]
        .unique()
        .sort()
        .to_list()
        if selected_country_code is not None
        else df["plot_id"].unique().sort().to_list()
    )
    plot_selector_ui = mo.ui.dropdown(
        options=options_list, label="🌲 Select ICP Forest Plot", value=options_list[-1]
    )

    mo.hstack([country_ui, plot_selector_ui])
    return (plot_selector_ui,)


@app.cell
def _(clean_wdf, plot_selector_ui):
    from trunx.gp3.plot_function import plot_weather_data

    plot_weather_data(clean_wdf, plot_selector_ui.value)
    return


@app.cell
def _(df, pl, plot_selector_ui):
    # Number of growth periods
    df.filter(pl.col("plot_id") == plot_selector_ui.value).group_by("period_end").agg(
        avg_diameter=pl.col("diameter_end").mean(),
        num_tress=pl.len(),
        age=pl.col("soph_avg_age").mean(),
    )
    return


if __name__ == "__main__":
    app.run()
