"""Process ICP weather data."""

from typing import Optional, Union

import pandas as pd
import polars as pl


class prepare_icp_weather_data:
    """Prepare ICP weather data."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df_cleaned: pl.DataFrame | None = None

    def clean_data(self) -> pl.DataFrame:
        """Read and clean ICP daily weather data."""
        df = pl.read_csv(self.file_path, separator=";")

        # Cast numeric columns
        df = df.with_columns(
            [
                pl.col("daily_min").cast(pl.Float64),
                pl.col("daily_mean").cast(pl.Float64),
                pl.col("daily_max").cast(pl.Float64),
            ]
        )

        # Correct inconsistencies
        df = df.with_columns(
            [
                pl.when(pl.col("daily_mean") < pl.col("daily_min"))
                .then(None)
                .otherwise(pl.col("daily_min"))
                .alias("daily_min"),
                pl.when(pl.col("daily_mean") > pl.col("daily_max"))
                .then(None)
                .otherwise(pl.col("daily_max"))
                .alias("daily_max"),
            ]
        )

        # Create plot_id and parse date
        df = (
            df.with_columns(
                (
                    pl.col("code_country").cast(pl.Utf8).str.zfill(2)
                    + "."
                    + pl.col("code_plot").cast(pl.Utf8).str.zfill(4)
                ).alias("plot_id")
            )
            .with_columns(
                pl.col("date_observation")
                .str.strptime(pl.Date, "%Y-%m-%d")
                .alias("date_observation")
            )
            .with_columns(
                [
                    pl.col("date_observation").dt.year().alias("year"),
                    pl.col("date_observation").dt.month().alias("month"),
                ]
            )
            .with_columns(pl.col("date_observation").dt.strftime("%m-%Y").alias("month_year"))
            .filter(pl.col("year") > 1960)
        )

        # Aggregate duplicate daily records
        group_cols = [
            "code_country",
            "code_plot",
            "code_variable",
            "date_observation",
            "plot_id",
            "month_year",
        ]
        df = (
            df.group_by(group_cols)
            .agg([pl.all().exclude(group_cols).mean()])
            .drop(
                [
                    "code_data_origin",
                    "code_data_status",
                    "other_obs",
                    "q_flag",
                    "change_date",
                    "code_line",
                    "line_nr",
                ]
            )
        )

        self.df_cleaned = df
        return df

    def aggregate_monthly_variables(
        self, code_variables: list[str], plot_ids: None | str | list[str] = None
    ) -> pl.DataFrame:
        """Aggregate multiple daily variables into monthly summaries per plot."""
        if self.df_cleaned is None:
            self.df_cleaned = self.clean_data()

        df = self.df_cleaned

        # Aggregation rules
        rules = {
            "PR": {"mean": False, "sum": True, "min": False, "max": False},
            "AT": {"mean": True, "sum": False, "min": True, "max": True, "frost": True},
            "RH": {"mean": True, "sum": False, "min": True, "max": True},
            "WS": {"mean": True, "sum": False, "min": False, "max": True},
            "WD": {"mean": True, "sum": False, "min": False, "max": False},
            "SR": {"mean": True, "sum": False, "min": False, "max": False},
        }

        unknown_vars = [v for v in code_variables if v not in rules]
        if unknown_vars:
            raise ValueError(f"Unknown code_variable(s): {unknown_vars}")

        # Filter by plots
        if plot_ids is not None:
            if isinstance(plot_ids, str):
                df = df.filter(pl.col("plot_id") == plot_ids)
            elif isinstance(plot_ids, list):
                df = df.filter(pl.col("plot_id").is_in(plot_ids))
            else:
                raise ValueError("plot_ids must be None, str, or list of str")

        # Filter by variables
        df = df.filter(pl.col("code_variable").is_in(code_variables))

        agg_exprs = []
        for var in code_variables:
            rule = rules[var]
            if rule.get("sum"):
                agg_exprs.append(
                    pl.when(pl.col("code_variable") == var)
                    .then(pl.col("daily_mean"))
                    .sum()
                    .alias(f"{var.lower()}_sum")
                )
            if rule.get("mean"):
                agg_exprs.append(
                    pl.when(pl.col("code_variable") == var)
                    .then(pl.col("daily_mean"))
                    .mean()
                    .alias(f"{var.lower()}_mean")
                )
            if rule.get("min"):
                agg_exprs.append(
                    pl.when(pl.col("code_variable") == var)
                    .then(pl.col("daily_min"))
                    .min()
                    .alias(f"{var.lower()}_min")
                )
            if rule.get("max"):
                agg_exprs.append(
                    pl.when(pl.col("code_variable") == var)
                    .then(pl.col("daily_max"))
                    .max()
                    .alias(f"{var.lower()}_max")
                )
            if rule.get("frost"):
                agg_exprs.append(
                    pl.when(pl.col("code_variable") == var)
                    .then((pl.col("daily_min") < 0).cast(pl.Int64))
                    .sum()
                    .alias("frost_days")
                )

        group_cols = ["month_year", "plot_id", "code_country", "code_plot"]
        result = (
            df.group_by(group_cols)
            .agg(agg_exprs)
            .with_columns(pl.col("month_year").str.strptime(pl.Date, "%m-%Y").alias("month_year"))
            .with_columns(
                [
                    pl.col("month_year").dt.year().alias("year"),
                    pl.col("month_year").dt.month().alias("month"),
                ]
            )
            .sort(["code_country", "code_plot", "plot_id", "year", "month"])
        )

        return result

    def compute_plot_temporal_coverage(self) -> pl.DataFrame:
        """Compute monthly summaries and temporal coverage per plot."""
        if self.df_cleaned is None:
            raise ValueError("Data not cleaned. Run clean_data() first.")

        df = self.df_cleaned

        # Monthly summary
        monthly_summary = (
            df.with_columns(
                [
                    pl.col("daily_mean").fill_nan(None),
                    pl.col("daily_min").fill_nan(None),
                    pl.col("daily_max").fill_nan(None),
                ]
            )
            .group_by(
                [
                    "plot_id",
                    "code_country",
                    "code_plot",
                    "code_variable",
                    "month_year",
                    "year",
                    "month",
                ]
            )
            .agg(
                [
                    pl.col("daily_mean").mean().alias("avg_daily_mean"),
                    pl.col("daily_min").min().alias("min_daily_min"),
                    pl.col("daily_max").max().alias("max_daily_max"),
                    (pl.col("daily_min") < 0).sum().alias("frost_days"),
                    pl.col("daily_completeness").mean().alias("avg_completeness"),
                ]
            )
            .sort(["year", "month"])
        )

        # Temporal coverage
        coverage = (
            monthly_summary.with_columns(
                pl.col("month_year").str.strptime(pl.Date, "%m-%Y").alias("month_year_date")
            )
            .group_by(["plot_id", "code_variable"])
            .agg(
                [
                    pl.len().alias("n_rows"),
                    pl.col("month_year_date").min().alias("min_month_year"),
                    pl.col("month_year_date").max().alias("max_month_year"),
                    (
                        (
                            pl.col("month_year_date").max().dt.year()
                            - pl.col("month_year_date").min().dt.year()
                        )
                        * 12
                        + (
                            pl.col("month_year_date").max().dt.month()
                            - pl.col("month_year_date").min().dt.month()
                        )
                        + 1
                    ).alias("n_months"),
                ]
            )
            .sort("plot_id")
        )

        # Missing months
        missing_months_list = []
        for row in coverage.iter_rows(named=True):
            plot_id = row["plot_id"]
            code_var = row["code_variable"]
            min_month = row["min_month_year"]
            max_month = row["max_month_year"]
            existing_months = (
                monthly_summary.filter(
                    (pl.col("plot_id") == plot_id) & (pl.col("code_variable") == code_var)
                )
                .select("month_year")
                .to_series()
                .to_list()
            )
            all_months = (
                pd.date_range(start=min_month, end=max_month, freq="MS")
                .strftime("%m-%Y")
                .to_list()
            )
            missing = [m for m in all_months if m not in existing_months]
            missing_months_list.append(
                {"plot_id": plot_id, "n_missing_months": len(missing), "missing_months": missing}
            )

        missing_months_df = pl.DataFrame(missing_months_list)
        coverage = coverage.join(missing_months_df, on="plot_id")
        return coverage


if __name__ == "__main__":
    file_path = "./data/raw/ICP/595_mm_20260227091917/mm_mem.csv"
    processor = prepare_icp_weather_data(file_path)

    df_cleaned = processor.clean_data()
    monthly_agg = processor.aggregate_monthly_variables(["AT", "PR", "SR"])
    coverage = processor.compute_plot_temporal_coverage()

    print(df_cleaned.head())
    print(monthly_agg.head())
    print(coverage.head())
