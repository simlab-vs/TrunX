import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    sites = [1,2,3]

    mo.ui.dropdown(sites)
    return


@app.cell
def _():
    import pandas as pd

    df = pd.read_excel("legend_deposition_2026-07-28.xlsx")
    df2 = pd.read_csv("monthly_dep_lwf_2026-07-28.csv")
    return (pd,)


@app.cell
def _():
    from pathlib import Path

    data_dir = Path("data")

    files = sorted(data_dir.iterdir())

    files
    return Path, files


@app.cell
def _(files):
    for file in files:
        print(f"{file.name:60} {file.suffix}")
    return


@app.cell
def _(Path, pd):
    lai_path = Path("LAI_Licor_3_rings_all_years.xlsx")

    raw_lai = pd.read_excel(
        lai_path,
        header=None,
    )

    raw_lai.head()
    return (raw_lai,)


@app.cell
def _(pd, raw_lai):
    lai = raw_lai.iloc[3:].copy()

    lai.columns = [
        "plot",
        "subplot",
        "date",
        "plot_type",
        "lai_miller",
        "se_miller",
        "lai_norman_campbell",
        "se_norman_campbell",
        "angle_miller",
        "se_angle_miller",
        "angle_norman_campbell",
        "se_angle_norman_campbell",
        "number_of_points",
        "season",
    ]

    lai = lai.reset_index(drop=True)

    # Strip whitespace from text fields
    for column in ["plot", "subplot", "plot_type", "season"]:
        lai[column] = lai[column].astype(str).str.strip()

    # Convert "." to missing values
    lai = lai.replace(".", pd.NA)

    # Parse dates
    lai["date"] = pd.to_datetime(lai["date"], errors="coerce")

    # Numeric variables
    numeric_columns = [
        "lai_miller",
        "se_miller",
        "lai_norman_campbell",
        "se_norman_campbell",
        "angle_miller",
        "se_angle_miller",
        "angle_norman_campbell",
        "se_angle_norman_campbell",
        "number_of_points",
    ]

    for column in numeric_columns:
        lai[column] = pd.to_numeric(lai[column], errors="coerce")

    lai.head()
    return (lai,)


@app.cell
def _(lai, mo):
    plot_selector = mo.ui.dropdown(
        options=["All"] + sorted(
            lai["plot"].dropna().unique().tolist()
        ),
        value="All",
        label="Plot",
    )

    subplot_selector = mo.ui.dropdown(
        options=["All"] + sorted(
            lai["subplot"].dropna().unique().tolist()
        ),
        value="All",
        label="Subplot",
    )

    plot_type_selector = mo.ui.dropdown(
        options=["All"] + sorted(
            lai["plot_type"].dropna().unique().tolist()
        ),
        value="All",
        label="Plot type",
    )

    season_selector = mo.ui.dropdown(
        options=["All"] + sorted(
            lai["season"].dropna().unique().tolist()
        ),
        value="All",
        label="Season",
    )

    mo.hstack(
        [
            plot_selector,
            subplot_selector,
            plot_type_selector,
            season_selector,
        ]
    )
    return plot_selector, plot_type_selector, season_selector, subplot_selector


@app.cell
def _(
    lai,
    plot_selector,
    plot_type_selector,
    season_selector,
    subplot_selector,
):
    filtered_lai = lai.copy()

    if plot_selector.value != "All":
        filtered_lai = filtered_lai[
            filtered_lai["plot"] == plot_selector.value
        ]

    if subplot_selector.value != "All":
        filtered_lai = filtered_lai[
            filtered_lai["subplot"] == subplot_selector.value
        ]

    if plot_type_selector.value != "All":
        filtered_lai = filtered_lai[
            filtered_lai["plot_type"] == plot_type_selector.value
        ]

    if season_selector.value != "All":
        filtered_lai = filtered_lai[
            filtered_lai["season"] == season_selector.value
        ]

    filtered_lai
    return (filtered_lai,)


@app.cell
def _(mo):
    measurement_options = {
        "LAI — Miller": "lai_miller",
        "LAI — Norman & Campbell": "lai_norman_campbell",
        "Mean angle — Miller": "angle_miller",
        "Mean angle — Norman & Campbell": "angle_norman_campbell",
    }

    measurement_selector = mo.ui.dropdown(
        options=list(measurement_options.keys()),
        value="LAI — Miller",
        label="Measurement",
    )

    measurement_selector
    return measurement_options, measurement_selector


@app.cell
def _(measurement_options, measurement_selector):
    selected_measurement = measurement_options[
        measurement_selector.value
    ]

    selected_measurement
    return (selected_measurement,)


@app.cell
def _():
    import plotly.graph_objects as go
    import numpy as np

    return (go,)


@app.cell
def _(filtered_lai, go, measurement_selector, selected_measurement):
    plot_data = filtered_lai[
        ["date", selected_measurement]
    ].copy()

    plot_data = (
        plot_data
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ============================================================
    # SETTINGS
    # ============================================================

    # Gaps longer than this are shown as dashed lines
    max_gap_days = 365 * 2


    # ============================================================
    # FIGURE
    # ============================================================

    fig = go.Figure()


    # ============================================================
    # 1. MEASURED OBSERVATIONS
    # ============================================================

    measured = (
        plot_data
        .dropna(subset=[selected_measurement])
        .sort_values("date")
        .reset_index(drop=True)
    )

    fig.add_trace(
        go.Scatter(
            x=measured["date"],
            y=measured[selected_measurement],
            mode="markers",
            name="Measured",
            marker=dict(size=8),

            # Full date shown when hovering
            hovertemplate=(
                "<b>Date:</b> %{x|%d %B %Y}"
                "<br>"
                f"<b>{measurement_selector.value}:</b> "
                "%{y:.2f}"
                "<extra></extra>"
            ),

            showlegend=True,
        )
    )


    # ============================================================
    # 2. CONNECT MEASUREMENTS
    # ============================================================

    for i in range(len(measured) - 1):

        current = measured.iloc[i]
        next_row = measured.iloc[i + 1]

        gap_days = (
            next_row["date"] - current["date"]
        ).days

        # --------------------------------------------------------
        # Normal gap → solid line
        # --------------------------------------------------------

        if gap_days <= max_gap_days:

            fig.add_trace(
                go.Scatter(
                    x=[
                        current["date"],
                        next_row["date"],
                    ],
                    y=[
                        current[selected_measurement],
                        next_row[selected_measurement],
                    ],
                    mode="lines",
                    line=dict(width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # --------------------------------------------------------
        # Long gap → dashed line
        # --------------------------------------------------------

        else:

            fig.add_trace(
                go.Scatter(
                    x=[
                        current["date"],
                        next_row["date"],
                    ],
                    y=[
                        current[selected_measurement],
                        next_row[selected_measurement],
                    ],
                    mode="lines",
                    line=dict(
                        width=2,
                        dash="dash",
                    ),
                    name="Long data gap",
                    showlegend=(i == 0),
                    hoverinfo="skip",
                )
            )


    # ============================================================
    # 3. HIGHLIGHT NaN DATES
    # ============================================================

    missing_dates = (
        plot_data.loc[
            plot_data[selected_measurement].isna(),
            "date"
        ]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    for date in missing_dates:

        fig.add_vline(
            x=date,
            line_width=2,
            line_dash="dot",
            line_color="red",
            showlegend=False,
        )


    # ============================================================
    # 4. LEGEND ENTRY FOR MISSING OBSERVATIONS
    # ============================================================

    if len(missing_dates) > 0:

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(
                    width=2,
                    dash="dot",
                    color="red",
                ),
                name="Missing observation",
                showlegend=True,
                hoverinfo="skip",
            )
        )


    # ============================================================
    # 5. LAYOUT
    # ============================================================

    fig.update_layout(
        title=f"{measurement_selector.value} over time",
        xaxis_title="Date",
        yaxis_title=measurement_selector.value,

        # Important: inspect individual observations
        hovermode="closest",
    )

    fig
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
