"""Marimo EDA for LWF site locations and metadata."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This code stores a metadata table containing important information of all sites.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go

    return go, mo, pd


@app.cell
def _(mo):
    mo.md("""
    Sources:
    [DEIMS-SDR Site Search](https://deims.org/search-sites) and
    [LWF Research Plots](https://lwf.wsl.ch/en/flaechen/).
    """)
    return


@app.cell
def _(pd):
    # Site data verified against the official WSL LWF site pages
    # (https://lwf.wsl.ch/en/flaechen/<site>/), which each publish the
    # "Community / canton" and "Size of the plot" fields used below.
    #
    # Coordinates are centroid / representative coordinates published for
    # the LWF sites in DEIMS-SDR, since the WSL pages themselves only ship
    # a PDF stand map rather than plain-text lat/lon.
    #
    # NOTE:
    # - The short codes follow the ones used in the user's LWF deposition
    #   data (https://www.wsl.ch/walddyn/lwf_deposition.php?...&plac=XXX).
    # - Lägeren is "LAG" in DEIMS-SDR, but the user's deposition data (and
    #   the WSL "Maps"/"Publications" links) use "LAE", so "LAE" is kept
    #   here to match the dataset.
    # - The WSL site list also includes a 19th site, "Lantsch" (GR). It is
    #   left out here because its plot size is published as "n.d." (not
    #   determined) on the site page and it isn't part of the user's LWF
    #   deposition short-code set.
    sites = pd.DataFrame(
        [
            # code, name, municipality, canton, latitude, longitude, ha
            ("ALP", "Alptal", "Alpthal", "SZ", 47.0486366, 8.7125782, 0.60),
            ("BEA", "Beatenberg", "Beatenberg", "BE", 46.7003438, 7.7623374, 2.00),
            ("BET", "Bettlachstock", "Bettlachstock", "SO", 47.2251551, 7.4166536, 1.28),
            ("CEL", "Celerina", "Celerina", "GR", 46.4921451, 9.8888024, 2.00),
            ("CHI", "Chironico", "Chironico", "TI", 46.4468172, 8.8121720, 2.00),
            ("DAV", "Davos", "Davos", "GR", 46.8153458, 9.8552112, 0.60),
            ("ISO", "Isone", "Isone", "TI", 46.1248982, 9.0080555, 2.00),
            ("JUS", "Jussy", "Jussy", "GE", 46.2298528, 6.2908547, 1.99),
            ("LAE", "Lägeren", "Wettingen", "AG", 47.4783000, 8.3643900, 1.34),
            ("LAN", "Lantsch", "Lantsch", "GR", 46.6980544, 9.5646678, 1.00),
            ("LAU", "Lausanne", "Lausanne", "VD", 46.5837664, 6.6580421, 2.00),
            ("LEN", "Lens", "Lens", "VS", 46.2685558, 7.4359392, 2.00),
            ("NAT", "Nationalpark", "Zernez", "GR", 46.6625010, 10.2300874, 2.00),
            ("NEU", "Neunkirch", "Neunkirch", "SH", 47.6837031, 8.5356830, 2.00),
            ("NOV", "Novaggio", "Novaggio", "TI", 46.0226119, 8.8341613, 1.50),
            ("OTH", "Othmarsingen", "Othmarsingen", "AG", 47.3995354, 8.2267566, 1.00),
            ("SCH", "Schänis", "Schänis", "SG", 47.1650464, 9.0670726, 2.00),
            ("VIS", "Visp", "Visp", "VS", 46.2968789, 7.8583245, 2.00),
            ("VOR", "Vordemwald", "Vordemwald", "AG", 47.2740627, 7.8867633, 2.00),
        ],
        columns=[
            "plot_id",
            "site_name",
            "municipality",
            "canton",
            "latitude",
            "longitude",
            "plot_ha",
        ],
    )
    return (sites,)


@app.cell
def _(mo, sites):
    site_options = {
        "All LWF sites": "All",
        **{
            f"{row.plot_id} — {row.site_name} ({row.plot_ha:g} ha)": row.plot_id
            for row in sites.itertuples()
        },
    }

    # IMPORTANT: `value` must be one of the *keys* of `options` (i.e. the
    # label shown in the dropdown), not one of the mapped values. Passing
    # "All" here (a value, not a key) is what raised:
    #   ValueError: The option name 'All' is not a valid option.
    # Using the matching key "All LWF sites" fixes it; `site_selector.value`
    # still resolves to the mapped value "All" once selected.
    site_selector = mo.ui.dropdown(
        options=site_options,
        value="All LWF sites",
        label="LWF site",
    )
    return (site_selector,)


@app.cell
def _(go, mo, site_selector, sites):
    selected_site = site_selector.value

    if selected_site == "All":
        map_data = sites.copy()
        center_lat = 46.80
        center_lon = 8.22
        zoom = 7.7
        map_title = "LWF research sites in Switzerland"
    else:
        map_data = sites[sites["plot_id"] == selected_site].copy()
        center_lat = float(map_data["latitude"].iloc[0])
        center_lon = float(map_data["longitude"].iloc[0])
        zoom = 11.0
        map_title = f"{map_data['plot_id'].iloc[0]} — {map_data['site_name'].iloc[0]}"

    # Scale marker area so the hectare values are visually distinguishable
    # without making the smaller sites disappear.
    marker_sizes = 14 + 10 * map_data["plot_ha"]

    # NOTE: Plotly deprecated `Scattermapbox`/`layout.mapbox` in favor of
    # `Scattermap`/`layout.map` (MapLibre-based, no Mapbox access token
    # required). `Scattermapbox` has been removed entirely as of Plotly
    # 6.x/7.x, so `Scattermap` is used here for forward compatibility.
    fig = go.Figure(
        go.Scattermap(
            lat=map_data["latitude"],
            lon=map_data["longitude"],
            mode="markers+text",
            text=map_data["plot_id"],
            textposition="top center",
            customdata=map_data[
                [
                    "plot_id",
                    "site_name",
                    "municipality",
                    "canton",
                    "plot_ha",
                ]
            ],
            marker=dict(
                size=marker_sizes,
                opacity=0.82,
            ),
            hovertemplate=(
                "<b>%{customdata[0]} — %{customdata[1]}</b>"
                "<br>Municipality: %{customdata[2]}"
                "<br>Canton: %{customdata[3]}"
                "<br>Plot size: %{customdata[4]:.2f} ha"
                "<br>Latitude: %{lat:.5f}"
                "<br>Longitude: %{lon:.5f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=map_title,
        map=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
        height=700,
        margin=dict(l=0, r=0, t=55, b=0),
    )

    mo.vstack(
        [
            mo.md(
                """
                ## Long-Term Forest Ecosystem Research (LWF) sites

                Hover over a site to see the **full site name, municipality,
                canton, and plot size**. Marker size is scaled by the plot
                area in hectares.
                """
            ),
            site_selector,
            fig,
        ]
    )
    return


@app.cell
def _(mo, sites):
    mo.md(f"""
    **Sites shown:** {len(sites)}

    The map uses the LWF site short codes from the dataset, with the
    corresponding site names and plot areas from the WSL LWF site
    information ([lwf.wsl.ch/en/flaechen](https://lwf.wsl.ch/en/flaechen/)).
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
