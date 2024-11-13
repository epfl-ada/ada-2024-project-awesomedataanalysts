import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram_with_percentile(data, title, q=50, **hist_kwargs):
    percentile = np.percentile(data, q)
    n = np.sum(np.array(data) > percentile)

    plt.hist(data, **hist_kwargs)
    plt.axvline(percentile, color="r")

    plt.title(f"""{title}\n{q:.1f}th percentile (n={n:.0f}) in red corresponds to values >= {percentile:.3f} """)

    return percentile

def plot_avg_overall_map(df, title):
    # compute mean `avg_overall` by location
    location_data = df[["avg_overall", "location"]].copy()
    location_data = location_data.dropna()
    location_data["location"] = location_data["location"].apply(
        lambda x: "United States of America" if x.startswith("United States") else x
    )
    location_data = location_data.groupby("location")["avg_overall"].mean()

    # get map shapes data and combine with `avg_overall` values
    # from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
    world = gpd.read_file("./data/geopandas/ne_110m_admin_0_countries")
    world = world.merge(location_data, left_on='SOVEREIGNT', right_on='location', how='left')

    # plot map
    plt.figure(figsize=(15, 10))
    ax = plt.gca()

    world.boundary.plot(ax=ax)
    world.plot(column="avg_overall", ax=ax, legend=True,
               legend_kwds={"label": "average rating",
                            "orientation": "vertical"})

    # fix colorbar size
    cbar = ax.get_figure().axes[-1]
    cbar.set_position([0.77, 0.3, 0.03, 0.4]) # [left, bottom, width, height]

    ax.set_title(title)
