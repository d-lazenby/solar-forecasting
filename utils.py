import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import widgets, interact
import folium 
from collections import defaultdict
import base64

from typing import List

FONT_SIZE_TICKS = 15
FONT_SIZE_TITLE = 20
FONT_SIZE_AXES = 17

cmap = "twilight"
COLORS = sns.color_palette(cmap)
sns.set_palette(COLORS)


MONTHS = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}


def fix_dates(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    if 'Datetime' in data.columns:
        return data
    
    data['Datetime'] = data.apply(
        lambda x: datetime.strptime(f"{x['Date']} {x['Time']}", "%Y%m%d %H%M"), axis=1)
    
    data.drop(['Date', 'Time', 'Month', 'Hour', 'YRMODAHRMI'], axis='columns', inplace=True)
    
    data = data[["Datetime"] + [col for col in list(data.columns) if col not in ["Datetime", 'PolyPwr']] + ['PolyPwr']]
    
    return data

def fix_units(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    data = df.copy()
    features = ['Altitude', 'Wind.Speed', 'Visibility', 'Cloud.Ceiling']
    converted = data[features].describe().round(3).loc[['min', '25%', '50%', 'mean', '75%', 'max']].reset_index(drop=True).copy()
    
    # From paper. Each column in from_paper contains the agregate statistics in the order above
    from_paper = pd.DataFrame([[0.3, 0.6, 140, 244, 417, 593], 
                  [0, 9.7, 14.5, 16.6, 22.5, 78.9], 
                  [0, 16.1, 16.1, 15.6, 16.1, 16.1],
                  [0, 4.3, 22, 15.7, 22, 22]]).T
    
    # Calculate conversion factor and modify feature accordingly
    for idx, feature in enumerate(features):
        cf = np.mean(from_paper.iloc[:, idx] / converted.iloc[:, idx])
        data[feature] = data[feature] * cf
        
    return data


def visualise_missing_data(df: pd.core.frame.DataFrame) -> None:
    """
    Stacked bar chart to visualise missing data. total_possible_points is a hypothetical maximum of 
    readings that could be taken at each location from over the date range of df (see EDA notebook for details).
    
    Args:
        df: The dataset.
    """
    # Get df of counts and labels for plot
    total_possible_points = 11976

    missing_df = df.copy().groupby('Location').count()['Datetime'].sort_values(ascending=False).to_frame().reset_index()
    missing_df.rename(columns={'Datetime': 'num_readings'}, inplace=True)

    missing_df['prop_missing'] = (missing_df['num_readings'] / total_possible_points).round(2)

    locations = missing_df['Location'].values
    num_readings = missing_df['num_readings'].values
    missing_readings = (total_possible_points - missing_df['num_readings']).values
    percentage_labels = [f"{int(v*100)}%" for v in missing_df['prop_missing'].values]
    
    # Plot bar chart
    plt.figure(figsize=(8,4))

    bars1 = plt.bar(locations, num_readings, color=None)
    bars2 = plt.bar(locations, missing_readings, bottom=num_readings, color=None, alpha=0.2)

    plt.tight_layout()

    for b1, b2, l in zip(bars1, bars2, percentage_labels):
        plt.text(b1.get_x() + b2.get_width() / 2, b1.get_height() + 0.5, l,
                ha='center', va='bottom', color='black', fontsize=10)

    plt.xlabel('Location', fontsize=FONT_SIZE_AXES)
    plt.ylabel('Count', fontsize=FONT_SIZE_AXES)
    plt.title('Available data points from a hypothetical maximum of 11976')

    plt.xticks(range(len(locations)), labels=[l.replace(' ', '\n') for l in locations], fontsize=11, rotation=45)

    plt.show()
    
def add_day_month(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    if 'month_of_year' in df.columns:
        return df
    data = df.copy()
    data['month_of_year'] = pd.DatetimeIndex(data['Datetime']).month
    data['hour_of_day'] = pd.DatetimeIndex(data['Datetime']).hour
    data.loc[data['hour_of_day'] == 0, 'hour_of_day'] = 24
    data = data[[f for f in data.columns if f != 'Power (W)'] + ['Power (W)']]
    return data

def numerical_distributions(
    df: pd.core.frame.DataFrame, 
    features: List[str], 
    bins: int = 32,
    plot_type: str = 'hist') -> None:
    """
    Plot distributions of all numerical features at once. Can plot histograms,
    KDEs or boxplots.
    
    Args:
        df: The dataset.
        features: List of features to plot. Should only be numerical features.
        bins (optional): Number of bins that the histograms use.
        plot_type (optional): Type of plot. Can be 'hist' (default), 'kde' or 'box'.
    """
    # Set plot dimensions
    nplots = len(features)
    ncols = (nplots if nplots < 3 else 3)
    nrows = (nplots // 3 if nplots % 3 == 0 else nplots // 3 + 1)
    
    _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))
    plt.tight_layout()
    
    for i, ax_ in enumerate(ax.flatten()):
        if i >= nplots:
            # Suppress axes without plots
            ax_.axis('off')
        else:
            x = df[features[i]].values
            if plot_type == 'hist':
                sns.histplot(x, bins=bins, color=COLORS[i % len(COLORS)], ax=ax_)
                ylabel = "Count"
                ax_.set_ylabel((ylabel if i % 3 == 0 else ""))
                ax_.set_xlabel(f"{features[i]}")
            elif plot_type == 'kde':
                sns.kdeplot(x, color=COLORS[i % len(COLORS)], ax=ax_, linewidth=2)
                ylabel = "Density"
                ax_.set_ylabel((ylabel if i % 3 == 0 else ""))
                ax_.set_xlabel(f"{features[i]}")
            elif plot_type == 'box':
                sns.boxplot(x, color=COLORS[i % len(COLORS)], ax=ax_, width=0.25, medianprops={'color': 'yellow', 'alpha': 0.9, 'linewidth': 1.5})
                ylabel = f"{features[i]}"
                ax_.set_ylabel(ylabel)
            ax_.tick_params(axis="both")
            
    plt.subplots_adjust(hspace=0.25, wspace=0.2)
    plt.show()
    
def plot_skew(df: pd.core.frame.DataFrame, features: List[str]) -> None:
    """
    Bar chart to visualise skews of numerical features.
    
    Args:
        df: The dataset.
        features: List of numerical features.
    """
    x = df[features].skew()
    skews = np.round(x.values, 2)
    labels = x.index
    bars = plt.bar(x=labels, height=skews, color=None, edgecolor=None)
    

    for bar, skew in zip(bars, skews):
        if bar.get_height() < 0:
            pos = bar.get_height() - 0.3
        else:
            pos = bar.get_height() 
        plt.text(bar.get_x() + bar.get_width() / 2, pos, skew,
                ha='center', va='bottom', color='black', fontsize=10)
        
    plt.xticks(range(len(labels)), labels=[l.replace(' ', '\n') if l != 'AmbientTemp (deg C)' else 'AmbientTemp\n(deg C)' for l in labels], fontsize=9, rotation=45)
    plt.ylim([-5.7, 1.2])
    plt.ylabel("Skew")
    plt.show();

def plot_power_against_time(df: pd.core.frame.DataFrame) -> None:
    """
    Line plot to show relationship of average power against timeframe of either month of year or
    hour of day.
    
    Args:
        df: The dataset.
    """
    time_increments = ['hour_of_day', 'month_of_year']
    power = 'Power (W)'

    if 'month_of_year' not in df.columns:
        df = add_day_month(df)
        
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4))

    for i, ax_ in enumerate(ax.flatten()):
        data = df[[power, time_increments[i]]].copy()
        data_grouped = data.groupby(time_increments[i]).agg(({power: ['mean', 'std']}))

        keys = data_grouped[power].index
        values = data_grouped[power]['mean'].values
        stds = data_grouped[power]['std'].values

        ax_.plot(keys, values, '-o', color=None, markeredgecolor=None)
        ax_.fill_between(keys, values - stds, values + stds, alpha=0.2, color=None)
        ax_.set_title(f'Avg. {power} / {time_increments[i].split("_")[0]}')
        ax_.set_ylabel(f'Avg. {power}')
        ax_.set_xlabel(time_increments[i][0].upper() + time_increments[i][1:].replace('_', ' '))
        if time_increments[i] == 'month_of_year':
            ax_.set_xticks(keys, labels=[MONTHS[k] for k in keys], fontsize=11)
            
    plt.show();

    

def create_map(df: pd.core.frame.DataFrame, time_increment: str = 'month_of_year') -> folium.Map:
    """
    Interactive map displaying geolocations of PV arrays. Includes popups of average (hourly or monthly)
    power output of a given cell.
    
    Args:
        df: The dataset.
        time_increment (optional): how to average the power output. Can be 'month_of_year' or 'hour_of_day'. 
    """
    
    power = 'Power (W)'
    if 'month_of_year' not in df.columns:
        df = add_day_month(df)
    data = df[['Latitude (deg)', 'Longitude (deg)', power, 'Location', time_increment]].copy()
    data_grouped = data.groupby(['Location', time_increment]).agg(({power: ['mean', 'std']}))
    ymin = data_grouped[power]['mean'].min()
    ymax = data_grouped[power]['mean'].max()

    grouped_means = defaultdict(dict)
    grouped_stds = defaultdict(dict)

    for index, row in data_grouped.iterrows():
        grouped_means[index[0]][index[1]] = row[0]
        grouped_stds[index[0]][index[1]] = row[1]

    for key in grouped_means:
        if (time_increment == 'month_of_year'):
            keys = list(grouped_means[key].keys())
            label = 'Monthly average'
        else:
            keys = list(grouped_means[key].keys())
            label = 'Hourly average'

        values, stds = [], []
        for subkey in keys:
            values.append(grouped_means[key][subkey])
            stds.append(grouped_stds[key][subkey])
        
        values = np.array(values)
        stds = np.array(stds)
        
        plt.plot(keys, values, '-o', label=label, color=None)
        plt.fill_between(keys, values - stds, values + stds, alpha=0.2, color=None)
        plt.ylim(ymin, ymax)
        plt.title(f'Location {key} avg. {power} / {time_increment.split("_")[0]}')
        plt.ylabel(f'Avg. {power}')
        plt.xlabel(time_increment[0].upper() + time_increment[1:].replace('_', ' '))
        plt.legend(loc="upper left")
        if (time_increment == 'month_of_year'):
            plt.xticks(keys, labels=[MONTHS[k] for k in keys], fontsize=11)
        plt.savefig(f'img/tmp/{key}.png', dpi=50)

        plt.clf()

    data_grouped_grid = data.groupby('Location').agg(({power: 'mean', 'Latitude (deg)': 'min', 'Longitude (deg)': 'min'}))
    
    data_grouped_grid_array = np.array(
        [
            data_grouped_grid['Latitude (deg)'].values,
            data_grouped_grid['Longitude (deg)'].values,
            data_grouped_grid[power].values,
            data_grouped_grid.index.values
        ]
    ).T

    m = folium.Map(
        location=[29.519934, -105.6],
        tiles='openstreetmap',
        zoom_start=3.5,
        width=1200,
        height=800
    )

    width, height = 300, 300
    fg = folium.FeatureGroup(name="My Map")
    for lat, lng, _, location in data_grouped_grid_array:
        enc = base64.b64encode(open(f"img/tmp/{location}.png", 'rb').read())
        html = "<img src='data:image/png;base64,{}'>".format
        iframe = folium.IFrame(html(enc.decode('UTF-8', errors='ignore')), width=width+10, height=height+10)
        popup = folium.Popup(iframe, max_width=300, max_height=250)
        fg.add_child(folium.CircleMarker(location=[lat, lng], radius = 15, popup=popup,
        fill_color=None, color='', fill_opacity=0.5))
        m.add_child(fg)
        
    return m

def histogram_plot(df: pd.core.frame.DataFrame, features: List[str], bins: int = 16) -> None:
    """Interactive histogram plots to investigate variation of features by location.

    Args:
        df: The dataset.
        features: List of features to include in the plot.
        bins (optional): Number of bins in the histogram. Defaults to 16.
    """

    def _plot(location, feature):
        data = df[df['Location'] == location]
        plt.figure(figsize=(8, 5))
        x = data[feature].values
        plt.xlabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        sns.histplot(x, bins=bins, color=None)
        plt.ylabel(f"Count", fontsize=FONT_SIZE_AXES)
        plt.title(f"{feature} at {location}", fontsize=FONT_SIZE_TITLE)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.show()

    location_selection = widgets.Dropdown(
        options=df['Location'].unique(), 
        value=df['Location'].unique()[-1], 
        description="Location"
    )

    feature_selection = widgets.Dropdown(
        options=features,
        value="Power (W)",
        description="Feature",
    )

    interact(_plot, location=location_selection, feature=feature_selection)
    
def compare_histograms(df: pd.core.frame.DataFrame, features: List[str], bins: int = 16) -> None:
    """Interactive histogram plots for side-by-side comparison of features 
    at two different locations.

    Args:
        df: The dataset.
        features: List of features to include in the plot.
        bins (optional): Number of bins in the histogram. Defaults to 16.
    """

    def _plot(location1, location2, feature):
        data1 = df[df['Location'] == location1]
        data2 = df[df['Location'] == location2]

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        x1 = data1[feature].values
        x2 = data2[feature].values

        ax1.set_xlabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        ax2.set_xlabel(f"{feature}", fontsize=FONT_SIZE_AXES)

        ax1.set_ylabel(f"Count", fontsize=FONT_SIZE_AXES)
        ax2.set_ylabel(f"Count", fontsize=FONT_SIZE_AXES)

        sns.histplot(x1, bins=bins, ax=ax1, color=None)
        sns.histplot(x2, bins=bins, ax=ax2, color=None)

        ax1.set_title(f"{location1}", fontsize=FONT_SIZE_TITLE)
        ax1.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

        ax2.set_title(f"{location2}", fontsize=FONT_SIZE_TITLE)
        ax2.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

        fig.tight_layout()
        fig.show()

    location_selection1 = widgets.Dropdown(
        options=df['Location'].unique(),
        value=df['Location'].unique()[-2],
        description="Location 1",
        style={"description_width": "initial"},
    )

    location_selection2 = widgets.Dropdown(
        options=df['Location'].unique(),
        value=df['Location'].unique()[-1],
        description="Location 2",
        style={"description_width": "initial"},
    )

    feature_selection = widgets.Dropdown(
        options=features,
        value='Power (W)',
        description="Feature",
    )

    interact(
        _plot,
        location1=location_selection1,
        location2=location_selection2,
        feature=feature_selection,
    )
    
def compare_box_violins(df: pd.core.frame.DataFrame, features: List[str]) -> None:
    """Interactive violin/box plots for comparison of features across all locations.

    Args:
        df: The data.
        features: List of features to include in the plot.
    """
    locations = df["Location"].unique()
    if len(locations) > len(COLORS):
        # Extend the color map to the number of locations.
        q, r = divmod(len(locations), len(COLORS))
        long_colors = q * COLORS + COLORS[:r]

    def _plot(feature="AmbientTemp", plot_type="box"):
        plt.figure(figsize=(18, 8))
        scale = "linear"
        plt.yscale(scale)
        if plot_type == "Violin":
            sns.violinplot(
                data=df, y=feature, x="Location", order=locations, hue='Location', palette=long_colors 
                )
        elif plot_type == "Box":
            sns.boxplot(
                data=df, y=feature, x="Location", order=locations, hue='Location', palette=long_colors, 
                medianprops={'color': 'yellow', 'alpha': 0.9, 'linewidth': 1.5}
                )
        plt.title(f"{feature}", fontsize=FONT_SIZE_TITLE)
        plt.ylabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        plt.xlabel(f"Location", fontsize=FONT_SIZE_AXES)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.xticks(range(len(locations)), labels=[l.replace(' ', '\n') for l in locations], fontsize=11)
        
        plt.show()

    feature_selection = widgets.Dropdown(
        options=features,
        value='Power (W)',
        description="Feature",
    )

    plot_type_selection = widgets.Dropdown(
        options=["Violin", "Box"], description="Plot Type"
    )

    interact(_plot, feature=feature_selection, plot_type=plot_type_selection)
    
    
def scatterplot(df: pd.core.frame.DataFrame, features: List[str]) -> None:
    """Interactive scatterplots of the data.

    Args:
        df: The data.
        features: List of features to include in the plot.
    """
    if df.isna().sum().sum() != 0:
        df_clean = df.dropna(inplace=False)
    else:
        df_clean = df.copy()

    def _plot(location, var_x, var_y, add_noise):
        plt.figure(figsize=(8, 5))
        df_clean_2 = df_clean[df_clean['Location'] == location]
        x = df_clean_2[var_x].values
        y = df_clean_2[var_y].values
        if add_noise and var_x == 'WindSpeed (km/h)':
            x += np.random.uniform(-1, 1, len(x))
            x = [v if v >= 0 else 0 for v in x ]

        plt.plot(
            x, y,
            marker='o', markersize=7, markerfacecolor=None, 
            markeredgewidth=0,
            linestyle='', 
            alpha=0.2
        )
        
        
        plt.xlabel(var_x, fontsize=FONT_SIZE_AXES)
        plt.ylabel(var_y, fontsize=FONT_SIZE_AXES)

        plt.title(f"{var_x} vs {var_y}", fontsize=FONT_SIZE_TITLE)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.show()

    location_selection = widgets.Dropdown(
        options=df['Location'].unique(), value=df['Location'].unique()[-1], description="Location"
    )

    x_var_selection = widgets.Dropdown(options=features, description="X-Axis")

    y_var_selection = widgets.Dropdown(
        options=features, description="Y-Axis", value="Power (W)"
    )

    add_noise_button = widgets.Checkbox(
        value=False, description="Blur wind", disabled=False
    )

    interact(
        _plot,
        location=location_selection,
        var_x=x_var_selection,
        var_y=y_var_selection,
        add_noise=add_noise_button
    )
    
def correlation_matrix(data: pd.core.frame.DataFrame) -> None:
    """Plots correlation matrix for a given dataset.

    Args:
        data: The dataset used.
    """
    
    continuous_cmap = sns.diverging_palette(12, 250, s=100, l=40, center='light', as_cmap=True)
    plt.figure(figsize=(10, 10))
    sns.heatmap(data.corr(), annot=True, cbar=False, cmap=continuous_cmap, vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Features")
    plt.show()
    
def plot_time_series(df: pd.core.frame.DataFrame, features: List[str]) -> None:
    """Creates interactive plots for the time series in the dataset.

    Args:
        df: The data used.
        features: Features to include in the plot.
    """

    def _plot(location, feature, date_range):
        data = df[df["Location"] == location]
        
        data = data[data["Datetime"] > date_range[0]]
        data = data[data["Datetime"] < date_range[1]]
        plt.figure(figsize=(15, 5))
        plt.plot(data["Datetime"], data[feature], "-", color=None)
        plt.ylabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        plt.xlabel(f"Date", fontsize=FONT_SIZE_AXES)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.show()

    location_selection = widgets.Dropdown(
        options=df['Location'].unique(),
        value=df['Location'].unique()[-1],
        description="Location",
    )

    feature_selection = widgets.Dropdown(
        options=features,
        value='Power (W)',
        description="Feature",
    )

    dates = pd.date_range(datetime(2017, 5, 23), datetime(2018, 10, 4), freq="D")

    options = [(date.strftime("%d %b %y"), date) for date in dates]
    index = (0, len(options) - 1)

    date_slider_selection = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description="Date",
        orientation="horizontal",
        layout={"width": "550px"},
    )

    interact(
        _plot,
        location=location_selection,
        feature=feature_selection,
        date_range=date_slider_selection,
    )
    
def group_plots(df: pd.core.frame.DataFrame, cat_feature: str, num_feature: str) -> None:
    """
    Plots numerical features grouped by a categorical feature to show interaction effects.
    
    Args:
        df: The dataset.
        cat_feature: The grouping feature.
        num_feature: The numerical feature.
    """
    sns.lmplot(
        x=num_feature, y="Power (W)", hue=cat_feature, col=cat_feature,
        data=df, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4, palette=COLORS
    )
    plt.show()