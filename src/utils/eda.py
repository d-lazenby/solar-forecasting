import pandas as pd
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

from typing import List

FONT_SIZE_TICKS = 15
FONT_SIZE_TITLE = 20
FONT_SIZE_AXES = 17

COLORS = sns.color_palette("twilight")
sns.set_palette(COLORS)

MONTHS = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}


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
    data = data[[f for f in data.columns if f != 'PolyPwr'] + ['PolyPwr']]
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
                sns.kdeplot(x, color=COLORS[i % len(COLORS)], ax=ax_, linewidth=2, fill=True)
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
    skews = df[features].skew().round(2)
    labels = skews.index
    
    # Color bars based on values 
    v = max([abs(skews.min()), abs(skews.max())])
    norm = matplotlib.colors.Normalize(vmin=-v, vmax=v)
    cmap = matplotlib.colors.ListedColormap(sns.diverging_palette(12, 250, s=100, l=40, center='light', as_cmap=False, n=100)).reversed()

    bars = plt.bar(x=labels, height=skews, color=cmap(norm(skews.values)), edgecolor=None)
    
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
    power = 'PolyPwr'

    if 'month_of_year' not in df.columns:
        df = add_day_month(df)
        
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4))

    for i, ax_ in enumerate(ax.flatten()):
        data = df[[power, time_increments[i]]].copy()
        data_grouped = data.groupby(time_increments[i]).agg(({power: ['mean', 'std']}))

        keys = data_grouped[power].index
        values = data_grouped[power]['mean'].values
        stds = data_grouped[power]['std'].values

        ax_.plot(keys, values, '-o', color=COLORS[3], markeredgecolor=None, markerfacecolor=None)
        ax_.fill_between(keys, values - stds, values + stds, alpha=0.2, color=COLORS[0])
        ax_.set_title(f'Avg. {power} / {time_increments[i].split("_")[0]}')
        ax_.set_ylabel(f'Avg. {power}')
        ax_.set_xlabel(time_increments[i][0].upper() + time_increments[i][1:].replace('_', ' '))
        if time_increments[i] == 'month_of_year':
            ax_.set_xticks(keys, labels=[MONTHS[k] for k in keys], fontsize=11)
            
    plt.show();

def plot_grouped_power_against_time(df: pd.core.frame.DataFrame, time_increment: str = 'hour_of_day', cat_col: str = 'Location') -> None:
    """
    Line plot to show relationship of average power against timeframe of either month of year or
    hour of day grouped by categorical column, e.g. location or season.
    
    Args:
        df: The dataset.
        time_increment (optional): 'hour_of_day' or 'month_of_year'.
        cat_col (optional): Categorical column to group on. 
    """
    # time_increments = ['hour_of_day', 'month_of_year']
    power = 'PolyPwr'

    if 'month_of_year' not in df.columns:
        df = add_day_month(df)
    
    unique_cats = df[cat_col].unique()

    nplots = len(unique_cats)
    if nplots == 4:
        ncols, nrows = 2, 2
    else:    
        ncols = (nplots if nplots < 3 else 3)
        nrows = (nplots // 3 if nplots % 3 == 0 else nplots // 3 + 1)

    _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows), sharey=True)
    plt.tight_layout()
    # FILTER BY CAT COL FIRST THEN IT SHOULD BE EASY
    data = df[[power, cat_col, time_increment]].copy()

    for i, ax_ in enumerate(ax.flatten()):
        if i >= nplots:
            # Suppress axes without plots
            ax_.axis('off')
        else:
            data_grouped = data[data[cat_col] == (cat_i := unique_cats[i])].groupby(time_increment)[power]
            keys = data_grouped.mean().index
            values = data_grouped.mean().values
            stds = data_grouped.std().values
            ax_.plot(keys, values, '-o', color=COLORS[3], markeredgecolor=None, markerfacecolor=None, label=f'{cat_i}')
            ax_.fill_between(keys, values - stds, values + stds, alpha=0.2, color=COLORS[0])
            # ax_.set_title(f'{cat_i}')
            ax_.set_ylabel(f'Avg. {power}')
            ax_.set_xlabel(time_increment[0].upper() + time_increment[1:].replace('_', ' '))
            ax_.legend(loc='upper left')
            if time_increment == 'month_of_year':
                ax_.set_xticks(keys, labels=[MONTHS[k] for k in keys], fontsize=11, rotation=45)
                ax_.set_xlabel(None)

    plt.subplots_adjust(hspace=0.25, wspace=None)
    plt.show()    
    
def correlation_matrix(data: pd.core.frame.DataFrame, annot: bool = True) -> None:
    """Plots correlation matrix for a given dataset.

    Args:
        data: The dataset used.
    """
    
    continuous_cmap = sns.diverging_palette(12, 250, s=100, l=40, center='light', as_cmap=True).reversed()
    plt.figure(figsize=(10, 10))
    sns.heatmap(data.corr(), annot=annot, cbar=False, cmap=continuous_cmap, vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Features")
    plt.show()
        
def group_plots(df: pd.core.frame.DataFrame, cat_feature: str, num_feature: str) -> None:
    """
    Plots numerical features grouped by a categorical feature to show interaction effects.
    
    Args:
        df: The dataset.
        cat_feature: The grouping feature.
        num_feature: The numerical feature.
    """
    sns.lmplot(
        x=num_feature, y="PolyPwr", hue=cat_feature, col=cat_feature,
        data=df, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4, palette=COLORS
    )
    plt.show()
    
def make_mi_scores(inputs: pd.core.frame.DataFrame, target: pd.Series) -> pd.Series:
    inputs = inputs.copy()
    for col in inputs.select_dtypes(["object", "category"]):
        inputs[col], _ = inputs[col].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in inputs.dtypes]
    mi_scores = mutual_info_regression(inputs, target, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=inputs.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(mi_scores: pd.Series) -> None:
    mi_scores = mi_scores.sort_values(ascending=True)
    yvalues = np.arange(len(mi_scores))
    ylabels = list(mi_scores.index)
    plt.barh(yvalues, mi_scores)
    plt.yticks(yvalues, ylabels)
    plt.title("Mutual Information Scores")
    
def plot_percentages_by_label(df: pd.core.frame.DataFrame, feature: str) -> None:
    """
    Plots label frequency for categorical features with a red line to indicate potentially rare labels
    that appear in less than 5% of the dataset. 
    """
    total_readings = df.shape[0]
    temp_df = pd.Series(df[feature].value_counts() / total_readings)
    
    fig = temp_df.sort_values(ascending=False).plot.bar()
    fig.set_xlabel(feature)
    fig.axhline(y=0.05, color='red', ls='--')
    fig.set_ylabel('Percentage of readings')
    plt.show()
    
def calculate_mean_target_per_category(df: pd.core.frame.DataFrame, feature: str) -> None:
    total_readings = df.shape[0]
    temp_df = pd.Series(df[feature].value_counts() / total_readings).reset_index()
    temp_df.columns = [feature, 'perc_readings']
    temp_df = temp_df.merge(df.groupby([feature])['PolyPwr'].mean().reset_index(),
                            on=feature,
                            how='left')

    return temp_df

def plot_categories(df: pd.core.frame.DataFrame, feature: str) -> None:
    _, ax = plt.subplots(figsize=(8,4))
    plt.xticks(df.index, df[feature], rotation=90)

    ax2 = ax.twinx()
    ax.bar(df.index, df["perc_readings"], color='lightgrey')
    ax2.plot(df.index, df["PolyPwr"], color='green')
    ax.axhline(y=0.05, color='red')
    ax.set_ylabel('Percentage of readings per category')
    ax.set_xlabel(feature)
    ax2.set_ylabel('Average Power output (W) per category')
    plt.show()
    
def group_rare_labels(df: pd.core.frame.DataFrame, feature: str) -> None:
    total = df.shape[0]

    temp_df = pd.Series(df[feature].value_counts() / total)

    grouping_dict = {
        k: ('rare' if k not in temp_df[temp_df >= 0.05].index else k) for k in temp_df.index
    }

    tmp = df[feature].map(grouping_dict)

    return tmp

def diagnostic_plots(df: pd.core.frame.DataFrame, features: List[str]) -> None:
    nrows = len(features)
    _, ax = plt.subplots(nrows=nrows, ncols=3, figsize=(16, 4 * nrows))
    plt.tight_layout()
    
    boxplot_props = {
        'boxprops': {'facecolor': COLORS[0], 'edgecolor': COLORS[3]},
        'medianprops': {'color': COLORS[3], 'alpha': 0.9, 'linewidth': 1.5},
        'whiskerprops': {'color': COLORS[3]},
        'capprops': {'color': COLORS[3]},
        'flierprops': {'markersize': 8, 'markerfacecolor': 'white', 'markeredgecolor': COLORS[3]},
    }

    for i, feature in enumerate(features):
        # histogram
        sns.histplot(df[feature], bins=30, ax=ax[i,0], color=COLORS[0], edgecolor=COLORS[3])
        ax[i,0].set_title('Histogram')

        # Q-Q plot
        stats.probplot(df[feature], dist="norm", plot=ax[i,1])
        ax[i,1].get_lines()[0].set_marker('o')
        ax[i,1].get_lines()[0].set_markerfacecolor("white")
        ax[i,1].get_lines()[0].set_markeredgecolor(COLORS[3])
        ax[i,1].get_lines()[0].set_markersize(8.0)
        ax[i,1].get_lines()[1].set_color('red')
        ax[i,1].get_lines()[1].set_linewidth(3.0)
        ax[i,1].set_ylabel('Quantiles')

        # boxplot
        sns.boxplot(y=df[feature], ax=ax[i,2], width=0.25, color=COLORS[0], **boxplot_props)
        ax[i,2].set_title('Boxplot')
        
    plt.subplots_adjust(hspace=0.25, wspace=0.2)
    plt.show()


def plot_variance(pca: PCA) -> None:
    # Create figure
    _, ax = plt.subplots(nrows=1, ncols=2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    
    # Explained variance
    explained_var_ratio = pca.explained_variance_ratio_
    sns.barplot(x=grid, y=explained_var_ratio, ax=ax[0], color=COLORS[0], edgecolor=COLORS[3])
    ax[0].set(xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0))
    
    # Cumulative Variance
    cumulative_var = np.cumsum(explained_var_ratio)
    sns.lineplot(x=np.r_[0, grid], y=np.r_[0, cumulative_var], ax=ax[1], marker='o', markeredgecolor=COLORS[3])
    ax[1].set(xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0))
    
    plt.show()
    
def hex_bin_plot(df: pd.core.frame.DataFrame, features: List[str], target: str) -> None:
    if (n_features := len(features)) != 2:
        raise ValueError(f"The function requires exactly 2 features and you have supplied {n_features}")

    cmap = matplotlib.colors.ListedColormap(sns.diverging_palette(12, 250, s=100, l=40, center='light', as_cmap=False, n=100)).reversed()
    
    hb = plt.hexbin(df[features[0]], df[features[1]], C=df[target], gridsize=30, reduce_C_function=np.mean, cmap=cmap)
    plt.colorbar(hb, label=f'Average {target}')
    plt.xlabel('Humidity')
    plt.ylabel('AmbientTemp')
    plt.title(f'Hexbin Plot of {features[0]} against {features[1]}')

    plt.show()