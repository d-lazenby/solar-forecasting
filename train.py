from comet_ml import Experiment

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

FONT_SIZE_TICKS = 15
FONT_SIZE_TITLE = 20
FONT_SIZE_AXES = 17

COLORS = sns.color_palette("twilight")
sns.set_palette(COLORS)


def load_data():
    data = pd.read_csv('data/base_features.csv')
    y = data.pop('Power (W)')
    X = data.copy()

    # Fix datatypes
    X[['month_of_year', 'hour_of_day']] = X[['month_of_year', 'hour_of_day']].astype('int8')
    X[['Season', 'Location']] = X[['Season', 'Location']].astype('category')
    
    X_train, X_testvalid, y_train, y_testvalid = train_test_split(X, y, train_size=0.6, random_state=0)
    X_valid, X_test, y_valid, y_test = train_test_split(X_testvalid, y_testvalid, train_size=0.5, random_state=1)

    print(f"Training size: {X_train.shape}\nValid size: {X_valid.shape}\nTest size: {X_test.shape}")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def make_mi_scores(inputs, target):
    inputs = inputs.copy()
    for col in inputs.select_dtypes(["object", "category"]):
        inputs[col], _ = inputs[col].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in inputs.dtypes]
    mi_scores = mutual_info_regression(inputs, target, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=inputs.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

# Feature engineering functions
def label_encode(inputs):
    X = inputs.copy()
    for col in X.select_dtypes(["category"]):
        X[col] = X[col].cat.codes
    return X


def get_interactions(inputs):
    numerical = [col for col in inputs.columns if inputs[col].dtypes not in ['object', 'category'] 
             and col not in ['Longitude (deg)', 'Latitude (deg)']]
    categorical = ['Season', 'Location']
    
    def _interactions(df, cat_col, num_col):
        X = pd.get_dummies(df[cat_col], prefix=num_col)
        X = X.mul(df[num_col], axis=0)
        return X
    
    interaction_cols = []
    for cat_col in categorical:
        for num_col in numerical:
            X = _interactions(inputs, cat_col, num_col)
            interaction_cols.append(X)
    X = inputs.join(interaction_cols)
    return X


# Feature pipeline
def create_features(inputs_train, inputs_valid):
    X = pd.concat([inputs_train, inputs_valid])
    
    # Interaction columns
    X = get_interactions(X)
    
    # Label encode categorical features
    X = label_encode(X)
    
    train_enc = X.loc[inputs_train.index]
    valid_enc = X.loc[inputs_valid.index]
    return train_enc, valid_enc

# Training functions
def get_metrics(y_true, y_pred):
    metrics = {}
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = root_mean_squared_error(y_true, y_pred)
    return metrics


def get_model(params={'objective': 'reg:squarederror',
                      'n_estimators': 1000,
                      'learning_rate': 0.1}):
    
    model = XGBRegressor(**params, random_state=0)
        
    return model


def run_experiment(exp_name, inputs_train, target_train, inputs_valid, target_valid, baseline=None):
    
    experiment = Experiment(project_name="solar-forecasting",
                            workspace="d-lazenby")
    experiment.set_name(exp_name)
    
    model = get_model()
    
    with experiment.train():    
        model.fit(inputs_train, target_train, 
            early_stopping_rounds=5, 
            eval_set=[(inputs_train, target_train), (inputs_valid, target_valid)],
            verbose=True)
        
    with experiment.validate():
        y_pred = model.predict(inputs_valid)
        metrics = get_metrics(target_valid, y_pred)
        
        experiment.log_metrics(metrics)
    
    if baseline:
        # Compare scores from this run with the previous best and log
        relative_metrics = {key: value - baseline[key] for key, value in metrics.items()}
        experiment.log_metrics(relative_metrics)
        
    experiment.end()
    
def run_test(inputs_train, target_train, inputs_valid, target_valid, baseline=None):
    model = get_model()
    model.fit(inputs_train, target_train, 
            early_stopping_rounds=5, 
            eval_set=[(inputs_train, target_train), (inputs_valid, target_valid)],
            verbose=False)
        
    y_pred = model.predict(inputs_valid)
    metrics = get_metrics(target_valid, y_pred)
    
    if baseline:
        # Compare scores from this run with the previous best and log
        relative_metrics = {key: value - baseline[key] for key, value in metrics.items()}
        return metrics, relative_metrics
    else:
        return metrics
    
def compare_metrics(scores: dict, baseline: dict) -> None:
    """
    Plots metrics against one another relative to a provided baseline for visual comparison.
    
    Args:
        scores: metric values for each of the experiments.
        baseline: metric values for the baseline.
    """
    metric_labels = list(baseline.keys())
    xticks_len = len(scores) - 1
    _, ax = plt.subplots(ncols=1, nrows=len(metric_labels), figsize=(6, 12), sharex=True)
    for i, ax_ in enumerate(ax.flatten()):
        metric = metric_labels[i]
        values = [scores[key][metric] for key in scores]
        ax_.plot(values, color=COLORS[i], label=f"{metric.upper()}")
        ax_.plot([0, xticks_len], [baseline[metric], baseline[metric]], color=COLORS[4], ls='--', label=f"Baseline {metric.upper()}")
        ax_.set_ylabel(f"{metric.upper()}")
        ax_.legend()
    plt.xlabel("Number of features removed")
    plt.legend()
    plt.subplots_adjust(hspace=0.0)
    
    plt.show();

