from dotenv import load_dotenv
import os

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Binarizer
from sklearn import set_config
set_config(transform_output = "pandas")

from feature_engine import encoding as ce

from sklego.preprocessing import RepeatingBasisFunction

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")

def make_feature_cloud_cover_score(df: pd.DataFrame) -> pd.Series:
    data = df.copy()
    
    data['VisibilityScore'] = np.select(
        [data['Visibility'] > 10, data['Visibility'] > 5, data['Visibility'] <= 5],
        [0, 1, 2]
    )
    data['CloudCeilingScore'] = np.select(
        [data['Cloud.Ceiling'] > 7.5, data['Cloud.Ceiling'] > 2.5, data['Cloud.Ceiling'] <= 2.5],
        [0, 1, 2]
    )

    data['CloudCoverScore'] = data['VisibilityScore'] + data['CloudCeilingScore']

    data = data.drop(columns=['VisibilityScore', 'CloudCeilingScore'], axis=1)
    
    return data['CloudCoverScore']

def make_feature_temp_div_hum(df: pd.DataFrame) -> pd.Series:
    data = df.copy()
    
    data['TempDivHum'] = data['AmbientTemp'] / data['Humidity']
    
    max_val = data['TempDivHum'].replace([np.inf, -np.inf], np.nan).dropna().max()
    
    data['TempDivHum'] = data['TempDivHum'].replace([np.inf, -np.inf], np.nan).fillna(max_val)
    
    return data['TempDivHum']

def load_base_data(path: str = DATA_PATH):
    
    def _fix_dates(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        if 'Datetime' in data.columns:
            return data
        
        data['Datetime'] = data.apply(
            lambda x: datetime.strptime(f"{x['Date']} {x['Time']}", "%Y%m%d %H%M"), axis=1)
        
        data.drop(['Date', 'Time', 'Month', 'Hour', 'YRMODAHRMI'], axis='columns', inplace=True)
        
        data = data[["Datetime"] + [col for col in list(data.columns) if col not in ["Datetime", 'PolyPwr']] + ['PolyPwr']]
        
        return data

    def _fix_units(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
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

    
    # Set up the data for running experiments
    raw = pd.read_csv(path)
    data = _fix_units(raw)
    data = _fix_dates(data)

    data['month'] = data['Datetime'].dt.month
    data['hour'] = data['Datetime'].dt.hour

    data = data.drop(columns='Datetime', axis=1)

    # Set up features
    temporal = ['month', 'hour']

    discrete = [col for col in data.columns if data[col].dtype != 'O' 
                and col != 'PolyPwr' and data[col].nunique() <= 20 and col not in temporal]

    continuous = [col for col in data.columns if data[col].dtype != 'O' 
                and col != 'PolyPwr' and col not in discrete + temporal]

    categorical = [col for col in data.columns if data[col].dtype == 'O']

    print(f"Discrete: {discrete}", f"Temporal: {temporal}", f"Continuous: {continuous}", f"Categorical: {categorical}", sep="\n")
    
    return data, discrete, temporal, continuous, categorical


def preprocess_data() -> pd.DataFrame:
    
    data, discrete, temporal, continuous, categorical = load_base_data()
    data['CloudCoverScore'] = make_feature_cloud_cover_score(data)
    data['TempDivHum'] = make_feature_temp_div_hum(data)
    data[['Altitude', 'CloudCoverScore']] = data[['Altitude', 'CloudCoverScore']].astype('O')
    
    # Re-order columns to fit with experiments.
    data = data[discrete + temporal + continuous + categorical + ['CloudCoverScore', 'TempDivHum'] + ['PolyPwr']]
    
    return data

def make_pipeline(model):
    # RepeatingBasisFunction and Binarizer rename the columns so we have to access them as below
    visibility_column_index = 19
    cloud_ceiling_column_index = 21
    temp_hum_ratio_column_index = 'remainder__x25'

    pipeline = Pipeline([
        
        ('rare_label_encoder', 
         ce.RareLabelEncoder(tol=0.05,
                             n_categories=4,
                             variables=['Location', 'CloudCoverScore'])),
        
        ('categorical_encoder',
         ce.OrdinalEncoder(encoding_method='ordered',
                           variables=['Season', 'Altitude'])),
        
        ('categorical_encoder_loc',
         ce.OrdinalEncoder(encoding_method='arbitrary',
                           variables=['Location', 'CloudCoverScore'])),
        
        ('rbf_month',
         RepeatingBasisFunction(remainder="passthrough",
                                n_periods=12,
                                column="month",
                                width=1.0,
                                input_range=(1,12))),
        
        ('binarizer_vis_cloudceil', 
         ColumnTransformer(transformers=[('binarize_vis', 
                                          Binarizer(threshold=15), 
                                          [visibility_column_index, cloud_ceiling_column_index])], 
                           remainder='passthrough')),
        
        ('binarizer_tempdivhum', 
         ColumnTransformer(transformers=[('binarize_tempdivhum', 
                                          Binarizer(threshold=4), 
                                          [temp_hum_ratio_column_index])], 
                           remainder='passthrough')),
        
        ('xgb', model)

    ])
    
    return pipeline