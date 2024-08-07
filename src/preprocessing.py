import src.utils.helper as helper

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Binarizer
from sklearn import set_config
set_config(transform_output = "pandas")

from feature_engine import encoding as ce

from sklego.preprocessing import RepeatingBasisFunction

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

def preprocess_data() -> pd.DataFrame:
    
    data, discrete, temporal, continuous, categorical = helper.load_base_data()
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