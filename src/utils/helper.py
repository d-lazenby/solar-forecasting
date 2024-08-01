from datetime import datetime
import numpy as np
import pandas as pd

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

def load_base_data():
    # Set up the data for running experiments
    raw = pd.read_csv("../data/raw.csv")
    data = fix_units(raw)
    data = fix_dates(data)

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
