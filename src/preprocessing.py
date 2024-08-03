import utils.eda as eda
import utils.helper as helper
import train

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import set_config
set_config(transform_output = "pandas")

from feature_engine.outliers import Winsorizer
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
    
    data, _, _, _, _ = helper.load_base_data()
    data['CloudCoverScore'] = make_feature_cloud_cover_score(data)
    data['TempDivHum'] = make_feature_temp_div_hum(data)
    data[['Altitude', 'CloudCoverScore']] = data[['Altitude', 'CloudCoverScore']].astype('O')
    
    return data

def make_pipeline(model):
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

    # X = data[discrete + temporal + continuous + categorical + ['CloudCoverScore', 'TempDivHum']].copy()
    # X['Altitude'] = X['Altitude'].astype('O')
    # X['CloudCoverScore'] = X['CloudCoverScore'].astype('O')

    # y = data['PolyPwr'].copy()

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X,  # predictors
    #     y,  # target
    #     train_size=0.8,  # percentage of obs in train set
    #     random_state=0)  # seed to ensure reproducibility

    # print(X_train.shape, X_test.shape)
    # print(X_train.columns)

if __name__ == '__main__':
    data = preprocess_data()
    pipeline = make_pipeline(train.build_model())
    print(pipeline.named_steps)
    

# X = data[discrete + temporal + continuous + categorical + ['CloudCoverScore', 'TempDivHum']].copy()
# X['Altitude'] = X['Altitude'].astype('O')
# X['CloudCoverScore'] = X['CloudCoverScore'].astype('O')

# y = data['PolyPwr'].copy()

# X_train, X_test, y_train, y_test = train_test_split(
#     X,  # predictors
#     y,  # target
#     train_size=0.8,  # percentage of obs in train set
#     random_state=0)  # seed to ensure reproducibility

# print(X_train.shape, X_test.shape)
# print(X_train.columns)

# visibility_column_index = 19
# cloud_ceiling_column_index = 21
# temp_hum_ratio_column_index = 'remainder__x25'

# pipeline = Pipeline([
    
#     ('rare_label_encoder', 
#      ce.RareLabelEncoder(tol=0.05,
#                          n_categories=4,
#                          variables=['Location', 'CloudCoverScore'])),
    
#     ('categorical_encoder',
#      ce.OrdinalEncoder(encoding_method='ordered',
#                        variables=['Season', 'Altitude'])),
    
#     ('categorical_encoder_loc',
#      ce.OrdinalEncoder(encoding_method='arbitrary',
#                        variables=['Location', 'CloudCoverScore'])),
    
#     ('rbf_month',
#      RepeatingBasisFunction(remainder="passthrough",
#                             n_periods=12,
#                             column="month",
#                             width=1.0,
#                             input_range=(1,12))),
    
#     ('binarizer_vis_cloudceil', 
#      ColumnTransformer(transformers=[('binarize_vis', 
#                                       Binarizer(threshold=15), 
#                                       [visibility_column_index, cloud_ceiling_column_index])], 
#                        remainder='passthrough')),
    
#     ('binarizer_tempdivhum', 
#      ColumnTransformer(transformers=[('binarize_tempdivhum', 
#                                       Binarizer(threshold=4), 
#                                       [temp_hum_ratio_column_index])], 
#                        remainder='passthrough')),
    
#     ('xgb', train.build_model())

# ])

# # def load_data(random_state=0):
# #     data = pd.read_csv('data/base_features.csv')
    
# #     y = data.pop('Power (W)')
# #     X = data.copy()
    
# #     # Drop Altitude as it is perfectly correlated with pressure
# #     X = X.drop('Altitude (m)', axis='columns')
    
# #     # Fix datatypes
# #     X[['month_of_year', 'hour_of_day']] = X[['month_of_year', 'hour_of_day']].astype('int8')
# #     X[['Season', 'Location']] = X[['Season', 'Location']].astype('category')
    
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random_state)

# #     print(f"Training size: {X_train.shape}\nTest size: {X_test.shape}")
    
# #     return X_train, y_train, X_test, y_test

# # # Feature engineering functions
# # def label_encode(inputs):
# #     X = inputs.copy()
# #     for col in X.select_dtypes(["category"]):
# #         X[col] = X[col].cat.codes
# #     return X

# # def interactions(inputs, cat_col, num_col):
# #     X = pd.get_dummies(inputs[cat_col], prefix=num_col)
# #     X = X.mul(inputs[num_col], axis=0)
# #     return X

# # def group_transforms(inputs, new_col_name, cat_col, num_col, transform):
# #     X = pd.DataFrame()
# #     X[new_col_name] = inputs.groupby(cat_col)[num_col].transform(transform)
# #     # X["MedLocTemp"] = inputs.groupby("Location")["AmbientTemp (deg C)"].transform("median")
# #     return X

# # def drop_features(inputs, col_names):
# #     X = inputs.copy()
# #     X = X.drop(columns=col_names)
# #     return X


# # # Feature pipeline
# # def create_features(inputs, args):
# #     X = inputs.copy()
    
# #     # X = drop_features(X, ['Location'])
    
# #     # Interaction columns
# #     # X = X.join(interactions(X, 'Location', 'AmbientTemp (deg C)'))
# #     # X = X.join(interactions(X, 'Season', 'AmbientTemp (deg C)'))
    
# #     # Group transforms
# #     # X = X.join(group_transforms(X, *args))
    
# #     # Label encode categorical features
# #     X = label_encode(X)
# #     print(f"\nFeature space after engineering: {X.shape}")
# #     return X
