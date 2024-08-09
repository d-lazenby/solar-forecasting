from comet_ml import Experiment
from dotenv import load_dotenv
import os

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold

from sklearn.pipeline import Pipeline

import xgboost
from xgboost import XGBRegressor

def build_model(params: dict = {'objective': 'reg:squarederror',
                                'n_estimators': 200,
                                'learning_rate': 0.1,
                                'early_stopping_rounds': 10}) -> XGBRegressor:
    
    model = XGBRegressor(**params, random_state=0)
        
    return model

def train_model(inputs: pd.core.frame.DataFrame, 
                target: pd.Series, 
                pipeline: Pipeline, 
                n_splits: int = 5, 
                random_state: int = 42) -> dict:
    
    X = inputs.copy()
    y = target.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    print(f"Training on {n_splits} folds", "*"*40, sep="\n")
    
    test_scores = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        # Split train set into train and validation
        X_train_fold, X_val, y_train_fold, y_val = train_test_split(X_train_fold, y_train_fold, test_size=0.2, random_state=random_state)
        
        # Transform eval_set
        val_pipeline = pipeline[:-1]
        val_pipeline.fit(X_train_fold, y_train_fold)
        X_val = val_pipeline.transform(X_val)
        
        # Fit model on train fold and use validation for early stopping
        pipeline.fit(X_train_fold, y_train_fold, xgb__eval_set=[(X_val, y_val)], xgb__verbose=False)

        # Predict on test set
        y_pred = pipeline.predict(X_test_fold)
        test_score = r2_score(y_test_fold, y_pred)
        test_scores.append(test_score)
        
        print(f"R2 score on fold {fold+1}: {test_score}")
    
    print("*"*40)
        
    scores = {'r2': test_scores,
              'mean': np.mean(test_scores),
              'std': np.std(test_scores)}
    
    return scores


def run_experiment(exp_name: str, 
                   inputs: pd.core.frame.DataFrame, 
                   target: pd.Series, 
                   pipeline: Pipeline,
                   ) -> dict:
    
    load_dotenv()
    COMET_API_KEY = os.getenv("COMET_API_KEY")
    
    experiment = Experiment(
        api_key=COMET_API_KEY,
        project_name="solar-forecasting",
        workspace="d-lazenby"
        )
    
    experiment.set_name(exp_name)
    
    with experiment.train():    
        scores = train_model(inputs=inputs, target=target, pipeline=pipeline)
        experiment.log_metrics({'mean': scores['mean'], 'std': scores['std']})
    
    experiment.end()
    return scores
