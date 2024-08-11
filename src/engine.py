import os

from preprocessing import preprocess_data, make_pipeline
from train import build_model, train_model, run_experiment

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import set_config
set_config(transform_output = "pandas")

import pickle

TRAIN_SIZE = 0.8
MODEL_PATH = "models/final_model.bin"

def main(exp_name: str = None, save_model: bool = False) -> None:
    
    X = preprocess_data()
    y = X.pop('PolyPwr')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=0) 

    print(X_train.shape, X_test.shape)
    print(X_train.columns)
    
    params = {'eta': 0.02532227240864557,
              'max_depth': 9,
              'subsample': 0.5701902725252715,
              'colsample_bytree': 0.5700791751417876,
              'gamma': 4.468310078336034,
              'min_child_weight': 9.562538361029706,
              'lambda': 7.292064154574558,
              'alpha': 9.313263852489994,
              'objective': 'reg:squarederror',
              'n_estimators': 100000,
              'early_stopping_rounds': 100}

    pipeline = make_pipeline(build_model(params=params))
    print(pipeline.named_steps)
    
    if not exp_name:
        scores = train_model(inputs=X_train, target=y_train, pipeline=pipeline)
    else:
        scores = run_experiment(inputs=X_train, target=y_train, pipeline=pipeline, exp_name=exp_name)
    
    print(f"Mean R2: {scores['mean']} ± {scores['std']}\n")
    
    # Train final model on full training set
    print("Conducting final training run...")
    
    pipeline = make_pipeline(build_model(params=params))
    
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
        
    # Transform eval_set
    val_pipeline = pipeline[:-1]
    val_pipeline.fit(X_train_final, y_train_final)
    X_val = val_pipeline.transform(X_val)
    
    pipeline.fit(X_train_final, y_train_final, xgb__eval_set=[(X_val, y_val)], xgb__verbose=False)
    
    # Predict on test set
    y_pred = pipeline.predict(X_test)
    test_score = r2_score(y_test, y_pred)
    
    print(f"Final Test R2: {test_score}")

    if save_model:
        # saving model
        if not os.path.exists('models'):
            os.makedirs('./models')
        
        with open(MODEL_PATH, 'wb') as f_out:
            pickle.dump(pipeline, f_out)
            
        print(f"Model saved at {MODEL_PATH}")
                
if __name__ == '__main__':
    main(save_model=True)