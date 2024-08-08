from preprocessing import preprocess_data, make_pipeline
from train import build_model, train_model, run_experiment

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import set_config
set_config(transform_output = "pandas")

TRAIN_SIZE = 0.8

def main(exp_name: str = None) -> None:
    
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
    
    print(f"Mean R2: {scores['mean']}", f"Std: {scores['std']}")
    
    y_pred = pipeline.predict(X_test)
    test_score = r2_score(y_test, y_pred)
    print(f"Test score: {test_score}")
        
if __name__ == '__main__':
    main()