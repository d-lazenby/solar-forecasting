from preprocessing import preprocess_data, make_pipeline
from train import build_model, train_model, run_experiment

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import set_config
set_config(transform_output = "pandas")

def main(exp_name: str = None) -> None:
    
    X = preprocess_data()
    y = X.pop('PolyPwr')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0) 

    print(X_train.shape, X_test.shape)
    print(X_train.columns)
    
    pipeline = make_pipeline(build_model())
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
    # main(exp_name='test-api-key-insert')
    main()