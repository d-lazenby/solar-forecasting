from flask import Flask, request, jsonify, render_template

import pickle
import pandas as pd
from src.preprocessing import make_feature_cloud_cover_score, make_feature_temp_div_hum

def make_inference(data: dict) -> float:
    # Preprocess to add new columns from feature engineering
    data = pd.DataFrame([data])

    data['CloudCoverScore'] = make_feature_cloud_cover_score(data)
    data['TempDivHum'] = make_feature_temp_div_hum(data)
    data[['Altitude', 'CloudCoverScore']] = data[['Altitude', 'CloudCoverScore']].astype('O')

    # Need the features in this order for the feature engineering pipeline
    ordered_features = ['Latitude', 'Longitude', 'Altitude', 'month', 'hour', 'Humidity', 
                        'AmbientTemp', 'Wind.Speed', 'Visibility', 'Pressure', 'Cloud.Ceiling',
                        'Location', 'Season', 'CloudCoverScore', 'TempDivHum']

    X = data[ordered_features].copy()

    # Make inference
    y_pred = pipeline.predict(X)
    
    return y_pred[0]


input_file = "./models/final_model.bin"

app = Flask('inference')

with open(input_file, 'rb') as f_input:
    pipeline = pickle.load(f_input)


@app.route("/")
def home():
    return render_template("index.html")

@app.route('/inference', methods=['POST'])
def predict():
    data = request.get_json()
    
    y_pred = make_inference(data)
    
    result = {
        "power": float(y_pred)
    }
    
    return render_template('index.html', prediction=jsonify(result))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)