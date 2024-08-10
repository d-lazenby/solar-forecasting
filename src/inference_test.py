import requests

test_reading = {"Location": "Camp Murray",
                "Latitude": 47.11,
                "Longitude": -122.57,
                "Altitude": 25.480964821029243,
                "Season": "Winter",
                "Humidity": 81.71997,
                "AmbientTemp": 12.86919,
                "Wind.Speed": 8.053963638825048,
                "Visibility": 16.096494845360827,
                "Pressure": 1010.6,
                "Cloud.Ceiling": 22.028989750451927,
                "month": 12,
                "hour": 11}

url = "http://localhost:9696/inference"

response = requests.post(url=url, json=test_reading).json()
print(response)