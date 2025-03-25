from flask import Flask, request, jsonify
import pickle
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)

# Load trained model
model_path = "models/sales_forecast.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route("/")  # This will fix the issue
def home():
    return "Welcome to the Medicine Sales Prediction API! Use the /predict endpoint to get predictions."

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({"message": "Use a POST request to get predictions."})

    try:
        data = request.get_json()
        periods = int(data.get("periods", 12))
        forecast = model.get_forecast(steps=periods)
        forecast_values = forecast.predicted_mean.tolist()
        return jsonify({"forecast": forecast_values})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
