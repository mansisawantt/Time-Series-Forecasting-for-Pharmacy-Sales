from flask import Flask, render_template, request, jsonify
import os
import pickle
import pandas as pd

app = Flask(__name__)

# Path where trained models are stored
MODEL_PATH = "models/"
CATEGORIES = ["M01AB", "M01AE", "N02BA", "N02BE"]

# Load models into a dictionary
loaded_models = {}
for category in CATEGORIES:
    model_file = os.path.join(MODEL_PATH, f"{category}_model.pkl")
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            loaded_models[category] = pickle.load(f)
    else:
        print(f"⚠️ Warning: Model for {category} not found!")

@app.route("/")
def home():
    """Render the HTML page."""
    return render_template("index.html", categories=CATEGORIES)

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    category = request.form.get("category")  # Get category from dropdown
    periods = int(request.form.get("periods", 12))  # Get forecast periods

    if not category or category not in loaded_models:
        return jsonify({"error": f"Model for category {category} not found!"})

    # Load the model for the selected category
    model = loaded_models[category]

    # Generate forecast
    forecast = model.get_forecast(steps=periods)
    forecast_values = forecast.predicted_mean.tolist()

    return jsonify({"category": category, "forecast": forecast_values})

if __name__ == "__main__":
    app.run(debug=True)
