import pandas as pd
import os
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Define paths
data_path = "data/preprocessed/"
model_save_path = "models/"
os.makedirs(model_save_path, exist_ok=True)  # Ensure models folder exists

def train_sarima_model(data, category):
    """Train SARIMA model for a given category."""
    print(f" Training model for {category}...")

    model = SARIMAX(data[category], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()

    print(f" Model for {category} trained successfully!")
    return model_fit

def save_model(model, category):
    """Save trained SARIMA model to a file."""
    model_filename = f"{category}_model.pkl"
    with open(os.path.join(model_save_path, model_filename), 'wb') as f:
        pickle.dump(model, f)

    print(f"💾 Model for {category} saved as {model_filename}")

# Load sales data
sales_file = os.path.join(data_path, "salesmonthly.csv")
if not os.path.exists(sales_file):
    print(f" Error: {sales_file} not found!")
    exit(1)

sales_data = pd.read_csv(sales_file)

# Ensure 'datum' column exists and is in datetime format
if "datum" not in sales_data.columns:
    print(" Warning: 'datum' column missing in salesmonthly.csv! Exiting...")
    exit(1)

sales_data["datum"] = pd.to_datetime(sales_data["datum"])
sales_data.set_index("datum", inplace=True)

# Medicine categories
categories = ["M01AB", "M01AE", "N02BA", "N02BE"]

# Track if any model was trained
any_trained = False

for category in categories:
    if category in sales_data.columns:
        model = train_sarima_model(sales_data, category)
        save_model(model, category)
        any_trained = True
    else:
        print(f" Warning: Data for {category} not found in salesmonthly.csv!")

if not any_trained:
    print(" No valid categories found in salesmonthly.csv! Exiting...")
