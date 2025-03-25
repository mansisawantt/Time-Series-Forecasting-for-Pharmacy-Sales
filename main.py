from src.model import train_sarima_model, save_model
import pandas as pd
import os

# Define paths
data_path = "data/preprocessed/"
sales_file = os.path.join(data_path, "salesmonthly.csv")

# Load sales data
if not os.path.exists(sales_file):
    print(f" Error: {sales_file} not found!")
    exit(1)

sales_data = pd.read_csv(sales_file)

# Ensure 'datum' column exists
if "datum" not in sales_data.columns:
    print(" Warning: 'datum' column missing in salesmonthly.csv! Exiting...")
    exit(1)

sales_data["datum"] = pd.to_datetime(sales_data["datum"])
sales_data.set_index("datum", inplace=True)

# Medicine categories
categories = ["M01AB", "M01AE", "N02BA", "N02BE"]

for category in categories:
    if category in sales_data.columns:
        model = train_sarima_model(sales_data, category)
        save_model(model, category)
    else:
        print(f" Warning: Data for {category} not found in salesmonthly.csv!")
