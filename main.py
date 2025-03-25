from src.preprocess import load_data, preprocess_data, save_preprocessed_data
from src.model import train_sarima_model, save_model
import os

# Define paths
data_path = "data/"
preprocessed_path = "data/preprocessed/"
model_save_path = "models/sales_forecast.pkl"

if __name__ == "__main__":
    # Step 1: Load and Preprocess Data
    datasets = load_data(data_path)
    for name, df in datasets.items():
        datasets[name] = preprocess_data(df)
    save_preprocessed_data(datasets, preprocessed_path)
    print("Data preprocessing complete.")
    
    # Step 2: Train Model
    monthly_sales = datasets.get("salesmonthly.csv")
    if monthly_sales is not None:
        model = train_sarima_model(monthly_sales)
        save_model(model, model_save_path)
        print("Model training complete. Saved at models/sales_forecast.pkl")
    else:
        print("Monthly sales data not found! Model training skipped.")