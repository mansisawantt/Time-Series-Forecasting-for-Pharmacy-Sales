import pandas as pd
import os
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX

def load_preprocessed_data(data_path):
    """Loads preprocessed sales datasets."""
    datasets = {}
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_path, file))
            datasets[file] = df
    return datasets

def train_sarima_model(df, date_col='datum', sales_col='M01AB'):  
    """Trains a SARIMA model for time series forecasting."""
    print("Columns in dataset:", df.columns)  
    
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    
    model = SARIMAX(df[sales_col], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()
    return model_fit

def save_model(model, model_path):
    """Saves the trained model."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    data_path = "data/preprocessed/"
    model_save_path = "models/sales_forecast.pkl"
    
    datasets = load_preprocessed_data(data_path)
    
    # Train on monthly sales data
    monthly_sales = datasets.get("salesmonthly.csv")
    if monthly_sales is not None:
        model = train_sarima_model(monthly_sales)
        save_model(model, model_save_path)
        print("Model training complete. Saved at models/sales_forecast.pkl")
    else:
        print("Monthly sales data not found!")
