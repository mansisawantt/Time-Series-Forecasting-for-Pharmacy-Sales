import pandas as pd
import os

def load_data(data_path):
    """Loads all sales datasets from the data folder."""
    datasets = {}
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_path, file))
            datasets[file] = df
    return datasets

def preprocess_data(df, date_col='date'):
    """Preprocesses the sales data by handling missing values and converting date columns."""
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def save_preprocessed_data(datasets, save_path):
    """Saves preprocessed datasets to the specified folder."""
    os.makedirs(save_path, exist_ok=True)
    for name, df in datasets.items():
        df.to_csv(os.path.join(save_path, name), index=False)

if __name__ == "__main__":
    data_path = "data/"
    save_path = "data/preprocessed/"
    datasets = load_data(data_path)
    
    # Preprocess each dataset
    for name, df in datasets.items():
        datasets[name] = preprocess_data(df)
    
    # Save preprocessed data
    save_preprocessed_data(datasets, save_path)
    print("Preprocessing complete. Saved in data/preprocessed/")