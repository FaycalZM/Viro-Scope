import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from data_ingestion import load_raw_data


def preprocess_data(data: pd.DataFrame):
    # Fill missing values
    data = data.fillna(method='ffill').fillna(method='bfill')

    # Scale data if necessary (Min-Max scaling)
    scaled_data = (data - data.min()) / (data.max() - data.min())
    return scaled_data


def save_preprocessed_data(data: pd.DataFrame, file_name='processed_trends_data.csv'):
    processed_data_dir = os.path.join('data', 'processed')
    os.makedirs(processed_data_dir, exist_ok=True)
    file_path = os.path.join(processed_data_dir, file_name)
    data['flu'].to_csv(file_path)
    print(f"Processed data saved to {file_path}")


def load_processed_data(file_name='processed_trends_data.csv'):
    file_path = os.path.join('data', 'processed', file_name)
    return pd.read_csv(file_path, parse_dates=True, index_col='date')


if __name__ == "__main__":
    data = load_raw_data()
    processed_data = preprocess_data(data)
    save_preprocessed_data(processed_data)
