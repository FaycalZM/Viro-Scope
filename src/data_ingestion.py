from pytrends.request import TrendReq
import pandas as pd
from dateutil import parser
import datetime
import os


def fetch_google_trends_data(keywords, timeframe='today 5-y', geo='', category=0) -> pd.DataFrame:
    pytrends = TrendReq(hl='en-US')
    pytrends.build_payload(keywords, cat=category,
                           timeframe=timeframe, geo=geo, gprop='')

    # Get interest overtime
    data = pytrends.interest_over_time()
    if 'isPartial' in data.columns:
        data = data.drop(columns=['isPartial'])

    daily_data = data.resample('D').interpolate(method='linear')

    return daily_data


def save_raw_data(data: pd.DataFrame, file_name='google_trends_data.csv'):
    raw_data_dir = os.path.join('data', 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)
    file_path = os.path.join(raw_data_dir, file_name)
    data.to_csv(file_path)
    print(f"Data saved to {file_path}")


def load_raw_data(file_name='google_trends_data.csv'):
    file_path = os.path.join('data', 'raw', file_name)
    return pd.read_csv(file_path, parse_dates=True, index_col='date')


if __name__ == "__main__":
    keywords = ["flu"]
    data = fetch_google_trends_data(keywords)
    save_raw_data(data)
