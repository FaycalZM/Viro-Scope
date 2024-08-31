import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from io import StringIO
from fluvio import Fluvio, Offset
from data_ingestion import load_raw_data
from utils.utils import data_chunking


CONSUMER_TOPIC = "google-trends-data"
PRODUCER_TOPIC = "preprocessed-trends-data"


def preprocess_data(data: pd.DataFrame):
    # Fill missing values
    data = data.fillna(method='ffill').fillna(method='bfill')

    # Scale data (Min-Max scaling)
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
    fluvio = Fluvio.connect()
    consumer = fluvio.partition_consumer(CONSUMER_TOPIC, 0)
    producer = fluvio.topic_producer(PRODUCER_TOPIC)

    raw_data_json = ""
    stream = consumer.stream(Offset.from_beginning(0))
    # continuous streaming
    for record in stream:
        if record.value_string() == "done":
            # convert json back to DataFrame
            data = pd.read_json(raw_data_json)
            # preprocess raw data
            processed_data = preprocess_data(data)
            save_preprocessed_data(processed_data)
            # send it to the corresponding topic
            processed_data_json = processed_data.reset_index().to_json(
                orient='records', date_format="iso")
            string_chunks = data_chunking(processed_data_json)
            for chunk in string_chunks:
                # push data to the topic
                producer.send_string(chunk)
                # flush the last entry
            producer.flush()
            producer.send_string("done")
            print(f"Preprocessed data pushed to {PRODUCER_TOPIC} topic")

            # reset raw_data_json
            raw_data_json = ""
        else:
            raw_data_json += record.value_string()
