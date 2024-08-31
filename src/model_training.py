from data_prep import load_processed_data
from model_evaluation import evaluate_model_performance, plot_predictions
import os
import json
from keras.src.layers import Dense, LSTM, Dropout
from keras.src.models.sequential import Sequential
from keras.src.optimizers import Adam
from keras._tf_keras.keras.models import load_model
from fluvio import Fluvio, Offset
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from io import StringIO
from utils.utils import data_chunking, split_data, load_lstm_model
matplotlib.use('Qt5Agg')


WINDOW_SIZE = 60
CONSUMER_TOPIC = "preprocessed-trends-data"
PRODUCER_TOPIC = "predictions-data"


# Build and compile the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.add(Dropout(0.2))
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


# Prepare the data for LSTM input by creating sequences
def prepare_data_for_lstm(data, window_size):
    data = data.values if hasattr(data, 'values') else np.array(data)
    X, Y = [], []
    # enough data points to form a single input sequence for the model to predict the next time step
    if len(data) <= window_size:
        X.append(data)
        # no next step => no Y values
    # multiple input sequences
    else:
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, 0])
            Y.append(data[i, 0])

    X = np.array(X)

    if X.ndim == 2:
        X = X.reshape(X.shape[0], X.shape[1], 1)

    Y = np.array(Y) if Y else None
    return X, Y


def train_model(data, window_size=WINDOW_SIZE, epochs=60, batch_size=32):
    X, Y = prepare_data_for_lstm(data, window_size)
    model = create_lstm_model((X.shape[1], 1))
    model.fit(X, Y, epochs=epochs, batch_size=batch_size)
    print(f"Model trained with {epochs} epochs and {batch_size} batch size")
    return model


def save_model(model, filename='lstm_model.keras'):
    model_dir = os.path.join('data', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)
    model.save(model_path)
    print(f"Model saved to {model_path}")


def make_predictions(model, data, window_size=WINDOW_SIZE):
    if window_size > len(data):
        raise ValueError(
            "Window size cannot be larger than the length of the data")
    X, _ = prepare_data_for_lstm(data, window_size)
    return model.predict(X)


if __name__ == "__main__":
    # data = load_processed_data()

    # #  split data
    # train_data, test_data = split_data(data)

    # window_size = WINDOW_SIZE
    # model = train_model(data=train_data, window_size=window_size)
    # save_model(model)

    fluvio = Fluvio.connect()
    consumer = fluvio.partition_consumer(CONSUMER_TOPIC, 0)
    producer = fluvio.topic_producer(PRODUCER_TOPIC)

    processed_data_json = ""
    stream = consumer.stream(Offset.from_beginning(0))
    # continuous streaming
    for record in stream:
        if record.value_string() == "done":
            # convert json back to DataFrame
            data = pd.read_json(
                StringIO(processed_data_json), orient='records')
            if 'index' in data.columns:
                data = data.drop(columns='index')
            train_data, test_data = split_data(data)
            # train the model
            window_size = WINDOW_SIZE
            model = train_model(data=train_data, window_size=window_size)
            save_model(model)
            # model = load_lstm_model('lstm_model.keras')
            # evaluate model performance
            predictions, actual, mse, mae = evaluate_model_performance(
                model, test_data, window_size)
            # convert predictions, actual data to json string and send it to the corresponding topic
            predictions_json = json.dumps(np.array(predictions).tolist())
            actual_json = json.dumps(actual.tolist())
            output = json.loads(
                '{"predictions" : [],"actual":[],"mse":0,"mae":0}')
            output['predictions'] = predictions_json
            output['actual'] = actual_json
            output['mse'] = mse
            output['mae'] = mae
            output = json.dumps(output)
            string_chunks = data_chunking(output)
            for chunk in string_chunks:
                # push data to the topic
                producer.send_string(chunk)
                # flush the last entry
            producer.flush()
            producer.send_string("done")
            print(f"model evaluation results pushed to {PRODUCER_TOPIC} topic")

            # reset raw_data_json
            processed_data_json = ""
        else:
            processed_data_json += record.value_string()
