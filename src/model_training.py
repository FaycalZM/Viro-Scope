from data_prep import load_processed_data
import os
from keras.src.layers import Dense, LSTM, Dropout
from keras.src.models.sequential import Sequential
from keras.src.optimizers import Adam
# from keras.src.models import load_model
# from tensorflow.python.keras.models import load_model
from keras._tf_keras.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


# Split the dataset into training and testing sets
def split_data(data, train_size=0.8):
    train_len = int(len(data) * train_size)
    train_data = data[:train_len]
    test_data = data[train_len:]
    return train_data, test_data


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
    if len(data) == window_size:
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


def train_model(data, window_size=60, epochs=100, batch_size=32):
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


def load_lstm_model(model_name='lstm_model.keras'):
    model_path = os.path.join('data', 'models', model_name)
    model = load_model(model_path)
    return model


def make_predictions(model, data, window_size=30):
    if window_size > len(data):
        raise ValueError(
            "Window size cannot be larger than the length of the data")
    X, _ = prepare_data_for_lstm(data, window_size)
    return model.predict(X)


if __name__ == "__main__":
    data = load_processed_data()

    #  split data
    train_data, test_data = split_data(data)

    window_size = 60
    model = train_model(data=train_data, window_size=window_size)
    save_model(model)
