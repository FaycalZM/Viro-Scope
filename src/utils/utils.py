from keras._tf_keras.keras.models import load_model
import os


BUFFER_MAX_SIZE = 16304


def data_chunking(data_json: str) -> list[str]:
    # the buffer max_size is 16384 bytes, so it's not possible to send the data directly -> divide and conquer
    # convert string data to bytes
    byte_data = data_json.encode('utf-8')

    # split the bytes data into chunks of chunk_size bytes
    chunk_size = BUFFER_MAX_SIZE
    byte_chunks = [byte_data[i:i + chunk_size]
                   for i in range(0, len(byte_data), chunk_size)]

    # convert each chunk back to string
    string_chunks = [chunk.decode('utf-8', errors='ignore')
                     for chunk in byte_chunks]

    return string_chunks

# Split the dataset into training and testing sets


def split_data(data, train_size=0.8):
    train_len = int(len(data) * train_size)
    train_data = data[:train_len]
    test_data = data[train_len:]
    return train_data, test_data


def load_lstm_model(model_name='lstm_model.keras'):
    model_path = os.path.join('data', 'models', model_name)
    model = load_model(model_path)
    return model
