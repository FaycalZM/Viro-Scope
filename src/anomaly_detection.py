from data_prep import load_processed_data
from model_training import make_predictions, load_lstm_model, split_data, WINDOW_SIZE
from model_evaluation import plot_predictions
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


def detect_anomalies(actual, predictions, threshold=2):
    """
    Detect anomalies in a time series by comparing the residuals of predicted values with the actual values.

    Parameters:
    - actual: The actual values from the test set.
    - predictions: The predicted values by the model.
    - threshold: The number of standard deviations above the mean above which a value is considered an anomaly. Defaults to 2.

    Returns:
    - anomalies: A boolean array indicating whether each value is an anomaly.
    - residuals: The absolute difference between each actual value and its predicted value.
    - threshold: The number of standard deviations above the mean used to detect anomalies.

    """
    residuals = np.abs(actual - predictions)
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    anomalies = (residuals > mean_residual + std_residual * threshold)

    return anomalies, residuals, threshold


def plot_anomalies(dates, actual, predictions, anomalies):
    """
    Plot the actual data, predictions, and highlight anomalies.

    Parameters:
    - dates: The corresponding dates for the actual data.
    - actual: The actual values from the test set.
    - predictions: The predicted values by the model.
    - anomalies: A DataFrame of detected anomalies.
    """
    plt.figure(figsize=(15, 8))

    # Plot actual data and predictions
    plt.plot(dates, actual, label='Actual Data', color='blue')
    plt.plot(dates, predictions, label='Predictions',
             color='red', linestyle='dashed')

    # Highlight anomalies
    # Convert dates and actual to numpy arrays if they aren't already
    dates = np.array(dates)
    actual = np.array(actual)

    # Extract the anomalies indices from the anomalies dataframe
    anomalies_indices = np.where(anomalies)[0]

    # Plot the anomalies
    plt.scatter(dates[anomalies_indices], actual[anomalies_indices],
                color='orange', label='Anomalies', s=75, edgecolor='black')

    plt.title('Actual Data vs Predictions with Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Search Volume')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    data = load_processed_data()
    train_data, test_data = split_data(data)
    window_size = WINDOW_SIZE
    model = load_lstm_model(model_name='lstm_model.keras')
    predictions = make_predictions(model, test_data, window_size)
    dates = test_data.index[window_size:]
    actual = test_data[window_size:]
    anomalies, residuals, threshold = detect_anomalies(
        actual, predictions, threshold=2.0)
    plot_anomalies(dates, actual, predictions, anomalies)
