from model_training import load_lstm_model, load_processed_data, split_data
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


def evaluate_model_performance(model, test_data, window_size):
    """
    Evaluate the model's performance on the test set.

    Parameters:
    - model: Trained LSTM model.
    - test_data: Test data for evaluation.
    - window_size: The number of previous time steps used as input for prediction.

    Returns:
    - predictions: Predicted values for the test set.
    - Y_test: Actual values for the test set.
    - mse: Mean Squared Error of the predictions.
    - mae: Mean Absolute Error of the predictions.
    """

    test_data = test_data.values if hasattr(
        test_data, 'values') else np.array(test_data)

    # Prepare test data for prediction
    X_test, Y_test = [], []
    for i in range(window_size, len(test_data)):
        X_test.append(test_data[i-window_size:i, 0])
        Y_test.append(test_data[i, 0])

    X_test = np.array(X_test).reshape(-1, window_size, 1)
    Y_test = np.array(Y_test)

    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(Y_test, predictions)
    mae = mean_absolute_error(Y_test, predictions)

    return predictions, Y_test, mse, mae


def plot_predictions(dates, predictions, actual):
    """
    Plot the model's predictions against the actual values.

    Parameters:
    - dates: The corresponding dates for the test set.
    - predictions: Predicted values for the test set.
    - actual: Actual values for the test set.
    """
    # Adjust the predictions and actual data to align with the original dataset
    plt.figure(figsize=(14, 7))

    plt.plot(dates, actual, label='Actual Data', color='blue')
    plt.plot(dates, predictions, label='Predictions',
             color='red', linestyle='dashed')

    plt.title('Actual vs Predicted Search Volume')
    plt.xlabel('Date')
    plt.ylabel('Search Volume')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    data = load_processed_data()
    train_data, test_data = split_data(data)
    window_size = 60
    dates = test_data.index[window_size:]

    model = load_lstm_model(model_name='lstm_model.keras')
    predictions, actual, mse, mae = evaluate_model_performance(
        model, test_data, window_size)

    plot_predictions(dates, predictions, actual)
