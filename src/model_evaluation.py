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
    - data: The full dataset (both training and test data).
    - window_size: The number of previous time steps used as input for prediction.

    Returns:
    - predictions: Predicted values for the test set.
    - actual: Actual values for the test set.
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


# def plot_predictions(dates, actual, predicted):

    dates = np.array(dates)

    plt.figure(figsize=(14, 7))

    # Plot the actual values
    plt.plot(actual,
             label='Actual Values', color='blue')

    # Plot the predicted values
    plt.plot(predicted,
             label='Predicted Values', color='red', linestyle='dashed')

    # # Mark the training-test split
    # plt.axvline(x=dates, color='green',
    #             linestyle='--', label='Train-Test Split')

    plt.title('Actual vs Predicted Search Volumes on Test Data')
    plt.xlabel('Date')
    plt.ylabel('Search Volume')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predictions(predictions, actual, train_data, test_data, window_size):
    """
    Plot the model's predictions against the actual values.

    Parameters:
    - predictions: Predicted values for the test set.
    - actual: Actual values for the test set.
    - train_data: The training data.
    - test_data: The test data.
    - window_size: The number of previous time steps used as input for prediction.
    """
    # Adjust the predictions and actual data to align with the original dataset
    prediction_plot = np.empty_like(test_data)
    prediction_plot[:, :] = np.nan
    prediction_plot[window_size:len(
        predictions) + window_size, :] = predictions

    # Plotting the results
    plt.figure(figsize=(15, 8))
    plt.plot(range(len(train_data)), train_data, label='Training Data')
    plt.plot(range(len(train_data), len(train_data) + len(test_data)),
             test_data, label='Actual Data', color='blue')
    plt.plot(range(len(train_data), len(train_data) + len(test_data)),
             prediction_plot, label='Predictions', color='red')
    plt.title('Model Predictions vs Actual Data')
    plt.xlabel('Time')
    plt.ylabel('Search Volume')
    plt.legend()
    plt.show()

# Example usage:
# plot_predictions(predictions, actual, train_data, test_data, window_size)


if __name__ == "__main__":
    data = load_processed_data()
    train_data, test_data = split_data(data)
    window_size = 60

    # # Take the last 'window_size' values before the last time step as the test data
    # # This is done so that the model can be evaluated on unseen data
    # test_data = data[:window_size]
    model = load_lstm_model(model_name='lstm_model.keras')
    predictions, actual, mse, mae = evaluate_model_performance(
        model, test_data, window_size)

    plot_predictions(predictions, actual, train_data, test_data, window_size)
    # # Generate a date range for the last 5 years with weekly intervals
    # dates = pd.date_range(end=pd.to_datetime('today').strftime('%Y-%m-%d'),
    #                       periods=len(test_data), freq='W')

    # # Assuming predictions have been generated for the test data
    # plot_predictions(dates, test_data, predictions)
    # print(test_data)
