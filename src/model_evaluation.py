from model_training import load_lstm_model, load_processed_data, make_predictions
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


def plot_predictions(dates, actual, predicted):

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


if __name__ == "__main__":
    data = load_processed_data()
    window_size = 30

    last_n_weeks = 18
    # Take the last 'window_size' values before the last time step as the test data
    # This is done so that the model can be evaluated on unseen data
    # test_data = data[-window_size-last_n_weeks:-last_n_weeks]
    test_data = data[3:window_size+3]
    model = load_lstm_model(model_name='lstm_model.keras')

    predictions = make_predictions(model, test_data, window_size=window_size)

    print(predictions)

    # # Generate a date range for the last 5 years with weekly intervals
    # dates = pd.date_range(end=pd.to_datetime('today').strftime('%Y-%m-%d'),
    #                       periods=len(test_data), freq='W')

    # # Assuming predictions have been generated for the test data
    # plot_predictions(dates, test_data, predictions)
    # print(test_data)
