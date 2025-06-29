import warnings

from arima import *  # folosim arima_model(currency)
from matplotlib import pyplot
from pandas import read_csv
from rnn import *    # folosim rnn_model(currency)

# Hyperparametri globali (pentru plot)
TRAINING_PERCENTAGE = 0.7
TESTING_PERCENTAGE = 1 - TRAINING_PERCENTAGE
NUMBER_OF_PREVIOUS_DATA_POINTS = 3
LENGTH_DATA_SET = 0
TRAINING_SET_LENGTH = 0
TESTING_SET_LENGTH = 0

def plot(currency, raw_data, test_actual, testing_predict_rnn, testing_predict_arima, file_name):
    # Plot zona de test
    pyplot.figure(figsize=(10, 6))

    pyplot.plot(test_actual, label="Actual test data", color="blue")

    pyplot.plot(testing_predict_rnn, label="Testing prediction RNN", color="red")

    pyplot.plot(testing_predict_arima, label="Testing prediction ARIMA", color="green")

    pyplot.ylabel('currency values for 1 USD')
    pyplot.xlabel('number of days (TEST zone)')
    pyplot.title(f'USD/{currency} : TEST zone - RNN vs ARIMA vs Actual')

    pyplot.legend()
    pyplot.tight_layout()
    pyplot.savefig(file_name)
    pyplot.clf()

def main():
    global LENGTH_DATA_SET, TRAINING_SET_LENGTH, TESTING_SET_LENGTH

    # Load dataset header to extract currency options
    data_set_frame = read_csv(r'C:\Users\Renata\Desktop\Re-course\python-predictia\currency_prediction_data_set.csv',
                              header=0, index_col=0)
    column_headers = str([cur[4:] for cur in data_set_frame.columns.values.tolist()])
    currency = input('Enter any one of ' + column_headers + ' currencies \n').strip()

    # Load raw data pentru dimensiuni
    data_set_frame = read_csv(r'C:\Users\Renata\Desktop\Re-course\python-predictia\currency_prediction_data_set.csv',
                              header=0, index_col=0)
    raw_data_full = data_set_frame['USD/' + currency.upper()].values.tolist()

    LENGTH_DATA_SET = len(raw_data_full)
    TRAINING_SET_LENGTH = int(LENGTH_DATA_SET * TRAINING_PERCENTAGE)
    TESTING_SET_LENGTH = LENGTH_DATA_SET - TRAINING_SET_LENGTH

    #  Run ARIMA model
    print('\n--- Running ARIMA model ---')
    raw_data_arima, testing_predict_arima = arima_model(currency)

    #  Run RNN model
    print('\n--- Running RNN model ---')
    training_predict_rnn, testing_predict_rnn = rnn_model(currency)

    # Construim zona de TEST (actual)
    test_actual = raw_data_full[TRAINING_SET_LENGTH:]

    #  Construim lista TESTING RNN corect aliniată
    testing_rnn_full = [None] * len(test_actual)
    testing_rnn_full[NUMBER_OF_PREVIOUS_DATA_POINTS:] = list(testing_predict_rnn[:, 0])

    # lot combined results
    print('\nPlotting combined graph of both the models...')
    plot(currency, raw_data_full, test_actual, testing_rnn_full, testing_predict_arima,
         "testing_prediction_arima_and_rnn.pdf")

    print('\n Combined plot saved as testing_prediction_arima_and_rnn.pdf')

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()  
