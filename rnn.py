import warnings
import pandas as pd
import numpy as np
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

TRAINING_PERCENTAGE = 0.7
NUMBER_OF_PREVIOUS_DATA_POINTS = 3
LENGTH_DATA_SET = 0
np.random.seed(7)
TRAINING_SET_LENGTH = 0
TESTING_SET_LENGTH = 0

def training_testing_buckets(raw_data, training_percentage):
    global TRAINING_SET_LENGTH, TESTING_SET_LENGTH
    TRAINING_SET_LENGTH = int(LENGTH_DATA_SET * training_percentage)
    TESTING_SET_LENGTH = LENGTH_DATA_SET - TRAINING_SET_LENGTH
    training_set = raw_data[0:TRAINING_SET_LENGTH]
    testing_set = raw_data[TRAINING_SET_LENGTH:LENGTH_DATA_SET]
    return training_set, testing_set

def modify_data_set_rnn(training_set, testing_set):
    def create_sequences(data):
        actual, predict = [], []
        for i in range(len(data) - NUMBER_OF_PREVIOUS_DATA_POINTS - 1):
            actual.append(data[i: i + NUMBER_OF_PREVIOUS_DATA_POINTS])
            predict.append(data[i + NUMBER_OF_PREVIOUS_DATA_POINTS])
        return np.array(actual), np.array(predict)

    return (*create_sequences(training_set), *create_sequences(testing_set))

def load_data_set(currency):
    df = pd.read_csv("currency_prediction_data_set.csv", header=0, index_col=0, parse_dates=True)
    column_headers = df.columns.values.tolist()
    currency_index = column_headers.index('USD/' + currency.upper()) + 1
    df_full = pd.read_csv("currency_prediction_data_set.csv", usecols=[0, currency_index], engine='python', parse_dates=[0])
    dates = pd.to_datetime(df_full.iloc[:, 0]).tolist()
    raw_data = df_full.iloc[:, 1].tolist()
    global LENGTH_DATA_SET
    LENGTH_DATA_SET = len(raw_data)
    return raw_data, dates

def build_rnn_model(train_actual, train_predict):
    train_actual = train_actual.reshape((train_actual.shape[0], train_actual.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, input_shape=(NUMBER_OF_PREVIOUS_DATA_POINTS, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_actual, train_predict, epochs=100, batch_size=16, verbose=0)
    return model

def predict_rnn(model, train_actual, test_actual):
    train_actual = train_actual.reshape((train_actual.shape[0], train_actual.shape[1], 1))
    test_actual = test_actual.reshape((test_actual.shape[0], test_actual.shape[1], 1))
    return model.predict(train_actual), model.predict(test_actual)

def plot_rnn(currency, raw_data, training_predict, testing_predict, scaler, file_name, forecast_future=None):
    training_real = scaler.inverse_transform(training_predict)
    testing_real = scaler.inverse_transform(testing_predict)

    plt.figure(figsize=(10, 6))
    plt.plot(raw_data, label="Valori reale", color="blue")
    plt.plot(range(NUMBER_OF_PREVIOUS_DATA_POINTS, NUMBER_OF_PREVIOUS_DATA_POINTS + len(training_real)),
             training_real[:, 0], label="Predicții antrenare", color="green")
    plt.plot(range(TRAINING_SET_LENGTH + NUMBER_OF_PREVIOUS_DATA_POINTS,
                   TRAINING_SET_LENGTH + NUMBER_OF_PREVIOUS_DATA_POINTS + len(testing_real)),
             testing_real[:, 0], label="Predicții testare", color="red")

    if forecast_future:
        future_x = list(range(TRAINING_SET_LENGTH + NUMBER_OF_PREVIOUS_DATA_POINTS + len(testing_real),
                              TRAINING_SET_LENGTH + NUMBER_OF_PREVIOUS_DATA_POINTS + len(testing_real) + len(forecast_future)))
        plt.plot(future_x, forecast_future, label="Predicții viitoare", color="orange", linestyle="--")

    plt.title(f"Predicții RNN - USD/{currency.upper()}")
    plt.xlabel("Număr de zile")
    plt.ylabel(f"Valoare USD/{currency.upper()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def predict_future_days(model, last_window, days, scaler):
    future_preds = []
    window = last_window.copy()
    for _ in range(days):
        input_seq = np.array(window).reshape((1, NUMBER_OF_PREVIOUS_DATA_POINTS, 1))
        next_pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
        next_pred = scaler.inverse_transform([[next_pred_scaled]])[0, 0]
        future_preds.append(next_pred)
        window = np.append(window[1:], next_pred_scaled)
    return future_preds

def rnn_model(currency, forecast_days=10):
    raw_data, dates = load_data_set(currency)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(np.array(raw_data).reshape(-1, 1)).flatten()
    training_set, testing_set = training_testing_buckets(scaled_data, TRAINING_PERCENTAGE)
    train_actual, train_predict, test_actual, test_predict = modify_data_set_rnn(training_set, testing_set)
    model = build_rnn_model(train_actual, train_predict)
    training_predict, testing_predict = predict_rnn(model, train_actual, test_actual)
    last_window = scaled_data[-NUMBER_OF_PREVIOUS_DATA_POINTS:]
    future_predictions = predict_future_days(model, last_window, forecast_days, scaler)
    plot_rnn(currency, raw_data, training_predict, testing_predict, scaler, "predictie_rnn.pdf", future_predictions)

    return {
        "raw_data": raw_data,
        "training_predict": training_predict,
        "testing_predict": testing_predict,
        "future_predictions": future_predictions,
        "scaler": scaler
    }

warnings.filterwarnings("ignore")
