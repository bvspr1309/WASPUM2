import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from utils import scale_data, preprocess_data, inverse_transform
from data_fetcher import fetch_stock_data
import pandas as pd

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_save_model(ticker, start_date, end_date, look_back=60):
    df = fetch_stock_data(ticker, start_date, end_date)
    if df is not None:
        features_considered = ['Open', 'High', 'Low', 'Close', 'Volume']
        features = df[features_considered]
        scaler, scaled_data = scale_data(features.values)
        X, y = preprocess_data(scaled_data, look_back)
        
        X = np.reshape(X, (X.shape[0], X.shape[1], len(features_considered)))
        
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        
        return model, scaler
    else:
        print("Failed to fetch data for ticker:", ticker)
        return None, None

def predict_future_prices(model, scaler, recent_data, look_back=60):
    # Adjusted to predict a fixed number of future working days (approx. 28 days excluding weekends)
    # This does not account for public holidays.
    days_to_predict = 28
    predictions = []
    current_batch = recent_data[-look_back:].reshape((1, look_back, 1))
    
    for i in range(days_to_predict * 2):  # Assuming roughly 2x to account for weekends
        predicted_price = model.predict(current_batch)[0]
        next_day_index = (pd.Timestamp.today() + pd.Timedelta(days=i+1)).dayofweek
        if next_day_index < 5:  # Skip weekends
            predictions.append(predicted_price)
            if len(predictions) >= days_to_predict:
                break
        current_batch = np.append(current_batch[:,1:,:], [[predicted_price]], axis=1)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions


def predict_previous_month_prices(model, scaler, data, look_back=60):
    """
    Predicts prices for the previous month using the trained LSTM model.
    
    Parameters:
    - model: The trained LSTM model.
    - scaler: Scaler used for data preprocessing.
    - data: Data used for making predictions, should include the last 'look_back' days before the month you want to predict.
    - look_back: Number of days to look back for creating a single sequence.
    
    Returns:
    - predictions: Predicted prices for the previous month.
    """
    # Calculate the number of weekdays in the last month
    last_month_start = (pd.Timestamp.today().replace(day=1) - pd.Timedelta(days=1)).replace(day=1)
    last_month_end = pd.Timestamp.today().replace(day=1) - pd.Timedelta(days=1)
    weekdays_in_month = pd.date_range(start=last_month_start, end=last_month_end, freq='B')

    predictions = []
    current_batch = data[-look_back:].reshape((1, look_back, 1))
    
    for i in range(len(weekdays_in_month)):
        predicted_price = model.predict(current_batch)[0]
        predictions.append(predicted_price)
        current_batch = np.append(current_batch[:,1:,:], [[predicted_price]], axis=1)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions, weekdays_in_month