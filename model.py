import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_fetcher import fetch_stock_data
from utils import preprocess_data, scale_data

def create_dataset(data, look_back=60):
    """
    Converts an array of values into a dataset matrix.
    """
    X, Y = [], []
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

def train_lstm_model(ticker):
    """
    Fetches historical stock data, preprocesses it, trains an LSTM model, and saves the model.
    
    Parameters:
    - ticker: Stock ticker symbol as a string.
    
    Returns:
    - model: The trained LSTM model.
    - scaler: Scaler used for inverse transforming predicted data.
    """
    # Fetch stock data
    df = fetch_stock_data(ticker, period='1095d')
    
    # Preprocess data
    df_filtered = df[['Close']]
    values = df_filtered.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)
    
    # Create dataset for training
    X, y = create_dataset(scaled_data)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    
    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=1)
    
    # Optionally, save the model for later use
    # model.save('lstm_model.h5')
    
    return model, scaler

def predict_stock_price(model, scaler, ticker):
    """
    Predicts future stock price using the trained LSTM model.
    
    Parameters:
    - model: The trained LSTM model.
    - scaler: The MinMaxScaler used for the data.
    - ticker: Stock ticker symbol as a string for fetching recent data to predict future price.
    
    Returns:
    - prediction: The predicted stock price.
    """
    # Fetch recent stock data
    recent_data = fetch_stock_data(ticker, period='60d')  # Adjust period based on your model's look_back
    recent_data_filtered = recent_data[['Close']]
    last_60_days_scaled = scaler.transform(recent_data_filtered.values.reshape(-1, 1))
    
    # Prepare the data for prediction
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Make prediction
    prediction_scaled = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction_scaled)
    
    return prediction
