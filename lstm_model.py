import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from utils import scale_data, preprocess_data, inverse_transform
from data_fetcher import fetch_stock_data

def create_lstm_model(input_shape):
    """
    Creates and compiles an LSTM model given an input shape.
    
    Parameters:
    - input_shape: The shape of the input data (time_steps, features).
    
    Returns:
    - model: The compiled LSTM model.
    """
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
    """
    Fetches stock data, preprocesses, trains an LSTM model, and saves the model.
    
    Parameters:
    - ticker: Stock ticker symbol as a string.
    - start_date: Start date for fetching historical data.
    - end_date: End date for fetching historical data.
    - look_back: Number of days to look back for creating sequences.
    
    Returns:
    - model: The trained LSTM model.
    - scaler: Scaler used for data preprocessing.
    """
    # Fetch and prepare data
    df = fetch_stock_data(ticker, start_date, end_date)
    if df is not None:
        close_prices = df['Close'].values
        scaler, scaled_data = scale_data(close_prices)
        X, y = preprocess_data(scaled_data.flatten(), look_back)
        
        # Reshape input to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split the data into training and testing sets
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Create and train the LSTM model
        model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        
        # Optionally save the model
        # model.save(f"{ticker}_lstm_model.h5")
        
        return model, scaler
    else:
        print("Failed to fetch data for ticker:", ticker)
        return None, None

def predict_future_prices(model, scaler, recent_data, look_back=60):
    """
    Uses the trained LSTM model to predict future stock prices.
    
    Parameters:
    - model: The trained LSTM model.
    - scaler: Scaler used for data preprocessing.
    - recent_data: The most recent stock data for making predictions.
    - look_back: Number of days to look back for creating a single sequence.
    
    Returns:
    - predictions: Predicted future prices.
    """
    # Prepare recent data for prediction
    recent_scaled = scaler.transform(recent_data.reshape(-1, 1))
    X_test = np.array([recent_scaled[-look_back:]])
    X_test = np.reshape(X_test, (X_test.shape[0], look_back, 1))
    
    # Make predictions
    predicted_scaled = model.predict(X_test)
    predictions = inverse_transform(scaler, predicted_scaled)
    
    return predictions
