import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from utils import scale_data, preprocess_data, inverse_transform
from data_fetcher import fetch_stock_data

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
        close_prices = df['Close'].values
        scaler, scaled_data = scale_data(close_prices)
        X, y = preprocess_data(scaled_data.flatten(), look_back)
        
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        
        return model, scaler
    else:
        print("Failed to fetch data for ticker:", ticker)
        return None, None

def predict_future_prices(model, scaler, recent_data, look_back=60, days_to_predict=1):
    predictions = []
    current_batch = recent_data[-look_back:].reshape((1, look_back, 1))
    
    for i in range(days_to_predict):
        predicted_price = model.predict(current_batch)[0]
        predictions.append(predicted_price)
        
        current_batch = np.append(current_batch[:,1:,:], [[predicted_price]], axis=1)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions
