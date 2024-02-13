import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from data_fetcher import fetch_stock_data
from lstm_model import train_and_save_model, predict_future_prices
from utils import scale_data, preprocess_data

st.title('Stock Price Predictions with LSTM')

# Sidebar configuration
st.sidebar.header('User Input Parameters')

def user_input_features():
    today = date.today()
    ticker = st.sidebar.text_input('Stock Ticker', value='AAPL').upper()
    start_date = st.sidebar.date_input('Start Date', today - timedelta(days=365 * 2))
    end_date = st.sidebar.date_input('End Date', today)
    if start_date > end_date:
        st.sidebar.error('Error: End date must be after start date.')
    return ticker, start_date, end_date

ticker, start_date, end_date = user_input_features()

# Fetch and display recent data
st.header(f'Recent data for {ticker}')
data = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
if data is not None and not data.empty:
    st.write(data.tail())

# LSTM Model Prediction
st.header('Predict Future Stock Prices')

if st.button('Predict'):
    with st.spinner('Training model...'):
        model, scaler = train_and_save_model(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if model is not None:
            st.success('Model trained successfully!')
            
            # Assuming you want to predict the next N days
            N = st.slider('Days to Predict:', 1, 30, 5)
            recent_data = data['Close'].values[-60:]  # Last 60 days for prediction input
            predictions = predict_future_prices(model, scaler, recent_data, look_back=60)
            
            st.header('Prediction Results')
            for i in range(N):
                # This is a simplification. You might want to adjust prediction logic based on your model's output
                if i < len(predictions):
                    st.write(f'Day {i+1}: ${predictions[i][0]:.2f}')
                else:
                    st.write(f'Day {i+1}: Prediction data not available')
        else:
            st.error('Model training failed.')
else:
    st.info('Click the predict button to forecast future prices.')

# Optional: Add more app functionality here

if __name__ == '__main__':
    st.set_page_config(page_title="LSTM Stock Price Prediction", layout="wide")
