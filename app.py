import streamlit as st
from datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from data_fetcher import fetch_stock_data
from lstm_model import train_and_save_model, predict_future_prices, predict_previous_month_prices
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up Streamlit app configuration
st.set_page_config(page_title="LSTM Stock Price Prediction", layout="wide")
st.title('Stock Price Predictions with LSTM')

# Sidebar configuration
st.sidebar.header('User Input Parameters')
ticker = st.sidebar.text_input('Stock Ticker', value='AAPL').upper()
today = date.today()
start_date = st.sidebar.date_input('Start Date', today - timedelta(days=365 * 2))
end_date = st.sidebar.date_input('End Date', today)

if start_date > end_date:
    st.sidebar.error('Error: End date must be after start date.')

# Display recent data
st.header(f'Recent Data for {ticker}')
data = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
if data is not None and not data.empty:
    st.write(data.tail(10))

# LSTM Model Prediction and Comparison
if st.button('Predict'):
    with st.spinner('Training model...'):
        model, scaler = train_and_save_model(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), look_back=60)
        if model is not None:
            st.success('Model trained successfully!')

            # Future prediction for the next 28 working days
            recent_data = data['Close'].values[-60:]
            future_dates = pd.date_range(start=pd.Timestamp.today(), periods=40, freq='B')[:28]
            predictions = predict_future_prices(model, scaler, recent_data, look_back=60)
            st.header('Prediction Results for the Next 28 Working Days')
            for date, prediction in zip(future_dates, predictions):
                st.write(f"{date.strftime('%Y-%m-%d')} ({date.strftime('%A')}): ${prediction:.2f}")

            # Predict and compare previous month's prices
            prev_data_start = today - timedelta(days=180)  # 6 months prior for training data
            prev_data_end = today - timedelta(days=1)  # Until yesterday
            prev_data = fetch_stock_data(ticker, prev_data_start.strftime('%Y-%m-%d'), prev_data_end.strftime('%Y-%m-%d'))
            prev_month_predictions, prediction_dates = predict_previous_month_prices(model, scaler, prev_data['Close'].values[-60:], look_back=60)
            
            # Fetch actual prices for the last month
            last_month_data = fetch_stock_data(ticker, prediction_dates[0].strftime('%Y-%m-%d'), prediction_dates[-1].strftime('%Y-%m-%d'))
            actual_prices = last_month_data['Close'].values
            dates = last_month_data['Date']
            
            # Ensure lengths match for plotting
            min_len = min(len(actual_prices), len(prev_month_predictions))
            actual_prices = actual_prices[:min_len]
            prev_month_predictions = prev_month_predictions[:min_len]
            dates = dates[:min_len]

            # Plot actual vs predicted prices
            plt.figure(figsize=(10, 5))
            plt.plot(dates, actual_prices, label='Actual Prices')
            plt.plot(dates, prev_month_predictions, label='Predicted Prices', linestyle='--')
            plt.title('Actual vs Predicted Prices for Last Month')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(plt)
            
            # Display evaluation metrics
            mse = mean_squared_error(actual_prices, prev_month_predictions)
            mae = mean_absolute_error(actual_prices, prev_month_predictions)
            r2 = r2_score(actual_prices, prev_month_predictions)
            st.write(f"Mean Squared Error (MSE): {mse}")
            st.write(f"Mean Absolute Error (MAE): {mae}")
            st.write(f"R^2 Score: {r2}")
        else:
            st.error('Model training failed.')
