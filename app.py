import streamlit as st
from datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from data_fetcher import fetch_stock_data
from lstm_model import train_and_save_model, predict_future_prices, predict_previous_month_prices
from utils import get_business_days_in_month, get_business_days_future

# Setting up the Streamlit app configuration
st.set_page_config(page_title="LSTM Stock Price Prediction", layout="wide")
st.title('Stock Price Predictions with LSTM')

# Sidebar for user input parameters
st.sidebar.header('User Input Parameters')
ticker = st.sidebar.text_input('Stock Ticker', value='AAPL').upper()
today = date.today()
start_date = st.sidebar.date_input('Start Date', today - timedelta(days=365))
end_date = st.sidebar.date_input('End Date', today)

if start_date > end_date:
    st.sidebar.error('Error: End date must be after start date.')

# Displaying recent stock data
st.header(f'Recent Stock Data for {ticker}')
data = fetch_stock_data(ticker, start_date, end_date)
if not data.empty:
    st.write(data.tail())

# Predicting future stock prices
st.header('Predict Future Stock Prices')
if st.button('Predict'):
    model, scaler = train_and_save_model(ticker, start_date, end_date)
    if model:
        future_dates = get_business_days_future(end_date, 28)
        predictions = predict_future_prices(model, scaler, data['Close'], 60, future_dates)
        
        st.subheader('Future Price Predictions')
        for date, price in zip(future_dates, predictions):
            st.write(f"{date.date()} ({date.strftime('%A')}): ${price:.2f}")
    else:
        st.error('Failed to train the model. Please check the data or try different parameters.')

# Predicting and evaluating the previous month's stock prices
st.header("Evaluation of Model's Performance for Previous Month")
previous_month_start, previous_month_end = get_business_days_in_month(today - pd.offsets.MonthBegin(2), today - pd.offsets.MonthBegin(1))
if st.button('Evaluate Model'):
    historical_data = fetch_stock_data(ticker, previous_month_start - timedelta(days=90), previous_month_end)
    predictions, actuals = predict_previous_month_prices(model, scaler, historical_data['Close'], previous_month_start, previous_month_end)

    if predictions is not None:
        # Plotting actual vs predicted prices
        plt.figure(figsize=(10, 5))
        plt.plot(historical_data['Date'][-len(actuals):], actuals, marker='o', label='Actual Prices')
        plt.plot(historical_data['Date'][-len(predictions):], predictions, marker='x', linestyle='--', label='Predicted Prices')
        plt.title('Actual vs Predicted Stock Prices for the Previous Month')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt)

        # Calculating and displaying evaluation metrics
        mse, mae, r2 = calculate_evaluation_metrics(actuals, predictions)
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"Mean Absolute Error (MAE): {mae}")
        st.write(f"R^2 Score: {r2}")
    else:
        st.error("Model evaluation failed. Insufficient or inadequate data for the previous month.")