import streamlit as st
from data_fetcher import fetch_stock_data, fetch_futures_data
from model import train_lstm_model, predict_stock_price
from watchlist import Watchlist
import pandas as pd

# Initialize watchlist
user_watchlist = Watchlist()

# Page configuration
st.set_page_config(page_title="Stock Price Prediction App", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Watchlist"))

# Futures data display on Home
if page == "Home":
    st.title("Market Futures Overview")
    futures_data = fetch_futures_data()
    for name, data in futures_data.items():
        if data is not None:
            st.write(f"{name}: Last Close = {data['Close'].iloc[-1]}")
        else:
            st.write(f"Failed to fetch data for {name}.")

    # Search functionality
    ticker = st.text_input("Search for a stock (Ticker):").upper()
    if ticker:
        # Display stock information or train and predict
        data = fetch_stock_data(ticker, period='1095d')
        if data is not None:
            st.write(f"Displaying information for {ticker}")
            # Assuming data contains a 'Date' and 'Close' column for simplicity
            st.line_chart(data.set_index('Date')['Close'])
            
            # Placeholder for training and prediction
            st.write("Training model for prediction (this could take some time)...")
            # Here, you'd call train_lstm_model and predict_stock_price
            # For simplicity and speed, we're not executing these in real-time
            # model, scaler = train_lstm_model(ticker)
            # prediction = predict_stock_price(model, scaler, ticker)
            # st.write(f"Predicted close price for the next day: {prediction[0][0]}")
            
            # Add to watchlist button
            if st.button(f"Add {ticker} to Watchlist"):
                user_watchlist.add_stock(ticker)
                st.success(f"{ticker} added to Watchlist")

elif page == "Watchlist":
    st.title("Your Watchlist")
    if not user_watchlist.get_watchlist():
        st.write("Your watchlist is empty.")
    else:
        for ticker in user_watchlist.get_watchlist():
            st.write(ticker)
            # This could be expanded to include more detailed stock info or predictions

