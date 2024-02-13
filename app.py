import streamlit as st
from datetime import date, timedelta
from data_fetcher import fetch_stock_data
from lstm_model import train_and_save_model, predict_future_prices
from utils import scale_data

# Ensure set_page_config is called first
st.set_page_config(page_title="LSTM Stock Price Prediction", layout="wide")
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

option = st.sidebar.selectbox('Choose an option', ['Visualize', 'Recent Data', 'Predict'])

if option == 'Visualize':
    st.header('Stock Data Visualization')
    data = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if data is not None and not data.empty:
        st.line_chart(data['Close'])
elif option == 'Recent Data':
    st.header(f'Recent Data for {ticker}')
    data = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if data is not None and not data.empty:
        st.write(data.tail(10))
elif option == 'Predict':
    st.header('Predict Future Stock Prices')
    if st.button('Predict'):
        with st.spinner('Training model...'):
            model, scaler = train_and_save_model(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), look_back=60)
            if model is not None:
                st.success('Model trained successfully!')
                
                # Assuming you want to predict the next N days
                N = st.slider('Days to Predict:', 1, 30, 5)
                data = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if data is not None and not data.empty:
                    recent_data = data['Close'].values[-60:]  # Use the last 60 days for prediction input
                    recent_data_scaled = scale_data(recent_data)[1]  # Only use scaled data
                    predictions = predict_future_prices(model, scaler, recent_data_scaled, look_back=60, days_to_predict=N)
                    st.header('Prediction Results')
                    for i, prediction in enumerate(predictions, 1):
                        st.write(f'Day {i}: ${prediction:.2f}')
            else:
                st.error('Model training failed.')
