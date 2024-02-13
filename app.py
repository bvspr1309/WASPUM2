import streamlit as st
from datetime import date, timedelta
import numpy as np
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from data_fetcher import fetch_stock_data
from lstm_model import train_and_save_model, predict_future_prices

# Initialize News API and Sentiment Analyzer
newsapi = NewsApiClient(api_key='ce3c053ec86f49d996b11a84c6c9d27a')
analyzer = SentimentIntensityAnalyzer()

# Set up Streamlit app configuration
st.set_page_config(page_title="LSTM Stock Price Prediction", layout="wide")
st.title('Stock Price Predictions with LSTM and Sentiment Analysis')

# Sidebar configuration for user input
st.sidebar.header('User Input Parameters')
ticker = st.sidebar.text_input('Stock Ticker', value='AAPL').upper()
today = date.today()
start_date = st.sidebar.date_input('Start Date', today - timedelta(days=365 * 2))
end_date = st.sidebar.date_input('End Date', today)

if start_date > end_date:
    st.sidebar.error('Error: End date must be after start date.')

# Function to fetch news and perform sentiment analysis
def get_news_sentiment(ticker):
    from_date = today - timedelta(days=30)  # Analyze news from the past 30 days
    to_date = today
    all_articles = newsapi.get_everything(q=ticker,
                                          from_param=from_date.isoformat(),
                                          to=to_date.isoformat(),
                                          language='en',
                                          sort_by='relevancy',
                                          page_size=100)
    sentiments = []
    for article in all_articles['articles']:
        text = article['title'] + '. ' + article['description'] if article['description'] else article['title']
        sentiment = analyzer.polarity_scores(text)
        sentiments.append(sentiment['compound'])
    return np.mean(sentiments) if sentiments else 0

# Display sentiment analysis results
st.header('Sentiment Analysis from News')
sentiment_score = get_news_sentiment(ticker)
sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
st.write(f"Average sentiment for {ticker}: {sentiment} ({sentiment_score:.2f})")

# Display recent data
st.header(f'Recent data for {ticker}')
data = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
if data is not None and not data.empty:
    st.write(data.tail(10))

# LSTM Model Prediction
st.header('Predict Future Stock Prices')
if st.button('Predict'):
    with st.spinner('Training model...'):
        model, scaler = train_and_save_model(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), look_back=60)
        if model is not None:
            st.success('Model trained successfully!')
            N = st.slider('Days to Predict:', 1, 30, 5)
            recent_data = data['Close'].values[-60:]
            predictions = predict_future_prices(model, scaler, recent_data, look_back=60, days_to_predict=N)
            st.header('Prediction Results')
            for i, prediction in enumerate(predictions, 1):
                st.write(f'Day {i}: ${prediction:.2f}')
        else:
            st.error('Model training failed.')
