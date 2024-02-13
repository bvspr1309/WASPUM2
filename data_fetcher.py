import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data for the specified ticker over a given period.
    
    Parameters:
    - ticker: The stock ticker symbol as a string.
    - start_date: The start date for fetching historical data (string in format YYYY-MM-DD).
    - end_date: The end date for fetching historical data (string in format YYYY-MM-DD).
    
    Returns:
    - data: A DataFrame containing the historical stock data.
    """
    try:
        # Download stock data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        # Reset index to make 'Date' a column, not an index
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        return None
