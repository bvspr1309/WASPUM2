import yfinance as yf

def fetch_stock_data(ticker, period='1095d'):
    """
    Fetches historical stock data for the specified ticker over a given period.
    
    Parameters:
    - ticker: The stock ticker symbol as a string.
    - period: The period over which to fetch historical data. Default is '1095d' (3 years).
    
    Returns:
    - data: A DataFrame containing the historical stock data.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        # Clean and prepare data, if necessary
        data.reset_index(inplace=True)  # Reset index to make 'Date' a column
        return data
    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        return None

def fetch_futures_data():
    """
    Fetches data for major futures indices like S&P, Dow, Nasdaq, and Russell 2000.
    You may need to know the ticker symbols for these futures or use proxies.
    
    Returns:
    - futures_data: A dictionary with the futures indices as keys and their data as values.
    """
    futures_dict = {
        'S&P Futures': '^GSPC',  # This is actually the S&P 500 Index, adjust if you have a direct future ticker
        'Dow Futures': '^DJI',
        'Nasdaq Futures': '^IXIC',
        'Russell 2000 Futures': '^RUT'  # Adjust with actual futures ticker if necessary
    }
    
    futures_data = {}
    for name, ticker in futures_dict.items():
        data = fetch_stock_data(ticker, period='1d')  # Fetching daily data; adjust as needed
        futures_data[name] = data
    
    return futures_data
