import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        return None
