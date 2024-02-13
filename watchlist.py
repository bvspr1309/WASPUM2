class Watchlist:
    def __init__(self):
        """
        Initializes the watchlist with an empty list.
        """
        self.stocks = []

    def add_stock(self, ticker):
        """
        Adds a stock ticker to the watchlist if it's not already present.
        
        Parameters:
        - ticker: Stock ticker symbol as a string.
        """
        if ticker not in self.stocks:
            self.stocks.append(ticker)
            print(f"{ticker} added to watchlist.")
        else:
            print(f"{ticker} is already in the watchlist.")

    def remove_stock(self, ticker):
        """
        Removes a stock ticker from the watchlist if it exists.
        
        Parameters:
        - ticker: Stock ticker symbol as a string.
        """
        if ticker in self.stocks:
            self.stocks.remove(ticker)
            print(f"{ticker} removed from watchlist.")
        else:
            print(f"{ticker} not found in the watchlist.")

    def get_watchlist(self):
        """
        Returns the current list of stock tickers in the watchlist.
        
        Returns:
        - stocks: List of stock ticker symbols.
        """
        return self.stocks

    def display_watchlist_info(self):
        """
        Fetches and displays information for each stock in the watchlist.
        """
        from data_fetcher import fetch_stock_data
        
        if not self.stocks:
            print("Your watchlist is currently empty.")
            return
        
        print("Watchlist Contents:")
        for ticker in self.stocks:
            data = fetch_stock_data(ticker, period='1d')  # Fetch daily data for the stock
            if data is not None:
                # Displaying only the last closing price as an example. Adjust as needed.
                last_close = data['Close'].iloc[-1]
                print(f"{ticker}: Last Close = {last_close}")
            else:
                print(f"Failed to fetch data for {ticker}.")
