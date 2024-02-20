import yfinance as yf
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data for the specified ticker over a given period,
    excluding weekends and US federal holidays.
    """
    # Download stock data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data.reset_index(inplace=True)
    
    # Exclude weekends
    data = data[data['Date'].dt.dayofweek < 5]
    
    # Exclude US federal holidays
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date).date
    data = data[~data['Date'].dt.date.isin(holidays)]
    
    return data
