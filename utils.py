import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import USFederalHolidayCalendar

def preprocess_data(data, look_back=60):
    """
    Prepares time series data for LSTM training by creating sequences of historical data with the specified look-back period.
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def scale_data(data):
    """
    Scales data to be between 0 and 1, which is useful for LSTM models.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaler, scaled_data

def inverse_transform(scaler, data):
    """
    Inverse transforms the scaled data back to its original scale.
    """
    return scaler.inverse_transform(data)

def get_business_days_in_month(start_date, end_date):
    """
    Generates a list of business days within a given range, excluding weekends and US federal holidays.
    """
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    return pd.date_range(start_date, end_date, freq=us_bd)

def get_business_days_future(start_date, days_ahead):
    """
    Generates a list of future business days from a given start date, excluding weekends and US federal holidays.
    """
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    end_date = pd.offsets.CustomBusinessDay(days=days_ahead, calendar=USFederalHolidayCalendar())
    return pd.date_range(start_date, periods=days_ahead, freq=us_bd)