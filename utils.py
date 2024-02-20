import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data, look_back=60):
    """
    Prepares time series data for LSTM training by creating sequences of historical data 
    with the specified look-back period and multiple features.
    
    Parameters:
    - data: DataFrame containing the stock data.
    - look_back: Number of previous time steps to use as input variables to predict the next time period.
    
    Returns:
    - X: Feature dataset (input sequences).
    - y: Target dataset (next time period's value for 'Close').
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back, 3])  # Assuming 'Close' is at index 3
    return np.array(X), np.array(y)

def scale_data(data):
    """
    Scales multiple features data to be between 0 and 1, which is useful for LSTM models. 
    This function returns the scaler for inverse transformation and the scaled data.
    
    Parameters:
    - data: Multi-dimensional data to be scaled.
    
    Returns:
    - scaler: The scaler used for data transformation.
    - scaled_data: Scaled version of the input data.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaler, scaled_data

def inverse_transform(scaler, data):
    """
    Performs inverse transformation of scaled data back to its original scale.
    
    Parameters:
    - scaler: The scaler used for the original scaling.
    - data: Scaled data to be inversely transformed.
    
    Returns:
    - Original scale data.
    """
    return scaler.inverse_transform(data)