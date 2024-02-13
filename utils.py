import numpy as np
import pandas as pd

def preprocess_data(data, feature_columns=['Close'], target_column='Close', look_back=60):
    """
    Preprocess the stock data for the LSTM model.
    
    Parameters:
    - data: DataFrame containing the stock data.
    - feature_columns: List of column names to use as features.
    - target_column: Column name to use as the target.
    - look_back: Number of past days to consider for predicting a future value.
    
    Returns:
    - X: Features data set.
    - y: Target data set.
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[feature_columns].iloc[i:(i + look_back)].values)
        y.append(data[target_column].iloc[i + look_back])
    
    X, y = np.array(X), np.array(y)
    return X, y

def scale_data(X_train, y_train, X_test):
    """
    Scale the data using MinMaxScaler for LSTM input.
    
    Parameters:
    - X_train: Training feature dataset.
    - y_train: Training target dataset.
    - X_test: Test feature dataset.
    
    Returns:
    - Scaled versions of the input datasets.
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = np.array([scaler.fit_transform(x) for x in X_train])
    X_test_scaled = np.array([scaler.transform(x) for x in X_test])
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    
    return X_train_scaled, y_train_scaled, X_test_scaled, scaler

def inverse_transform_y(y, scaler):
    """
    Inverse transform the scaled target values back to their original scale.
    
    Parameters:
    - y: Scaled target dataset.
    - scaler: The scaler used for original scaling.
    
    Returns:
    - y_orig: Target dataset in original scale.
    """
    y_orig = scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)
    return y_orig

def handle_error(message):
    """
    Generic error handling function to display error messages.
    
    Parameters:
    - message: The error message to display.
    
    Returns:
    - None
    """
    print(f"Error: {message}")
    # Depending on your application, you might want to log this error or notify the user differently.
