import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data, look_back=60):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaler, scaled_data

def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)

def prepare_data_for_prediction(data, look_back=60):
    prepared_data = []
    length = len(data)
    if length >= look_back:
        prepared_data.append(data[(length - look_back):])
    prepared_data = np.array(prepared_data)
    prepared_data = np.reshape(prepared_data, (prepared_data.shape[0], look_back, 1))
    return prepared_data
