import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset (historical AQI data)
@st.cache_data
def load_data():
    file_path = 'cleaned_city_day.csv'  # Path to your dataset
    data = pd.read_csv(file_path)
    return data

# Preprocess data: filter by city and return scaled AQI data
def preprocess_data(data, city_name):
    city_data = data[data['City'] == city_name]
    city_data = city_data[['Date', 'AQI']].fillna(method='ffill')
    city_data['Date'] = pd.to_datetime(city_data['Date'])
    city_data.set_index('Date', inplace=True)

    # Feature scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    city_data_scaled = scaler.fit_transform(city_data[['AQI']])
    
    return city_data_scaled, scaler

# Create time-series data
def create_time_series_data(data, time_steps=7):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps, 0])
        y.append(data[i+time_steps, 0])
    return np.array(X), np.array(y)

# Build and compile LSTM model
def build_lstm_model(time_steps=7):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict the next day's AQI
def predict_next_aqi(model, data, scaler, time_steps=7):
    last_days = data[-time_steps:]
    last_days_scaled = scaler.transform(last_days)
    last_days_scaled = last_days_scaled.reshape((1, time_steps, 1))
    
    next_day_aqi_scaled = model.predict(last_days_scaled)
    next_day_aqi = scaler.inverse_transform(next_day_aqi_scaled)
    return next_day_aqi[0][0]

# Main Streamlit app
def main():
    st.title('AQI Prediction by City')
    
    # Load data
    data = load_data()
    
    # Select city
    city_name = st.selectbox('Select City', data['City'].unique())
    
    # Preprocess and display message
    if st.button('Predict AQI'):
        city_data_scaled, scaler = preprocess_data(data, city_name)
        
        # Create time-series data
        time_steps = 7
        X, y = create_time_series_data(city_data_scaled, time_steps)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data into training and test sets
        split_ratio = 0.8
        split = int(len(X) * split_ratio)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build and train LSTM model
        model = build_lstm_model(time_steps)
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        
        # Predict next day's AQI
        predicted_aqi = predict_next_aqi(model, city_data_scaled, scaler, time_steps)
        st.write(f'Predicted AQI for {city_name} for the next day: {predicted_aqi:.2f}')

if __name__ == '__main__':
    main()
