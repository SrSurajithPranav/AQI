import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset
file_path = '/content/cleaned_city_day.csv'  # Replace with your file path if necessary
data = pd.read_csv(file_path)

# Encode the city names to numeric values
label_encoder = LabelEncoder()
data['City_encoded'] = label_encoder.fit_transform(data['City'])

# Drop non-numeric columns
data.drop(['Date', 'AQI_Bucket', 'City'], axis=1, inplace=True)

# Handle missing values
data = data.ffill()

# Split features (X) and target (y)
X = data.drop(['AQI'], axis=1)
y = data['AQI']

# Store feature names
features = X.columns

# Scale the features
scaler_features = MinMaxScaler()
X_scaled = scaler_features.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

# Reshape the input for LSTM [samples, time_steps, features]
X_lstm = X_scaled.values.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split data into training and test sets
X_train_lstm, X_test_lstm, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

# Train the LSTM model
lstm_model.fit(X_train_lstm, y_train, batch_size=32, validation_split=0.2, epochs=10, verbose=0)

# Get the output from the LSTM
X_train_lstm_out = lstm_model.predict(X_train_lstm)
X_test_lstm_out = lstm_model.predict(X_test_lstm)

# Train the Random Forest using the LSTM output
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_lstm_out, y_train)

# Train the XGBoost model using Random Forest output
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(rf_model.predict(X_train_lstm_out).reshape(-1, 1), y_train)

def predict_aqi_for_city(city_name):
    city_encoded = label_encoder.transform([city_name])[0]
    city_data = data[data['City_encoded'] == city_encoded]
    
    if city_data.empty:
        return f"No data available for '{city_name}'."
    
    X_city = city_data.drop(['AQI'], axis=1)
    X_city_scaled = scaler_features.transform(X_city)
    X_city_lstm = X_city_scaled.reshape((X_city_scaled.shape[0], 1, X_city_scaled.shape[1]))
    
    city_lstm_out = lstm_model.predict(X_city_lstm)
    city_rf_out = rf_model.predict(city_lstm_out).reshape(-1, 1)
    aqi_prediction = xgb_model.predict(city_rf_out)
    
    return f"Predicted current AQI for {city_name}: {aqi_prediction[-1]:.2f}"

def predict_next_aqi(model, data, scaler, time_steps=7):
    if len(data) < time_steps:
        return "Not enough data points for prediction"

    # Ensure data has the same columns as the training data
    data = data.reindex(columns=features, fill_value=0)
    
    # Scale the data
    data_scaled = scaler.transform(data)
    
    # Get the last 'time_steps' days of data
    last_days = data_scaled[-time_steps:]
    
    # Reshape the input to (samples, time_steps, features)
    last_days_reshaped = last_days.reshape((1, time_steps, last_days.shape[1]))

    # Predict the AQI for the next day
    next_day_aqi_scaled = model.predict(last_days_reshaped)

    # Inverse transform to get the original scale
    next_day_aqi = scaler.inverse_transform(np.hstack([next_day_aqi_scaled, np.zeros((next_day_aqi_scaled.shape[0], data.shape[1]-1))]))[0][0]
    return next_day_aqi

# Streamlit UI
st.title("Air Quality Index (AQI) Prediction")

# City selection
city_name = st.selectbox("Select a city:", options=label_encoder.classes_)

if st.button("Predict AQI"):
    city_encoded = label_encoder.transform([city_name])[0]
    city_data = data[data['City_encoded'] == city_encoded]
    
    if city_data.empty:
        st.write(f"No data available for '{city_name}'. Please try another city.")
    else:
        # Predict AQI for the next day
        next_day_features = city_data.drop(['AQI'], axis=1)
        predicted_next_day_aqi = predict_next_aqi(lstm_model, next_day_features, scaler_features)
        
        if isinstance(predicted_next_day_aqi, str):
            st.write(predicted_next_day_aqi)
        else:
            st.write(f"Predicted AQI for {city_name} for the next day: {predicted_next_day_aqi:.2f}")
        
        # Predict current AQI
        current_aqi_prediction = predict_aqi_for_city(city_name)
        st.write(current_aqi_prediction)
