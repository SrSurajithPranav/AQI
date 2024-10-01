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
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Set Streamlit page configuration
st.set_page_config(
    page_title="🌍 AQI Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to apply theme
def apply_theme(is_dark_mode):
    if is_dark_mode:
        theme_color = "#1E1E1E"
        text_color = "#FFFFFF"
        background_color = "#121212"
        sidebar_text_color = "#000000"
    else:
        theme_color = "#FFFFFF"
        text_color = "#000000"
        background_color = "#F0F2F6"
        sidebar_text_color = "#000000"

    st.markdown(
        f"""
        <style>
        :root {{
            --theme-color: {theme_color};
            --text-color: {text_color};
            --background-color: {background_color};
            --sidebar-text-color: {sidebar_text_color};
        }}
        body {{
            background-color: var(--background-color);
            color: var(--text-color);
        }}
        .stApp {{
            background-color: var(--background-color);
        }}
        .sidebar .sidebar-content {{
            background-color: var(--theme-color);
            color: var(--sidebar-text-color) !important;
        }}
        h1, h2, h3, h4, h5, h6, p {{
            color: var(--text-color) !important;
        }}
        .stButton>button {{
            background-color: #FF4B4B;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }}
        .stButton>button:hover {{
            background-color: #FF0000;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to load and preprocess data
@st.cache_resource
def load_data(file_path):
    data = pd.read_csv(file_path)
    label_encoder = LabelEncoder()
    data['City_encoded'] = label_encoder.fit_transform(data['City'])
    data['Date'] = pd.to_datetime(data['Date'])
    data.drop(['AQI_Bucket', 'City'], axis=1, inplace=True)
    data = data.ffill()
    return data, label_encoder

# Function to build and train models
@st.cache_resource
def build_and_train_models(data, _features):
    with st.spinner("Training models... This may take a few moments."):
        X = data.drop(['AQI', 'Date'], axis=1)
        y = data['AQI']
        scaler_features = MinMaxScaler()
        X_scaled = scaler_features.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=_features)
        X_lstm = X_scaled.values.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        X_train_lstm, X_test_lstm, y_train, y_test = train_test_split(
            X_lstm, y, test_size=0.2, random_state=42
        )
        
        # LSTM model
        lstm_model = Sequential()
        lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')
        
        # Use tqdm to create a progress bar
        with tqdm(total=10, desc="Training LSTM", leave=False) as pbar:
            for _ in range(10):
                lstm_model.fit(X_train_lstm, y_train, batch_size=32, validation_split=0.2, epochs=1, verbose=0)
                pbar.update(1)
        
        X_train_lstm_out = lstm_model.predict(X_train_lstm, verbose=0)
        
        # Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_lstm_out, y_train)
        
        # XGBoost model
        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_model.fit(rf_model.predict(X_train_lstm_out).reshape(-1, 1), y_train)
        
    return lstm_model, rf_model, xgb_model, scaler_features

# Function to store the latest prediction
@st.cache_data
def store_latest_prediction(city_name, prediction):
    st.session_state[f'{city_name}_latest_prediction'] = prediction

# Function to predict AQI for a city
def predict_aqi_for_city(city_name, label_encoder, data, lstm_model, rf_model, xgb_model, scaler_features):
    city_encoded = label_encoder.transform([city_name])[0]
    city_data = data[data['City_encoded'] == city_encoded]
    if city_data.empty:
        return f"No data available for '{city_name}'.", None
    
    X_city = city_data.drop(['AQI', 'Date'], axis=1)
    X_city = X_city[scaler_features.feature_names_in_]
    X_city_scaled = scaler_features.transform(X_city)
    X_city_lstm = X_city_scaled.reshape((X_city_scaled.shape[0], 1, X_city_scaled.shape[1]))
    
    city_lstm_out = lstm_model.predict(X_city_lstm)
    city_rf_out = rf_model.predict(city_lstm_out).reshape(-1, 1)
    aqi_prediction = xgb_model.predict(city_rf_out)
    
    latest_prediction = aqi_prediction[-1]
    store_latest_prediction(city_name, latest_prediction)
    
    return f"*Predicted current AQI for {city_name}: {latest_prediction:.2f}*", latest_prediction

# Function to plot AQI trends for the selected city
def plot_aqi_trends(city_name, data, label_encoder):
    city_encoded = label_encoder.transform([city_name])[0]
    city_data = data[data['City_encoded'] == city_encoded]
    if city_data.empty:
        st.error(f"No data available for '{city_name}'.")
        return
    city_data['Year'] = city_data['Date'].dt.year
    yearly_avg_aqi = city_data.groupby('Year')['AQI'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=yearly_avg_aqi['Year'], y=yearly_avg_aqi['AQI'], marker='o', ax=ax, color='#00aaff')
    ax.set_title(f'Yearly AQI Trends for {city_name}', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Average AQI', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

# Function to plot monthly AQI trends
def plot_monthly_aqi_trends(city_name, data, label_encoder):
    city_encoded = label_encoder.transform([city_name])[0]
    city_data = data[data['City_encoded'] == city_encoded]
    if city_data.empty:
        st.error(f"No data available for '{city_name}'.")
        return
    city_data['Month'] = city_data['Date'].dt.month
    monthly_avg_aqi = city_data.groupby('Month')['AQI'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=monthly_avg_aqi['Month'], y=monthly_avg_aqi['AQI'], ax=ax, palette='viridis')
    ax.set_title(f'Monthly AQI Trends for {city_name}', fontsize=16)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Average AQI', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

# Function to plot AQI comparison between cities
def plot_city_comparison(cities, data, label_encoder):
    fig, ax = plt.subplots(figsize=(10, 6))
    for city in cities:
        city_encoded = label_encoder.transform([city])[0]
        city_data = data[data['City_encoded'] == city_encoded]
        city_data['Year'] = city_data['Date'].dt.year
        yearly_avg_aqi = city_data.groupby('Year')['AQI'].mean().reset_index()
        sns.lineplot(x=yearly_avg_aqi['Year'], y=yearly_avg_aqi['AQI'], marker='o', label=city, ax=ax)
    ax.set_title("AQI Comparison Between Cities")
    ax.set_xlabel('Year')
    ax.set_ylabel('Average AQI')
    ax.legend()
    st.pyplot(fig)

# Function to provide AQI-based recommendations for outdoor activities
def aqi_recommendations(aqi_value):
    if aqi_value <= 50:
        return "Air quality is good. It's a great time for outdoor activities!"
    elif aqi_value <= 100:
        return "Air quality is moderate. You can go outside, but sensitive groups should take precautions."
    elif aqi_value <= 150:
        return "Air quality is unhealthy for sensitive groups. Limit prolonged outdoor exertion."
    elif aqi_value <= 200:
        return "Air quality is unhealthy. Everyone should limit prolonged outdoor exertion."
    elif aqi_value <= 300:
        return "Air quality is very unhealthy. Avoid outdoor activities."
    else:
        return "Air quality is hazardous. Stay indoors and avoid all outdoor activities."

# Function to analyze climate change impact
def climate_change_impact(data):
    data['Year'] = data['Date'].dt.year
    yearly_avg_aqi = data.groupby('Year')['AQI'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=yearly_avg_aqi['Year'], y=yearly_avg_aqi['AQI'], marker='o', ax=ax, color='#ff6347')
    ax.set_title('Climate Change Impact on AQI Over the Years', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Average AQI', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

# Function to fetch city-specific news using NewsAPI
def fetch_city_news(city_name, api_key):
    url = f"https://newsapi.org/v2/everything?q={city_name}&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    news = [article['title'] for article in articles[:3]]
    return news

# Function to provide city-specific news and recommendations
def city_specific_news(city_name, api_key):
    news = fetch_city_news(city_name, api_key)
    recommendations = [
        f"Stay updated with local news for {city_name}.",
        f"Check the AQI regularly if you live in {city_name}.",
        f"Follow health advisories issued by local authorities in {city_name}."
    ]
    return news, recommendations

# Load data and train models
@st.cache_resource
def load_data_and_train_models():
    file_path = r'C:\Users\srsur\Desktop\AQI\cleaned_city_day.csv'
    data, label_encoder = load_data(file_path)
    features = data.drop(['AQI', 'Date'], axis=1).columns
    lstm_model, rf_model, xgb_model, scaler_features = build_and_train_models(data, features)
    return data, label_encoder, lstm_model, rf_model, xgb_model, scaler_features

# Main execution starts here
st.title("🌍 Air Quality Index (AQI) Prediction App")
st.write("#### Predict AQI using ML models and visualize the trends in a beautiful dashboard")

# Dark mode toggle
theme_switch = st.toggle("🌙 Dark Mode", value=False)
apply_theme(theme_switch)

# Load data and models
data, label_encoder, lstm_model, rf_model, xgb_model, scaler_features = load_data_and_train_models()

# Tabs for better navigation
tab1, tab2, tab3, tab5, tab7, tab8 ,tab9 = st.tabs([
    "Predict AQI", "AQI Trends", "City Comparison", "Climate Change Impact", "AQI Map", "Performance Analysis","Pollution Graph"
])

with tab1:
    st.subheader("Select a City:")
    city_options = [""] + list(label_encoder.classes_)
    city_name = st.selectbox("Choose a city:", options=city_options, index=0)

    if st.button("Predict AQI") and city_name:
        st.subheader(f"AQI Prediction Results for {city_name}")
        prediction_result, _ = predict_aqi_for_city(city_name, label_encoder, data, lstm_model, rf_model, xgb_model, scaler_features)
        st.write(prediction_result)

with tab2:
    if city_name:
        st.write("### AQI Trends")
        plot_aqi_trends(city_name, data, label_encoder)
        st.write("### Monthly AQI Trends")
        plot_monthly_aqi_trends(city_name, data, label_encoder)

with tab3:
    st.write("### AQI Comparison Between Cities")
    cities_to_compare = st.multiselect("Select cities to compare:", options=label_encoder.classes_)
    if len(cities_to_compare) > 1:
        plot_city_comparison(cities_to_compare, data, label_encoder)
    else:
        st.info("Select at least two cities to compare their AQI.")
with tab5:
    st.write("### Climate Change Impact Analysis")
    climate_change_impact(data)

# Function to fetch AQI data using predicted AQI values
def fetch_aqi_data(cities, predict_aqi):
    geolocator = Nominatim(user_agent="aqi_app", timeout=10)
    data = []
    for city in cities:
        try:
            location = geolocator.geocode(city)
            if location:
                aqi = predict_aqi(city)  # Use the provided function to predict AQI
                data.append({
                    "city": city,
                    "aqi": aqi,
                    "lat": location.latitude,
                    "lon": location.longitude
                })
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            st.error(f"Geocoding service error for city {city}: {e}")
    return pd.DataFrame(data)

# List of cities
cities = [
    "Ahmedabad", "Aizawl", "Amaravati", "Amritsar", "Bengaluru", "Bhopal", "Ernakulam",
    "Gurugram", "Guwahati", "Hyderabad", "Jaipur", "Jorapokhar", "Kochi", "Kolkata",
    "Lucknow", "Mumbai", "Patna", "Shillong", "Talcher", "Thiruvananthapuram",
    "Visakhapatnam", "Brajrajnagar"
]

# Dummy function to predict AQI (replace with your actual prediction function)
def predict_aqi(city):
    # Replace this with your actual AQI prediction logic
    return 100  # Dummy value

# Fetch AQI data
aqi_df = fetch_aqi_data(cities, predict_aqi)

# Existing code for tab6

with tab7:
    st.write("### AQI Map for Selected Cities")
    st.map(aqi_df)

# Add this at the end of your script to handle session state initialization
if 'city_name' not in st.session_state:
    st.session_state.city_name = ""

with tab8:
    # Load data
    df = pd.read_csv("cleaned_city_day.csv")

    # Convert 'Date' to datetime and set it as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    # Function to calculate metrics for a single target variable
    def calculate_metrics(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r_squared = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rms = np.sqrt(np.mean(y_pred**2))

        return r_squared, rmse, rms

    # Initialize a dictionary to store results
    results = {}

    # Calculate metrics for each numerical variable
    for col in numerical_cols:
        if col != 'AQI':  # Exclude AQI as it's likely a combination of other variables
            X = df[numerical_cols].drop([col, 'AQI'], axis=1)
            y = df[col]
            r_squared, rmse, rms = calculate_metrics(X, y)
            results[col] = {'R-squared': r_squared, 'RMSE': rmse, 'RMS': rms}

    # Calculate metrics for AQI using all other numerical variables
    X_aqi = df[numerical_cols].drop('AQI', axis=1)
    y_aqi = df['AQI']
    r_squared, rmse, rms = calculate_metrics(X_aqi, y_aqi)
    results['AQI'] = {'R-squared': r_squared, 'RMSE': rmse, 'RMS': rms}

    # Convert results to DataFrame
    metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})

    st.write("### Performance Analysis")
    
    # Display metrics for each model
    st.write("#### Model Performance Metrics")
    for index, row in metrics_df.iterrows():
        st.write(f"{row['Model']} - R²: {row['R-squared']:.2f}, RMSE: {row['RMSE']:.2f}")
    
    # Bar plot for R² and RMSE
    st.write("### Model Comparison")
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    
    sns.barplot(x='Model', y='R-squared', data=metrics_df, ax=ax[0])
    ax[0].set_title('R² Comparison')
    
    sns.barplot(x='Model', y='RMSE', data=metrics_df, ax=ax[1])
    ax[1].set_title('RMSE Comparison')
    
    st.pyplot(fig)
with tab9:
    st.write("### Pollution Graphs")

    # Dummy data for pollutants
    pollutants_df = pd.DataFrame({
        'Date': pd.date_range(start='1/1/2020', periods=100, freq='D'),
        'PM2.5': np.random.rand(100) * 100,
        'PM10': np.random.rand(100) * 100,
        'NO': np.random.rand(100) * 100,
        'NO2': np.random.rand(100) * 100,
        'NOx': np.random.rand(100) * 100,
        'NH3': np.random.rand(100) * 100,
        'CO': np.random.rand(100) * 100,
        'SO2': np.random.rand(100) * 100,
        'O3': np.random.rand(100) * 100,
        'Benzene': np.random.rand(100) * 100,
        'Toluene': np.random.rand(100) * 100,
        'Xylene': np.random.rand(100) * 100
    })

    # Line plot for pollutants
    st.write("### Line Plot of Pollutants Over Time")
    fig, ax = plt.subplots(figsize=(12, 8))
    pollutants_df.set_index('Date').plot(ax=ax)
    ax.set_title('Pollutants Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Concentration')
    st.pyplot(fig)

    # Box plot for pollutants
    st.write("### Box Plot of Pollutants")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=pollutants_df.drop(columns=['Date']), ax=ax)
    ax.set_title('Box Plot of Pollutants')
    ax.set_xlabel('Pollutants')
    ax.set_ylabel('Concentration')
    st.pyplot(fig)

    # Heatmap for pollutants
    st.write("### Heatmap of Pollutants")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pollutants_df.drop(columns=['Date']).corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Heatmap of Pollutants Correlations')
    st.pyplot(fig)
