import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import xgboost as xgb
from pathlib import Path

#loading model and encoder
@st.cache_resource
def load_model():
    model_path = Path("app/../models/xgb_model.json")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    return model


@st.cache_resource
def load_encoders():
    encoder_path = Path("app/../models/label_encoders.pkl")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder file not found at {encoder_path}")
    with open(encoder_path, "rb") as f:
        encoders = pickle.load(f)
    return encoders


# preprocessing function
def preprocess_input(input_data: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = input_data.copy()

    # Extract date-time features
    df["Date"] = pd.to_datetime(df["Date"])
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S")

    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Hour"] = df["Time"].dt.hour
    df["Minute"] = df["Time"].dt.minute
    df["Second"] = df["Time"].dt.second

    # Encode categorical features
    cat_features = ["WeekStatus", "Day_of_week", "Load_Type"]
    for feature in cat_features:
        if feature in df.columns:
            le = encoders.get(feature)
            if le:
                df[feature] = le.transform(df[feature])
            else:
                raise ValueError(f"No encoder found for feature: {feature}")

    # Drop unused columns
    df.drop(columns=["Date", "Time"], inplace=True)

    return df


# streamlit ui
st.set_page_config(
    page_title="Steel Industry Energy Consumption Predictor",
    layout="wide",
    page_icon="⚙️",
)

st.title("⚙️ Steel Industry Energy Consumption Prediction")
st.markdown("Predict **energy consumption (kWh)** using operational and environmental data.")

# Sidebar for user input
st.sidebar.header("Input Parameters")

# Date and time
date_input = st.sidebar.date_input("Date")
time_input = st.sidebar.time_input("Time")

# Categorical inputs
week_status = st.sidebar.selectbox("Week Status", ["Weekday", "Weekend"])
day_of_week = st.sidebar.selectbox(
    "Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)
load_type = st.sidebar.selectbox("Load Type", ["Light Load", "Medium Load", "Maximum Load"])

# Numerical inputs
lagging_current_reactive_power = st.sidebar.number_input("Lagging Current Reactive Power (kVarh)", min_value=0.0)
leading_current_reactive_power = st.sidebar.number_input("Leading Current Reactive Power (kVarh)", min_value=0.0)
co2 = st.sidebar.number_input("CO₂ Emission (kg)", min_value=0.0)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
wind_speed = st.sidebar.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0)
visibility = st.sidebar.number_input("Visibility (km)", min_value=0.0, max_value=10.0)
dew_point = st.sidebar.number_input("Dew Point (°C)", min_value=-20.0, max_value=30.0)
apparent_temperature = st.sidebar.number_input("Apparent Temperature (°C)", min_value=-20.0, max_value=60.0)
pressure = st.sidebar.number_input("Pressure (mmHg)", min_value=500.0, max_value=1100.0)

# Combine inputs into DataFrame
input_data = pd.DataFrame(
    {
        "Date": [date_input],
        "Time": [time_input],
        "WeekStatus": [week_status],
        "Day_of_week": [day_of_week],
        "Load_Type": [load_type],
        "Lagging_Current_Reactive_Power_kVarh": [lagging_current_reactive_power],
        "Leading_Current_Reactive_Power_kVarh": [leading_current_reactive_power],
        "CO2": [co2],
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Wind_Speed": [wind_speed],
        "Visibility": [visibility],
        "Dew_Point": [dew_point],
        "Apparent_Temperature": [apparent_temperature],
        "Pressure": [pressure],
    }
)

# prediction
if st.button("Predict Energy Usage (kWh)"):
    try:
        model = load_model()
        encoders = load_encoders()

        processed_input = preprocess_input(input_data, encoders)
        prediction = model.predict(processed_input)[0]

        st.success(f"**Predicted Energy Usage:** {prediction:.2f} kWh")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
st.caption("Developed by Kami • Powered by XGBoost and Streamlit")


