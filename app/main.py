
# sreamlit application
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import xgboost as xgb


# load resources model
@st.cache_resource
def load_resources(path: Path, resource_type: str):
    if not path.exists():
        raise FileNotFoundError(f"resource file not found at {path}")
    
    try:
        if resource_type == "pickle":
            with open(path, "rb") as f:
                resource = pickle.load(f)
        
        elif resource_type == "xgboost":
            resource = xgb.XGBRegressor()
            resource.load_model(str(path))

        else:
            raise ValueError(f"unknown resource_type: {resource_type}")
        
    except Exception as e:
        raise RuntimeError(f"failed to  load resources from {path}: {e}")

    return resource

# resources paths
model_path = Path("app/../models/xgboost_energy_model.json")
encoder_path = Path("app/../models/label_encoder.pkl")

# loading model and encoder
model = load_resources(model_path, "xgboost")
encoder = load_resources(encoder_path, "pickle")

def preprocess_input(df, encoder):
    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year

    # Extract hour from Time
    df["hour"] = df["Time"].apply(lambda x: int(x.split(":")[0]) if isinstance(x, str) else int(x))

    # Derive temporal features
    df["Day_of_week"] = df["Date"].dt.day_name()
    df["WeekStatus"] = df["Day_of_week"].apply(lambda x: "Weekend" if x in ["Saturday", "Sunday"] else "Weekday")
    df["is_weekend"] = df["WeekStatus"].apply(lambda x: 1 if x == "Weekend" else 0)
    df["NSM"] = df["hour"] * 3600

    # Encode categorical features
    cat_features = ["WeekStatus", "Day_of_week", "Load_Type"]

    for col in cat_features:
        le = encoder[col]
        df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ 
                                else le.transform([le.classes_[0]])[0])  # unseen label fallback

    # Drop raw date/time
    df.drop(columns=["Date", "Time"], inplace=True)

    return df


# ============================================================
# Streamlit interface
# ============================================================
st.set_page_config(page_title="Steel Industry Energy Predictor", page_icon="⚡", layout="centered")

st.title("⚙️ Steel Industry Energy Consumption Predictor")
st.markdown(
    """
    This Streamlit app uses a trained **XGBoost model** to predict steel industry energy consumption.  
    Provide input values below and get an instant prediction.
    """
)

# -----------------------
# Input form
# -----------------------
with st.form("energy_form"):
    st.subheader("Enter Input Parameters")

    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Date")
        time = st.text_input("Time (HH:MM)", "08:00")
        lag_reactive = st.number_input("Lagging Current Reactive Power (kVarh)", min_value=0.0, value=120.0)
        lead_reactive = st.number_input("Leading Current Reactive Power (kVarh)", min_value=0.0, value=60.0)
    with col2:
        lag_pf = st.number_input("Lagging Current Power Factor", min_value=0.0, max_value=1.0, value=0.9)
        lead_pf = st.number_input("Leading Current Power Factor", min_value=0.0, max_value=1.0, value=0.7)
        co2 = st.number_input("CO2 Emission (tCO2)", min_value=0.0, value=2.5)
        load_type = st.selectbox("Load Type", ["Light_Load", "Medium_Load", "Heavy_Load"])

    submit = st.form_submit_button("Predict Energy Usage")

# -----------------------
# Prediction
# -----------------------
if submit:
    # Construct single-row DataFrame
    input_data = pd.DataFrame({
        "Date": [date],
        "Time": [time],
        "Lagging_Current_Reactive.Power_kVarh": [lag_reactive],
        "Leading_Current_Reactive_Power_kVarh": [lead_reactive],
        "Lagging_Current_Power_Factor": [lag_pf],
        "Leading_Current_Power_Factor": [lead_pf],
        "CO2(tCO2)": [co2],
        "Load_Type": [load_type]
    })

    # Preprocess input
    processed_input = preprocess_input(input_data, encoder)

    # Prediction
    prediction = model.predict(processed_input)[0]

    st.success(f"### ⚡ Predicted Energy Usage: {prediction:.3f} kWh")

    st.caption("Model: XGBoost | Encoder: Saved categorical transformer used during training")