
# sreamlit application
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt


# load resources model
@st.cache_resource
def load_resources(path):
    if not path.exists():
        raise FileNotFoundError(f"resource file not found at {path}")
    
    with open(path, "rb") as f:
        resource = pickle.load(f)

    return resource

# resources paths
model_path = Path("app/../models/steel_model.sav")
encoder_path = Path("app/../models/label_encoder.pkl")

# loading model and encoder
model = load_resources(model_path)
encoder = load_resources(encoder_path)

# tittle
st.title("Steel Industry Energy Consumption Predictor")
st.write("prediction energy usage (kWh) based on operational parameters.")

#user inputs
st.header("Input Parameters")

# date and time inputs
date_input = st.date_input("Date")
time_input = st.time_input("Time")
dt = datetime.combine(date_input, time_input)

# extract temporal features
hour = dt.hour
day = dt.day
month = dt.month
year = dt.year
is_weekend = int(dt.weekday() >= 5)

# numerical inputs
lagging_reactive = st.number_input("Lagging Current Reactve Power (kVarh)", value=0.0)
leading_reactive = st.number_input("Leading Current Reactive Power (kVarh)", value=0.0)
co2 = st.number_input("CO2 Emissions (tCO2)", value=0.0)
lagging_power_factor = st.number_input("Lagging Current Power Factor", value=0.0)
leading_power_factor = st.number_input("Leading Current Power Factor", value=0.0)
nsm = st.number_input("Net Standard Minutes (NSM)", value=0.0)

# categorical inputs