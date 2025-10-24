
# sreamlit application
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

MODEL_PATH = Path("app/../models/energy_model.sav")

# loading the model
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model file not found at {MODEL_PATH}")
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model

model = load_model()

# tittle
st.title("Steel Industry Energy Consumption Predictor")
st.write("prediction energy usage (kWh) based on operational parameters.")

#user inputs
st.header("Input Parameters")

# date and time inputs
date_input = st.date_input("Date")
time_input = st.time_input("Time")
dt = datetime.combine(date_input, time_input)