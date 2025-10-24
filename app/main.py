
# sreamlit application
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from pathlib import path
import matplotlib.pyplot as plt

MODEL_PATH = path("../models/energy_model.sav")

# loading the model
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model file not found at {MODEL_PATH}")
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model

model = load_model()