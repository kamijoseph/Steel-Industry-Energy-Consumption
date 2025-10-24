import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Steel Industry Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# Cache the model and encoders for better performance
@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    model = XGBRegressor()
    model.load_model("app/../models/xgb_model.json")
    return model

@st.cache_resource
def load_encoders():
    """Load the label encoders"""
    with open("app/../models/label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return encoders

# Load model and encoders
try:
    model = load_model()
    encoders = load_encoders()
except Exception as e:
    st.error(f"Error loading model or encoders: {e}")
    st.stop()

def preprocess_input(input_dict, encoders):
    """Preprocess the input data using the saved encoders"""
    processed_data = input_dict.copy()
    
    # Encode categorical features
    cat_features = ["WeekStatus", "Day_of_week", "Load_Type"]
    for feature in cat_features:
        processed_data[feature] = encoders[feature].transform([processed_data[feature]])[0]
    
    return processed_data

def create_features_from_datetime(date_input, time_input):
    """Create time-based features from date and time inputs"""
    # Combine date and time
    datetime_obj = datetime.combine(date_input, time_input)
    
    # Extract features
    features = {
        "hour": datetime_obj.hour,
        "day": datetime_obj.day,
        "month": datetime_obj.month,
        "year": datetime_obj.year,
        "is_weekend": 1 if datetime_obj.weekday() >= 5 else 0
    }
    
    # Calculate NSM (Number of Seconds from Midnight)
    nsm = datetime_obj.hour * 3600 + datetime_obj.minute * 60 + datetime_obj.second
    
    return features, nsm

def main():
    # Header
    st.title("⚡ Steel Industry Energy Consumption Predictor")
    st.markdown("""
    This application predicts energy consumption (in kWh) for steel industry operations 
    based on various operational parameters and time-based features.
    """)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📅 Date & Time Information")
        
        # Date and time inputs
        date_input = st.date_input("Select Date", value=datetime.now())
        time_input = st.time_input("Select Time", value=time(0, 15))
        
        # Calculate NSM and time features
        time_features, nsm = create_features_from_datetime(date_input, time_input)
        
        # Display calculated NSM
        st.info(f"**Calculated NSM (Seconds from Midnight):** {nsm}")
        
        # Load Type selection
        st.subheader("🔧 Load Type")
        load_type = st.selectbox(
            "Select Load Type",
            options=encoders["Load_Type"].classes_,
            help="Type of electrical load"
        )
    
    with col2:
        st.header("📊 Operational Parameters")
        
        # Numerical inputs - matching your actual data ranges
        lagging_reactive_power = st.slider(
            "Lagging Current Reactive Power (kVarh)",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.1,
            help="Range: 0-10 kVarh based on data sample"
        )
        
        leading_reactive_power = st.slider(
            "Leading Current Reactive Power (kVarh)",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
            help="Range: 0-5 kVarh based on data sample"
        )
        
        co2 = st.slider(
            "CO2 Emissions (tCO2)",
            min_value=0.0,
            max_value=0.1,
            value=0.0,
            step=0.001,
            help="Range: 0-0.1 tCO2 based on data sample"
        )
        
        lagging_power_factor = st.slider(
            "Lagging Current Power Factor",
            min_value=60.0,
            max_value=100.0,
            value=70.0,
            step=0.1,
            help="Range: 60-100 based on data sample"
        )
        
        leading_power_factor = st.slider(
            "Leading Current Power Factor",
            min_value=90.0,
            max_value=100.0,
            value=100.0,
            step=0.1,
            help="Range: 90-100 based on data sample"
        )
    
    # Determine WeekStatus and Day_of_week from date
    datetime_obj = datetime.combine(date_input, time_input)
    day_of_week = datetime_obj.strftime("%A")
    
    # Map WeekStatus (your data uses "Weekday"/"Weekend")
    week_status = "Weekend" if datetime_obj.weekday() >= 5 else "Weekday"
    
    # Create input dictionary matching your model's expected features
    input_data = {
        "Lagging_Current_Reactive.Power_kVarh": lagging_reactive_power,
        "Leading_Current_Reactive_Power_kVarh": leading_reactive_power,
        "CO2(tCO2)": co2,
        "Lagging_Current_Power_Factor": lagging_power_factor,
        "Leading_Current_Power_Factor": leading_power_factor,
        "NSM": nsm,
        "WeekStatus": week_status,
        "Day_of_week": day_of_week,
        "Load_Type": load_type,
        "hour": time_features["hour"],
        "day": time_features["day"],
        "month": time_features["month"],
        "year": time_features["year"],
        "is_weekend": time_features["is_weekend"]
    }
    
    # Display the input summary
    st.header("📋 Input Summary")
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.write("**Date & Time:**")
        st.write(f"- Date: {date_input}")
        st.write(f"- Time: {time_input}")
        st.write(f"- Day of Week: {day_of_week}")
        st.write(f"- Week Status: {week_status}")
        st.write(f"- Load Type: {load_type}")
        st.write(f"- NSM: {nsm} seconds")
    
    with summary_col2:
        st.write("**Operational Parameters:**")
        st.write(f"- Lagging Reactive Power: {lagging_reactive_power} kVarh")
        st.write(f"- Leading Reactive Power: {leading_reactive_power} kVarh")
        st.write(f"- CO2 Emissions: {co2} tCO2")
        st.write(f"- Lagging Power Factor: {lagging_power_factor}")
        st.write(f"- Leading Power Factor: {leading_power_factor}")
        st.write(f"- Hour: {time_features['hour']}")
        st.write(f"- Day: {time_features['day']}")
        st.write(f"- Month: {time_features['month']}")
    
    
    # Prediction button
    st.markdown("---")
    if st.button("🚀Predict Energy Consumption", use_container_width=True):
        try:
            # Preprocess the input data
            processed_data = preprocess_input(input_data, encoders)
            
            feature_names = [
                'Lagging_Current_Reactive.Power_kVarh',
                'Leading_Current_Reactive_Power_kVarh',
                'CO2(tCO2)',
                'Lagging_Current_Power_Factor',
                'Leading_Current_Power_Factor',
                'NSM',
                'WeekStatus',
                'Day_of_week',
                'Load_Type',
                'hour',
                'day',
                'month',
                'year',
                'is_weekend'
            ]
            
            # Create DataFrame with the correct feature order
            input_df = pd.DataFrame([processed_data])
            
            # Reorder columns to match training data
            input_df = input_df[feature_names]
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Display results
            st.success(f"## 🔮 Predicted Energy Consumption: **{prediction:.2f} kWh**")
            
            # Show confidence interval based on model performance
            st.info(f"""
            **📊 Prediction Confidence:**
            - Based on model performance: R² = 0.9986, RMSE = 1.60
            - Expected range: ± 1.6 kWh around prediction
            - Actual values in training data ranged from ~3-25 kWh
            """)
            
            # Show what the prediction means
            if prediction < 5:
                load_level = "Light Load"
            elif prediction < 15:
                load_level = "Medium Load"
            else:
                load_level = "Heavy Load"
                
            st.write(f"**📈 Load Level Classification:** {load_level}")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            import traceback
            st.error(f"Full error traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()