import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Function to load model
def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        st.error(f"Model file not found: {model_path}")
        return None

# Load the models
crop_model = load_model('Crop.pkl')
fertilizer_model = load_model('fertilizer.pkl')

# Set page config
st.set_page_config(page_title="Crop and Fertilizer Recommendation System", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About the Project", "Crop Recommendation", "Fertilizer Recommendation"])

# About the Project page
if page == "About the Project":
    st.title("About the Crop and Fertilizer Recommendation System")
    st.write("""
    This project aims to help farmers make informed decisions about crop selection and fertilizer usage.
    
    Our system uses machine learning models to provide recommendations based on various environmental 
    and soil factors. The two main components of this system are:

    1. **Crop Recommendation**: Suggests the most suitable crop based on soil composition and environmental conditions.
    2. **Fertilizer Recommendation**: Recommends the appropriate fertilizer based on soil characteristics and crop type.

    Use the sidebar to navigate between different functionalities of the application.
    """)

# Crop Recommendation page
elif page == "Crop Recommendation":
    st.title("Crop Recommendation")
    st.write("Enter the following details to get a crop recommendation:")

    col1, col2 = st.columns(2)
    
    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, help="Amount of Nitrogen in soil")
        P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, help="Amount of Phosphorus in soil")
        K = st.number_input("Potassium (K)", min_value=5, max_value=205, help="Amount of Potassium in soil")
        temperature = st.number_input("Temperature (°C)", min_value=8.0, max_value=44.0, help="Average temperature in Celsius")

    with col2:
        humidity = st.number_input("Humidity (%)", min_value=14.0, max_value=100.0, help="Average relative humidity")
        ph = st.number_input("pH", min_value=3.5, max_value=10.0, help="pH value of the soil")
        rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=300.0, help="Average annual rainfall in mm")

    if st.button("Predict Crop"):
        if crop_model is not None:
            try:
                input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                prediction = crop_model.predict(input_data)
                st.success(f"The recommended crop is: {prediction[0]}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
        else:
            st.error("Crop recommendation model is not available.")

# Fertilizer Recommendation page
elif page == "Fertilizer Recommendation":
    st.title("Fertilizer Recommendation")
    st.write("Enter the following details to get a fertilizer recommendation:")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, help="Current temperature in Celsius")
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, help="Current relative humidity")
        moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, help="Soil moisture content")
        soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Black", "Red", "Clayey"], help="Type of soil in the field")

    with col2:
        crop_type = st.selectbox("Crop Type", ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy", "Barley", "Wheat", "Millets", "Oil seeds", "Pulses", "Ground Nuts"], help="Type of crop being grown")
        nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=140, help="Amount of Nitrogen in soil")
        potassium = st.number_input("Potassium (K)", min_value=0, max_value=205, help="Amount of Potassium in soil")
        phosphorous = st.number_input("Phosphorous (P)", min_value=0, max_value=145, help="Amount of Phosphorous in soil")

    if st.button("Predict Fertilizer"):
        if fertilizer_model is not None:
            try:
                # Convert categorical variables to numerical
                soil_type_map = {"Sandy": 0, "Loamy": 1, "Black": 2, "Red": 3, "Clayey": 4}
                crop_type_map = {"Maize": 0, "Sugarcane": 1, "Cotton": 2, "Tobacco": 3, "Paddy": 4, "Barley": 5, "Wheat": 6, "Millets": 7, "Oil seeds": 8, "Pulses": 9, "Ground Nuts": 10}

                input_data = np.array([[temperature, humidity, moisture, soil_type_map[soil_type], crop_type_map[crop_type], nitrogen, potassium, phosphorous]])
                prediction = fertilizer_model.predict(input_data)
                st.success(f"The recommended fertilizer is: {prediction[0]}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
        else:
            st.error("Fertilizer recommendation model is not available.")


# Add a footer
st.sidebar.markdown("---")
st.sidebar.info("Developed by Tuhin Kumar Singha Roy")
st.sidebar.text("Version 1.0")

