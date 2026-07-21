import os
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Set page title and layout
st.set_page_config(
    page_title="Calorie Burn Predictor",
    page_icon="🔥",
    layout="centered"
)

# Title and description
st.title("🔥 Calorie Burn Predictor")
st.write(
    "Estimate the number of calories burned during your workout based on your personal metrics and activity data."
)

# Function to load the model (cached for performance)
@st.cache_resource
def load_trained_model():
    # Looks for the new joblib file first!
    if os.path.exists("calorie_model.joblib"):
        return joblib.load("calorie_model.joblib")
    elif os.path.exists("calorie_model.pkl.gz"):
        import gzip
        with gzip.open("calorie_model.pkl.gz", "rb") as f:
            return pickle.load(f)
    elif os.path.exists("calorie_model.pkl"):
        with open("calorie_model.pkl", "rb") as f:
            return pickle.load(f)
    else:
        return None

model = load_trained_model()

# Sidebar for User Inputs
st.sidebar.header("📊 Input User Parameters")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age (years)", min_value=10, max_value=90, value=25)
height = st.sidebar.slider("Height (cm)", min_value=100, max_value=220, value=170)
weight = st.sidebar.slider("Weight (kg)", min_value=30, max_value=150, value=70)
duration = st.sidebar.slider("Workout Duration (minutes)", min_value=1, max_value=180, value=30)
heart_rate = st.sidebar.slider("Average Heart Rate (bpm)", min_value=60, max_value=200, value=110)
body_temp = st.sidebar.slider("Body Temperature (°C)", min_value=36.0, max_value=42.0, value=38.5, step=0.1)

# Encode gender to numeric (Male = 0, Female = 1)
gender_encoded = 0 if gender == "Male" else 1

# Display entered parameters on main screen
st.subheader("Your Workout Profile")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Duration", value=f"{duration} min")
    st.metric(label="Gender", value=gender)
with col2:
    st.metric(label="Avg Heart Rate", value=f"{heart_rate} bpm")
    st.metric(label="Age / Weight", value=f"{age} yrs / {weight} kg")
with col3:
    st.metric(label="Body Temp", value=f"{body_temp} °C")
    st.metric(label="Height", value=f"{height} cm")

st.markdown("---")

# Prediction section
if st.button("🔥 Calculate Calories Burned", use_container_width=True):
    if model is None:
        st.error(
            "Model file not found! Please ensure `calorie_model.joblib` is uploaded to your GitHub repository."
        )
    else:
        features = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]])
        prediction = model.predict(features)[0]
        st.success(f"### Estimated Calories Burned: **{prediction:.2f} kcal**")
        st.balloons()
