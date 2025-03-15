"Streamlite working thanks to -pip install --upgrade streamlit google-ads- and -hash -r-"

import streamlit as st
import requests
import numpy as np
import json
import os
import pickle

# 🔹 API URL for BentoML service
BENTO_API_URL = "http://localhost:3000/classify"
METADATA_FILE = "/home/oihane/00_ToNoWaste/BentoML/Test_Automatic/models/latest/metadata.json"
MODEL_FILE = "/home/oihane/00_ToNoWaste/BentoML/Test_Automatic/models/latest/model.pkl"

# 🔹 Load model metadata from metadata.json
def load_model_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        return (
            metadata.get("version", "Unknown"), 
            metadata.get("accuracy", "Unknown"),
            metadata.get("timestamp", "Unknown"),
            metadata.get("algorithm", "Unknown")  # Ensure algorithm is stored in metadata
        )
    return "Unknown", "Unknown", "Unknown", "Unknown"

# 🔹 Extract the algorithm from `model.pkl`
def extract_model_algorithm():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        return type(model).__name__  # Algorithm name
    return "Unknown"

# 🔹 Extract Model Parameters
def extract_model_params():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        return model.get_params() if hasattr(model, "get_params") else "No parameters available"
    return "Not available"

# 🔹 Get Model Information
model_version, model_accuracy, model_timestamp, metadata_algorithm = load_model_metadata()
model_algorithm = extract_model_algorithm()
model_params = extract_model_params()

# 🔹 If `metadata.json` doesn't contain an algorithm name, use the detected one
algorithm_used = metadata_algorithm if metadata_algorithm != "Unknown" else model_algorithm

# 🔹 Streamlit UI
st.title("🌸 Iris Classification - BentoML")

# Model Info
st.sidebar.header("🔍 Model Information")
st.sidebar.write(f"**Version:** {model_version}")
st.sidebar.write(f"**Algorithm Used:** {algorithm_used}")
st.sidebar.write(f"**Accuracy:** {model_accuracy}")
st.sidebar.write(f"**Last Updated:** {model_timestamp}")

# Model Details
st.sidebar.subheader("⚙️ Model Parameters")
st.sidebar.json(model_params)

# User Input
st.header("🔬 Enter Features for Prediction")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Prediction Button
if st.button("🔮 Predict"):
    try:
        response = requests.post(BENTO_API_URL, json=input_data.tolist())
        if response.status_code == 200:
            prediction = response.json()
            st.success(f"🌼 Predicted Class: **{prediction}**")
        else:
            st.error("🚨 Error getting prediction.")
    except Exception as e:
        st.error(f"⚠️ API connection failed: {e}")

st.sidebar.write("📡 **BentoML API:** Running on `localhost:3000`")