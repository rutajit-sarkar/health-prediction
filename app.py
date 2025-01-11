import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load the model from a pickle file
model_path = "ensemble_model.pkl"  # Replace with your pickle file path

with open(model_path, "rb") as f:
    model = pickle.load(f)

#ADD ALL PREPROCESSING STEP IN THIS FUNCTION

# Function for scaling and prediction
def predict(features, X_train):
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit the scaler to the training data

    # Reshape the input features into a 2D array for scaling
    features = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features)  # Apply scaling

    prediction = model.predict(scaled_features)  # Predict
    return prediction[0]


# Sidebar for navigation
st.sidebar.title("Cancer Prediction")
page = st.sidebar.selectbox("Navigate", ["Home", "Breast Cancer Prediction","Disease 2 Prediction", "Disease 3 Prediction"])

# Home Page
if page == "Home":
    st.title("Disease Prediction App")
    st.write("""
    The model is trained using breast cancer data and leverages machine learning techniques for high accuracy.
    Use the **Navigate** page to get started!
    """)

# Prediction Page
if page == "Breast Cancer Prediction":
    st.title("Breast Cancer Prediction")

    # Input fields
    st.subheader("Enter the required inputs:")
    radius_mean = st.number_input("Radius Mean", value=0.0)
    texture_mean = st.number_input("Texture Mean", value=0.0)
    perimeter_mean = st.number_input("Perimeter Mean", value=0.0)
    area_mean = st.number_input("Area Mean", value=0.0)
    smoothness_mean = st.number_input("Smoothness Mean", value=0.0)
    compactness_mean = st.number_input("Compactness Mean", value=0.0)
    concavity_mean = st.number_input("Concavity Mean", value=0.0)
    concave_points_mean = st.number_input("Concave Points Mean", value=0.0)
    symmetry_mean = st.number_input("Symmetry Mean", value=0.0)
    radius_se = st.number_input("Radius SE", value=0.0)
    radius_worst = st.number_input("Radius Worst", value=0.0)
    texture_worst = st.number_input("Texture Worst", value=0.0)
    perimeter_worst = st.number_input("Perimeter Worst", value=0.0)
    area_worst = st.number_input("Area Worst", value=0.0)
    smoothness_worst = st.number_input("Smoothness Worst", value=0.0)
    compactness_worst = st.number_input("Compactness Worst", value=0.0)
    concavity_worst = st.number_input("Concavity Worst", value=0.0)
    concave_points_worst = st.number_input("Concave Points Worst", value=0.0)
    symmetry_worst = st.number_input("Symmetry Worst", value=0.0)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=0.0)

    # Placeholder for training data (to fit the scaler)
    # Replace this with the actual training dataset used during model development
    X_train = np.random.rand(100, 21)  # Dummy training data, replace with real training data

    # Collect inputs
    inputs = [
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        radius_se,
        radius_worst,
        texture_worst,
        perimeter_worst,
        area_worst,
        smoothness_worst,
        compactness_worst,
        concavity_worst,
        concave_points_worst,
        symmetry_worst,
        fractal_dimension_worst,
    ]
  # Placeholder for training data (to fit the scaler)
    # Replace this with the actual training dataset used during model development
    X_train = np.random.rand(100, 21)  # Dummy training data, replace with real training data

    # Prediction button
    if st.button("Predict"):
        prediction = predict(inputs)
        if prediction == 1:
            st.success("The tumor is predicted to be **Malignant**.")
        else:
            st.success("The tumor is predicted to be **Benign**.")
