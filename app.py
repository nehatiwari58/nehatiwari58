import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("💼 Income Classification Predictor")
st.write("This app predicts whether income is >50K or <=50K using Logistic Regression.")

# --- USER INPUTS USING SLIDERS ---
age = st.slider("Age", min_value=0, max_value=100, value=30)
education_num = st.slider("Education Number", min_value=0, max_value=20, value=10)
hours_per_week = st.slider("Hours Per Week", min_value=0, max_value=100, value=40)
capital_gain = st.slider("Capital Gain", min_value=0, max_value=10000, value=0, step=100)
capital_loss = st.slider("Capital Loss", min_value=0, max_value=5000, value=0, step=100)

# --- CATEGORICAL INPUTS ---
workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov"])
education = st.selectbox("Education", ["Bachelors", "HS-grad", "11th", "Masters", "Some-college", "Assoc-acdm"])
marital_status = st.selectbox("Marital Status", ["Married-civ-spouse", "Divorced", "Never-married", "Separated"])
occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners"])
relationship = st.selectbox("Relationship", ["Husband", "Not-in-family", "Wife", "Unmarried"])
race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Other"])
sex = st.selectbox("Sex", ["Male", "Female"])
native_country = st.selectbox("Native Country", ["United-States", "Mexico", "India", "Philippines", "Other"])

# --- PREDICT BUTTON ---
if st.button("Predict Income"):
    # Create DataFrame from inputs
    input_dict = {
        "age": age,
        "education-num": education_num,
        "hours-per-week": hours_per_week,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "workclass": workclass,
        "education": education,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "native-country": native_country
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encode categorical columns
    input_encoded = pd.get_dummies(input_df)

    # Match model’s training columns (assumes model trained on same categories)
    all_columns = model.feature_names_in_
    input_encoded = input_encoded.reindex(columns=all_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Income: {result}")

