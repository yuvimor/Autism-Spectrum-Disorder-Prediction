import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the saved Random Forest model
model_filename = 'model.joblib'
rf_model = joblib.load(model_filename)

# Streamlit App
st.title('Autism Spectrum Disorder Prediction')

# Sidebar with user input
st.sidebar.header('User Input')

# Binary features
binary_features = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'jaundice', 'austim', 'used_app_before'
]

for feature in binary_features:
    st.sidebar.checkbox(f'{feature}', key=feature)

# Non-binary features
age = st.sidebar.number_input('Age', min_value=1, max_value=100, step=1)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
ethnicity = st.sidebar.selectbox('Ethnicity', ['Category 1', 'Category 2', 'Category 3'])  # Add more categories as needed
country_of_res = st.sidebar.selectbox('Country of Residence', ['Country 1', 'Country 2', 'Country 3'])  # Add more countries as needed
result = st.sidebar.number_input('AQ1-10 screening test result', min_value=0, max_value=10, step=1)
relation = st.sidebar.selectbox('Relation of patient who completed the test', ['Parent', 'Relative', 'Self', 'Health care professional', 'Others'])

# Convert binary features to 0 or 1
binary_inputs = {feature: int(st.sidebar.checkbox(f'{feature}')) for feature in binary_features}

# Create a DataFrame with the user input
user_input = pd.DataFrame({
    **binary_inputs,
    'age': [age],
    'gender': [gender],
    'ethnicity': [ethnicity],
    'country_of_res': [country_of_res],
    'result': [result],
    'relation': [relation]
})

# Make predictions when the user clicks the button
if st.sidebar.button('Predict'):
    # Make predictions using the model
    prediction = rf_model.predict(user_input)

    # Display the prediction result
    st.write('Prediction:', 'Yes' if prediction[0] == 1 else 'No')

# Add more information or instructions as needed
st.sidebar.text('Fill in the required information and click Predict.')
