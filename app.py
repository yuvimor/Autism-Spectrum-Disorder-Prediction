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

# Create input fields for the features used during training
# You need to adapt this part based on the features in your training data
age = st.sidebar.number_input('Age', min_value=1, max_value=100, step=1)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
jaundice = st.sidebar.radio('Jaundice at birth', ['No', 'Yes'])
autism_family = st.sidebar.radio('Family history of autism', ['No', 'Yes'])
used_app_before = st.sidebar.radio('Used screening app before', ['No', 'Yes'])
result = st.sidebar.number_input('AQ1-10 screening test result', min_value=0, max_value=10, step=1)
relation = st.sidebar.selectbox('Relation of patient who completed the test', ['Parent', 'Relative', 'Self', 'Health care professional', 'Others'])

# Create a DataFrame with the user input
user_input = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'jaundice': [1 if jaundice == 'Yes' else 0],  # Convert to binary
    'autism': [1 if autism_family == 'Yes' else 0],  # Convert to binary
    'used_app_before': [1 if used_app_before == 'Yes' else 0],  # Convert to binary
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
