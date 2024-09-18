import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("RandomForestModel.pkl")
model1 = joblib.load("RandomForestClassifier.pkl")

# Define the app structure
st.title("Medical Insurance Claim Prediction")

# Organize input fields into two columns
col1, col2 = st.columns(2)

# Input fields for numerical features in the first column
with col1:
    age = st.number_input("Age", min_value=12, max_value=110, value=30)
    weight = st.number_input("Weight", min_value=10, max_value=150, value=70)
    bloodpressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
    no_of_dependents = st.selectbox("Number of Dependents", options=list(range(6)))

    # Dropdown for categorical feature
    sex = st.selectbox("Sex", options=["Male", "Female"])

with col2:
    smoker = st.selectbox("Smoker", options=["Yes", "No"])
    diabetes = st.selectbox("Diabetes", options=["Yes", "No"])
    regular_ex = st.selectbox("Regular Exercise", options=["Yes", "No"])
    
    # Dropdowns for categorical features
    hereditary_diseases = st.selectbox("Hereditary Diseases", 
        ['NoDisease', 'Diabetes', 'Alzheimer', 'Obesity', 'EyeDisease', 'Cancer', 
        'Arthritis', 'HeartDisease', 'Epilepsy', 'High BP'])
    city = st.selectbox("City", 
        ['NewOrleans', 'Nashville', 'Charleston', 'Brimingham', 'Memphis', 'Louisville',
        'Charlotte', 'Stamford', 'Newport', 'Harrisburg', 'Providence', 'Worcester', 
        'WashingtonDC', 'Atlanta', 'SanJose', 'Georgia', 'Houston', 'Raleigh', 'Oklahoma', 
        'LasVegas', 'Tucson', 'SanLuis', 'Kingman', 'Denver', 'Oxnard', 'SanDeigo', 'Oceanside', 
        'Carlsbad', 'Montrose', 'LosAngeles', 'Fresno', 'Reno', 'Pheonix', 'SantaFe', 'SilverCity', 
        'Mexicali', 'Bakersfield', 'Lovelock', 'Boston', 'NewYork', 'Phildelphia', 'Pittsburg', 
        'Prescott', 'Hartford', 'Portland', 'Cambridge', 'Springfield', 'Buffalo', 'AtlanticCity', 
        'Columbus', 'Rochester', 'Miami', 'Kingsport', 'PanamaCity', 'Florence', 'Knoxville', 
        'Tampa', 'Cleveland', 'Canton', 'IowaCity', 'Fargo', 'Marshall', 'Mandan', 'Waterloo', 
        'Columbia', 'Indianapolis', 'Cincinnati', 'Bloomington', 'KanasCity', 'FallsCity', 'Lincoln', 
        'Chicago', 'Minot', 'Brookings', 'Salina', 'GrandForks', 'Escabana', 'SanFrancisco', 
        'Eureka', 'SantaRosa', 'Youngstown', 'JeffersonCity', 'Minneapolis', 'Macon', 'Huntsville', 
        'Orlando', 'Warwick', 'Trenton', 'York', 'Baltimore', 'Syracuse'])
    job_title = st.selectbox("Job Title", 
        ['Student', 'HomeMakers', 'Singer', 'Actor', 'FilmMaker', 'Dancer', 'HouseKeeper', 
        'Manager', 'Police', 'Photographer', 'Beautician', 'CEO', 'Engineer', 'FashionDesigner', 
        'Politician', 'Accountant', 'Clerks', 'Architect', 'ITProfessional', 'DataScientist', 'Lawyer', 
        'Academician', 'Doctor', 'DefencePersonnels', 'Technician', 'Chef', 'FilmDirector', 'Blogger', 
        'Journalist', 'CA', 'Farmer', 'Analyst', 'GovEmployee', 'Buisnessman', 'Labourer'])

# Convert the categorical inputs to appropriate format for prediction
sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
regular_ex = 1 if regular_ex == "Yes" else 0

# Create the input data for prediction
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'weight': [weight],
    'bmi': [bmi],
    'hereditary_diseases': [hereditary_diseases],
    'no_of_dependents': [no_of_dependents],
    'smoker': [smoker],
    'city': [city],
    'bloodpressure': [bloodpressure],
    'diabetes': [diabetes],
    'regular_ex': [regular_ex],
    'job_title': [job_title]
})

# When the user clicks on Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Insurance Claim: ${prediction[0]:.2f}")
    prediction = model1.predict(input_data)
    st.success(f"Predicted Claim Class: {prediction[0]}")
