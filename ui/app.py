import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# Load model and dataset
model = joblib.load('../models/final_model.pkl')
data = pd.read_csv('../data/cleaned_heart_disease.csv')  # Ensure this file exists from preprocessing

st.title('Heart Disease Risk Prediction')

# User inputs
age = st.number_input('Age', min_value=1, max_value=120, value=30)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
trestbps = st.number_input('Resting Blood Pressure', min_value=0, value=120)
chol = st.number_input('Cholesterol', min_value=0, value=200)
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, value=150)
oldpeak = st.number_input('Oldpeak', min_value=0.0, value=1.0)

cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3], format_func=lambda x: 'typical angina' if x == 0 else ('atypical angina' if x == 1 else ('non-anginal pain' if x == 2 else 'asymptomatic')))
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: 'true' if x == 1 else 'false')
restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2], format_func=lambda x: 'normal' if x == 0 else ('having ST-T wave abnormality' if x == 1 else 'showing probable or definite left ventricular hypertrophy'))
exang = st.selectbox('Exercise Induced Angina', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2], format_func=lambda x: 'upsloping' if x == 0 else ('flat' if x == 1 else 'downsloping'))
ca = st.selectbox('Number of Major Vessels (0-3) Colored by Fluoroscopy', options=[0, 1, 2, 3], format_func=lambda x: f'Vessel {x}')
thal = st.selectbox('Thalassemia', options=[3, 6, 7], format_func=lambda x: 'Normal' if x == 3 else ('Fixed Defect' if x == 6 else 'Reversible Defect'))

input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

if st.button('Predict'):
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success(f'Low risk of heart disease')
    else:
        st.error(f'High risk of heart disease')

# Data Visualization Section
st.header('Explore Heart Disease Trends')

# Histogram of Age by num
st.subheader('Age Distribution by Heart Disease Status')
fig1, ax1 = plt.subplots()
sns.histplot(data=data, x='age', hue='num', multiple='stack', ax=ax1)
ax1.set_title('Age Distribution by Heart Disease Status')
st.pyplot(fig1)

# Correlation Heatmap
st.subheader('Correlation Heatmap')
fig2, ax2 = plt.subplots(figsize=(20, 18))
numeric_data = data.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax2)
ax2.set_title('Correlation Heatmap of Numeric Features')
st.pyplot(fig2)
