import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Raw GitHub dataset link
DATA_URL = 'https://raw.githubusercontent.com/VishalBhagat01/ML_MODEL/main/Mental_Heaalth.csv'

# Title
st.title("🧠 Mental Health Treatment Predictor")
st.write("This app predicts whether a person is likely to seek treatment for mental health issues based on basic demographic and work-related inputs.")

# Load and train model
@st.cache(allow_output_mutation=True)
def load_and_train():
    df = pd.read_csv(DATA_URL)
    df = df.dropna()

    # Encode target
    le = LabelEncoder()
    df['treatment'] = le.fit_transform(df['treatment'])

    # Features and target
    X = df.drop('treatment', axis=1)
    y = df['treatment']

    # Encode categorical features
    X = pd.get_dummies(X)

    # Train model
    model = RandomForestClassifier()
    model.fit(X, y)

    return model, X.columns

model, model_columns = load_and_train()

# Input form
with st.form("prediction_form"):
    age = st.slider("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    country = st.text_input("Country", "United States")
    self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
    family_history = st.selectbox("Any family history of mental illness?", ["Yes", "No"])
    work_interfere = st.selectbox("Does your work interfere with mental health?", ["Often", "Rarely", "Never", "Sometimes"])
    
    submit = st.form_submit_button("Predict")

# On form submission
if submit:
    user_input = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Country': country,
        'self_employed': self_employed,
        'family_history': family_history,
        'work_interfere': work_interfere
    }])

    input_encoded = pd.get_dummies(user_input)

    # Add any missing columns
    for col in model_columns:
        if col not in input_encoded:
            input_encoded[col] = 0

    # Match column order
    input_encoded = input_encoded[model_columns]

    # Make prediction
    prediction = model.predict(input_encoded)[0]
    st.success("✅ Likely to Seek Treatment" if prediction == 1 else "❌ Unlikely to Seek Treatment")
