import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
DATA_URL = 'https://raw.githubusercontent.com/VishalBhagat01/ML_MODEL/main/Mental_Heaalth.csv'

# Load and train model (only once)
def train_model():
    df = pd.read_csv(DATA_URL)
    df = df.dropna()

    # Encode target
    le = LabelEncoder()
    df['treatment'] = le.fit_transform(df['treatment'])

    X = df.drop('treatment', axis=1)
    y = df['treatment']
    X_encoded = pd.get_dummies(X)

    model = RandomForestClassifier()
    model.fit(X_encoded, y)
    
    return model, X_encoded.columns

model, model_columns = train_model()

# Inference function
def predict(age, gender, country, self_employed, family_history, work_interfere):
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Country': country,
        'self_employed': self_employed,
        'family_history': family_history,
        'work_interfere': work_interfere
    }])
    
    input_encoded = pd.get_dummies(input_df)

    # Ensure all required columns exist
    for col in model_columns:
        if col not in input_encoded:
            input_encoded[col] = 0

    input_encoded = input_encoded[model_columns]

    pred = model.predict(input_encoded)[0]
    return "✅ Likely to Seek Treatment" if pred == 1 else "❌ Unlikely to Seek Treatment"

# Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(18, 100, value=30, label="Age"),
        gr.Radio(["Male", "Female", "Other"], label="Gender"),
        gr.Textbox(label="Country", value="United States"),
        gr.Radio(["Yes", "No"], label="Are you self-employed?"),
        gr.Radio(["Yes", "No"], label="Family history of mental illness?"),
        gr.Radio(["Often", "Rarely", "Never", "Sometimes"], label="Work interference with mental health?")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="🧠 Mental Health Treatment Predictor",
    description="Fill out the form to predict whether someone is likely to seek mental health treatment."
)

demo.launch()
