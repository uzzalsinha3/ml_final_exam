#gradio app 

import gradio as gr
import pandas as pd
import pickle
import numpy as np

#  Load the Model
with open("Diabetes Prediction System.pkl", "rb") as f:
    model = pickle.load(f)

#  The Logic Function
def predict_diabetes(Pregnancies, Glucose, BloodPressure,
                     SkinThickness, Insulin, BMI,
                     DiabetesPedigreeFunction, Age):
    
    # Pack inputs into a DataFrame
    input_df = pd.DataFrame([[
                    Pregnancies, Glucose, BloodPressure,
                    SkinThickness, Insulin, BMI,
                    DiabetesPedigreeFunction, Age

    ]],
      columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','DiabetesPedigreeFunction', 'Age'
    ])
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    return int(prediction)

#  The App Interface
# Defining inputs in a list
inputs = [
    gr.Slider(0, 20, step=1, label="Pregnancies"),
    gr.Slider(0, 300, step=1, label="Glucose"),
    gr.Slider(0, 200, step=1, label="Blood Pressure"),
    gr.Slider(0, 100, step=1, label="Skin Thickness"),
    gr.Slider(0, 900, step=1, label="Insulin"),
    gr.Slider(0, 70, step=0.1, label="BMI"),
    gr.Slider(0, 3, step=0.01, label="Diabetes Pedigree Function"),
    gr.Slider(1, 120, step=1, label="Age")
]

app = gr.Interface(
    fn=predict_diabetes,
      inputs=inputs,
        outputs= gr.Number(label="Diabetes Prediction (0 = No, 1 = Yes)"), 
        title="Diabetes Predictor")

app.launch(share=True)