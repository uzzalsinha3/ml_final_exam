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
    
    # Return formatted result (Clipped 0-5)
    return int(prediction)

#  The App Interface
# Defining inputs in a list
inputs = [
    gr.Number(label="Pregnancies"),
    gr.Number(label="Glucose"),
    gr.Number(label="Bloodpressure"),
    gr.Number(label="Skinthickness"),
    gr.Number(label="Insulin"),
    gr.Number(label="BMI"),
    gr.Number(label="DiabetesPedigreeFunction"),
    gr.Number(label="Age")
]

app = gr.Interface(
    fn=predict_diabetes,
      inputs=inputs,
        outputs= gr.Number(label="Diabetes Prediction (0 = No, 1 = Yes)"), 
        title="Diabetes Predictor")

app.launch(share=True)