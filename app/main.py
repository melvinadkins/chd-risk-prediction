# Import libraries 
from fastapi import FastAPI 
from pydantic import BaseModel, Field 
import pandas as pd
import numpy as np 
import joblib 
from src.feature_engineering import FeatureEngineering
from src.preprocessing import Winsorizer
from src.config import MODEL_PATH

# Load trained model
model = joblib.load(MODEL_PATH)

# Initialize FastAPI app
app = FastAPI(title="CHD Risk Prediction API")

# Define input data model
class PatientData(BaseModel):
    age: int = Field(..., ge=18, le=120, description="Age of the patient in years")
    male: int = Field(..., ge=0, le=1, description="Gender of the patient (0 = female, 1 = male)")
    education: int = Field(..., ge=1, le=4, description="Education level of the patient")
    currentSmoker: int = Field(..., ge=0, le=1, description="Whether the patient is a current smoker (0 = no, 1 = yes)")
    cigsPerDay: int = Field(..., ge=0, le=70, description="Number of cigarettes smoked per day")
    BPMeds: int = Field(..., ge=0, le=1, description="Whether the patient is on blood pressure medication (0 = no, 1 = yes)")
    prevalentStroke: int = Field(..., ge=0, le=1, description="Whether the patient has a history of stroke (0 = no, 1 = yes)")
    prevalentHyp: int = Field(..., ge=0, le=1, description="Whether the patient has a history of hypertension (0 = no, 1 = yes)")
    diabetes: int = Field(..., ge=0, le=1, description="Whether the patient has diabetes (0 = no, 1 = yes)")
    totChol: int = Field(..., ge=100, le=1000, description="Total cholesterol level of the patient")
    sysBP: float = Field(..., ge=80, le=300, description="Systolic blood pressure of the patient")
    diaBP: float = Field(..., ge=40, le=150, description="Diastolic blood pressure of the patient")
    BMI: float = Field(..., ge=10, le=80, description="Body mass index of the patient")
    heartRate: int = Field(..., ge=40, le=150, description="Heart rate of the patient")
    glucose: int = Field(..., ge=40, le=400, description="Glucose level of the patient")

@app.get("/")
def home():
    return {"message": "Welcome to the CHD Risk Prediction API."}

@app.post("/predict")
def predict_chd_risk(data: PatientData):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Predict CHD risk probability and binary prediction
    risk_proba = model["pipeline"].predict_proba(input_data)[:, 1][0]
    risk_prediction = int(risk_proba >= model["threshold"])
    return {
        "chd_risk_probability": float(risk_proba),
        "chd_risk_prediction": risk_prediction
    }

class MultiplePatientData(BaseModel):
    # List of patient data for batch prediction
    patients: list[PatientData]

@app.post("/predict_batch")
def predict_chd_risk_batch(data: MultiplePatientData):
    # Convert list of patient data to DataFrame
    input_data = pd.DataFrame([patient.dict() for patient in data.patients])

    # Predict CHD risk probabilities and binary predictions for batch input
    risk_proba = model["pipeline"].predict_proba(input_data)[:, 1]
    risk_prediction = (risk_proba >= model["threshold"]).astype(int).tolist()
    return {
        "chd_risk_probabilities": risk_proba.tolist(),
        "chd_risk_predictions": risk_prediction
    }