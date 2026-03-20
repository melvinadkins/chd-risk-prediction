import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


NUM_FEATURES = [
    "age", "cigsPerDay", "totChol", "sysBP", 
    "diaBP", "BMI", "heartRate", "glucose", "pulse_pressure", "MAP"
]
CAT_FEATURES = [
    "male", "education", "BPMeds", "currentSmoker",
    "prevalentStroke", "prevalentHyp", "diabetes"
]

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Creates pulse_pressure and MAP features from blood pressure readings."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_fe = X.copy()
        X_fe["pulse_pressure"] = X_fe["sysBP"] - X_fe["diaBP"]
        X_fe["MAP"] = (X_fe["sysBP"] + 2 * X_fe["diaBP"]) / 3
        return X_fe
    
    def set_output(self, transform=None):
        """Set the output of the transformer to a pandas DataFrame."""
        return self