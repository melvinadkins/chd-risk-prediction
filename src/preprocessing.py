import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 

from src.feature_engineering import (
    FeatureEngineering,
    NUM_FEATURES,
    CAT_FEATURES
)

class Winsorizer(BaseEstimator, TransformerMixin):
    """Clips numerical features to [lower_quantile, upper_quantile] bounds
    fit on training data, preventing outliers from distorting the scaler."""

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.lower_bounds_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def set_output(self, transform=None):
        return self
    
def build_pipeline(model):
    """Builds the full feature engineering, preprocessing, and model pipeline.

    Parameters:
    -----------
    model: LogisticRegression instance
        The logistic regression model to be included in the pipeline.

    Returns:
    ---------
    Pipeline
        A scikit-learn Pipeline object that includes feature engineering,
        winsorization, imputation, scaling, and the logistic regression model.
    """
    return Pipeline([
        ("feature_engineering", FeatureEngineering()),
        ("preprocessor", ColumnTransformer([
            ("num", Pipeline([
                ("imputation", SimpleImputer(strategy="median")),
                ("winsorizer", Winsorizer()),
                ("scaler", StandardScaler()),
            ]), NUM_FEATURES),
            ("cat", SimpleImputer(strategy="most_frequent"), CAT_FEATURES),
        ])),
        ("model", model),
    ]).set_output(transform="pandas")