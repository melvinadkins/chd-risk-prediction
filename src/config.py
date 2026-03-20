import os 

RANDOM_STATE = 42 
THRESHOLD = 0.39 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "chd_risk_model.joblib")