from fastapi.testclient import TestClient 
from app.main import app

client = TestClient(app)

def test_prediction():
    # Create sample patient data for testing
    payload = {
        "age": 18,
        "male": 1,
        "education": 1,
        "currentSmoker": 1,
        "cigsPerDay": 70,
        "BPMeds": 1,
        "prevalentStroke": 1,
        "prevalentHyp": 1,
        "diabetes": 1,
        "totChol": 100,
        "sysBP": 80,
        "diaBP": 40,
        "BMI": 10,
        "heartRate": 40,
        "glucose": 40
    }
    
    # Send POST request to /predict endpoint and validate response
    response = client.post("/predict", json=payload)
    print(response.json())
    assert response.status_code == 200

    # Validate response content
    data = response.json()
    assert "chd_risk_probability" in data 
    assert "chd_risk_prediction" in data 
    assert 0 <= data["chd_risk_probability"] <= 1
    assert data["chd_risk_prediction"] in [0, 1]

def test_batch_prediction():
    # Create sample batch patient data for testing
    payload = {
        "patients": [
            {"age": 18, "male": 1, "education": 1, "currentSmoker": 1, "cigsPerDay": 70, "BPMeds": 1, "prevalentStroke": 1, "prevalentHyp": 1, "diabetes": 1, "totChol": 100, "sysBP": 80, "diaBP": 40, "BMI": 10, "heartRate": 40, "glucose": 40},
            {"age": 50, "male": 0, "education": 2, "currentSmoker": 0, "cigsPerDay": 0, "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 0, "diabetes": 0, "totChol": 200, "sysBP": 120, "diaBP": 80, "BMI": 25, "heartRate": 70, "glucose": 90},
            {"age": 70, "male": 1, "education": 3, "currentSmoker": 0, "cigsPerDay": 0, "BPMeds": 1, "prevalentStroke": 0, "prevalentHyp": 1, "diabetes": 1, "totChol": 250, "sysBP": 140, "diaBP": 90, "BMI": 30, "heartRate": 80, "glucose": 100}
            ]
        }
    
    # Send POST request to /predict_batch endpoint and validate response
    response = client.post("/predict_batch", json=payload)
    print(response.json())
    assert response.status_code == 200

    data = response.json()
    assert "chd_risk_probabilities" in data 
    assert "chd_risk_predictions" in data 

    assert len(data["chd_risk_probabilities"]) == 3
    assert len(data["chd_risk_predictions"]) == 3

    for prob in data["chd_risk_probabilities"]:
        assert 0 <= prob <= 1

    for pred in data["chd_risk_predictions"]:
        assert pred in [0, 1]

if __name__ == "__main__":
    test_prediction()
    test_batch_prediction()
    print("All tests passed!")