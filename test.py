import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Test prediction using form data
def test_predict_form():
    form_data = {
        "Reason_for_Absence": 1,
        "Month": 5,
        "Transportation_Expense": 120,
        "Age": 34,
        "Body_Mass_Index": 22,
        "Education": 1,
        "Children": 2,
        "Pets": 1
    }
    response = client.post("/predict", data=form_data)
    assert response.status_code == 200
    assert "Excessively Absent" in response.text or "Moderately to Not Absent" in response.text

# Test prediction using JSON data
def test_predict_json():
    json_data = {
        "Reason_for_Absence": 1,
        "Month": 5,
        "Transportation_Expense": 120,
        "Age": 34,
        "Body_Mass_Index": 22,
        "Education": 1,
        "Children": 2,
        "Pets": 1
    }
    response = client.post("/predict/json", json=json_data)
    assert response.status_code == 200
    assert "Prediction" in response.json()  # The JSON response should contain the Prediction key
    assert "Probability" in response.json()  # The JSON response should contain the Probability key

# Edge case: Test prediction using form data with invalid "Reason_for_Absence"
def test_predict_form_invalid_data():
    form_data = {
        "Reason_for_Absence": 999,  # Invalid value
        "Month": 5,
        "Transportation_Expense": 120,
        "Age": 34,
        "Body_Mass_Index": 22,
        "Education": 1,
        "Children": 2,
        "Pets": 1
    }
    response = client.post("/predict", json=form_data)
    assert response.status_code == 422  # Expecting validation error (422 Unprocessable Entity)

# Edge case: Test prediction using JSON data with missing fields
def test_predict_json_missing_fields():
    json_data = {
        "Reason_for_Absence": 1,
        "Month": 5,
        # Missing some fields here to test validation
    }
    response = client.post("/predict/json", json=json_data)
    assert response.status_code == 422  # Expecting validation error (422 Unprocessable Entity)
