import pytest
from fastapi.testclient import TestClient
from src.api.pancre_scan_api import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "PancreScan API is running."}

def test_predict_endpoint_missing_file():
    response = client.post("/predict")
    assert response.status_code == 422 # Unprocessable Entity (missing file)
