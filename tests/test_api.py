import json

import pytest
from fastapi.testclient import TestClient

from cybernn.preprocessing import Preprocessor
from app.prediction_api import API_KEY_ENV, ROOT, create_app

API_KEY = "test-api-key-" + "x" * 32


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, API_KEY)
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def complete_flow():
    preprocessor = Preprocessor.load(ROOT / "models" / "preprocessor.json")
    return {name: 0 for name in preprocessor.feature_names}


def test_health_is_public(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_requires_a_valid_api_key(client, complete_flow):
    response = client.post("/predict", json={"flows": [complete_flow]})
    wrong_key_response = client.post(
        "/predict",
        headers={"X-API-Key": "wrong-api-key-" + "y" * 32},
        json={"flows": [complete_flow]},
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert wrong_key_response.status_code == 401
    assert wrong_key_response.json()["error"]["code"] == "unauthorized"


def test_prediction_returns_scores_and_repair_counts(client, complete_flow):
    complete_flow["Flow Duration"] = None
    response = client.post(
        "/predict",
        headers={"X-API-Key": API_KEY},
        json={"flows": [complete_flow], "threshold": 0.0},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["rows_scored"] == 1
    assert result["threshold"] == 0.0
    assert result["repaired"]["missing_cells"] == 1
    assert result["predictions"][0]["flagged"] is True
    assert 0.0 <= result["predictions"][0]["score"] <= 1.0


def test_prediction_rejects_missing_features(client):
    response = client.post(
        "/predict",
        headers={"X-API-Key": API_KEY},
        json={"flows": [{"Flow Duration": 12}]},
    )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "missing_features"
    assert error["details"]["invalid_row_count"] == 1


@pytest.mark.parametrize("payload", [
    {"flows": []},
    {"flows": [{}], "threshold": 1.1},
    {"flows": [{}], "unexpected": True},
])
def test_prediction_returns_structured_validation_errors(client, payload):
    response = client.post("/predict", headers={"X-API-Key": API_KEY}, json=payload)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]
    json.dumps(error)


def test_schema_requires_authentication_and_lists_model_features(client):
    unauthorized = client.get("/schema")
    authorized = client.get("/schema", headers={"X-API-Key": API_KEY})

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert len(authorized.json()["feature_names"]) == 66
    assert authorized.json()["max_batch_size"] == 1000


def test_prediction_fails_closed_when_api_key_is_not_configured(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with TestClient(create_app()) as test_client:
        response = test_client.post("/predict", json={"flows": [{}]})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "api_key_not_configured"
