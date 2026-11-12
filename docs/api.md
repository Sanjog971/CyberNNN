# Prediction API

CyberNN includes an optional FastAPI service alongside the Streamlit dashboard. It loads the tracked model and preprocessor once at startup and exposes batch predictions; it does not capture traffic or take network actions.

## Run locally (Windows PowerShell)

Install the project's requirements, then create an API key in the current PowerShell session and start the service on loopback:

```powershell
$env:CYBERNN_API_KEY = (python -c "import secrets; print(secrets.token_urlsafe(32))")
python -m uvicorn app.prediction_api:app --host 127.0.0.1 --port 8000
```

Keep the key secret. It is read only from the environment, is never returned by the API, and should not be committed or reused as a GitHub token. The API rejects prediction requests if the key is absent or shorter than 32 characters. `GET /health` is public; `/schema` and `/predict` require the `X-API-Key` header.

Open `http://127.0.0.1:8000/docs` for the interactive API documentation. The authenticated `GET /schema` endpoint lists the required feature names, default threshold, and maximum batch size (1,000 flows).

## Predict

Send a JSON body containing a `flows` array. Every flow must include every feature returned by `/schema`; values may be null or non-numeric, and are replaced using training-set medians just as in the dashboard. An optional `threshold` must be between 0 and 1; if omitted, the validation-selected model threshold is used.

Example PowerShell request using a single synthetic all-zero record (replace its values with a real flow before interpreting the result):

```powershell
$headers = @{ "X-API-Key" = $env:CYBERNN_API_KEY }
$schema = Invoke-RestMethod -Uri http://127.0.0.1:8000/schema -Headers $headers
$flow = @{}
foreach ($feature in $schema.feature_names) { $flow[$feature] = 0 }
$body = @{ flows = @($flow) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post `
  -Headers $headers -ContentType "application/json" -Body $body
```

The response contains one score and `flagged` result per flow, the threshold used, and a count of repaired values. Scores are ranking values, not calibrated probabilities or proof of malicious activity.

Input and authentication failures return JSON in a consistent shape:

```json
{
  "error": {
    "code": "missing_features",
    "message": "Every flow must include all required model features.",
    "details": {
      "rows": [{"row": 0, "columns": ["Flow Duration"]}],
      "invalid_row_count": 1
    }
  }
}
```

## Deployment notes

The API key is a simple shared-secret access control, not user accounts or per-client authorization. The included command binds only to the local machine. Before exposing the service beyond loopback, deploy behind HTTPS with request-size and rate limits, store the key in the deployment platform's secret manager, and restrict network access. Do not expose the API key in a browser app or commit it to the repository.
