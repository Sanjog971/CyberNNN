"""Authenticated FastAPI service for scoring network-flow records."""

import os
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt
from starlette.exceptions import HTTPException as StarletteHTTPException

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cybernn.model import load_model, predict_scores
from cybernn.preprocessing import Preprocessor
from cybernn.validation import repair_report

MODEL_PATH = ROOT / "models" / "cybernn_mlp.pt"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.json"
API_KEY_ENV = "CYBERNN_API_KEY"
MAX_BATCH_SIZE = 1_000
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flows: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_BATCH_SIZE)
    threshold: StrictFloat | StrictInt | None = Field(default=None, ge=0.0, le=1.0)


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str, details: Any = None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def create_app():
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.preprocessor = Preprocessor.load(PREPROCESSOR_PATH)
        app.state.model, app.state.config = load_model(MODEL_PATH)
        yield

    app = FastAPI(
        title="CyberNN Prediction API",
        description="Score network-flow records with the CyberNN research model.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, error: APIError):
        body = {"code": error.code, "message": error.message}
        if error.details is not None:
            body["details"] = error.details
        return JSONResponse(
            status_code=error.status_code,
            content={"error": body, "path": request.url.path},
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request: Request, error: RequestValidationError):
        details = [
            {"location": list(item["loc"]), "message": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"error": {
                "code": "validation_error",
                "message": "The request body is invalid.",
                "details": details,
            }, "path": request.url.path},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, error: StarletteHTTPException):
        code = "not_found" if error.status_code == 404 else "http_error"
        message = "The requested resource was not found." if error.status_code == 404 else str(error.detail)
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": code, "message": message}, "path": request.url.path},
            headers=error.headers,
        )

    def require_api_key(
        api_key: Annotated[str | None, Depends(API_KEY_HEADER)],
    ) -> None:
        configured_key = os.environ.get(API_KEY_ENV)
        if not configured_key or len(configured_key) < 32:
            raise APIError(
                503,
                "api_key_not_configured",
                f"Set {API_KEY_ENV} to a secret of at least 32 characters.",
            )
        if api_key is None or not secrets.compare_digest(api_key, configured_key):
            raise APIError(401, "unauthorized", "A valid X-API-Key header is required.")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/schema", dependencies=[Depends(require_api_key)])
    def schema(request: Request):
        return {
            "feature_names": request.app.state.preprocessor.feature_names,
            "max_batch_size": MAX_BATCH_SIZE,
            "default_threshold": float(request.app.state.config["threshold_selected_on_validation"]),
            "score_semantics": "ranking score, not a calibrated probability",
        }

    @app.post("/predict", dependencies=[Depends(require_api_key)])
    def predict(payload: PredictionRequest, request: Request):
        preprocessor = request.app.state.preprocessor
        normalized_rows = []
        for index, row in enumerate(payload.flows):
            normalized = {}
            for column, value in row.items():
                name = column.strip()
                if name in normalized:
                    raise APIError(
                        422,
                        "duplicate_column",
                        "A flow contains column names that become duplicates after trimming spaces.",
                        {"row": index, "column": name},
                    )
                normalized[name] = value
            normalized_rows.append(normalized)

        missing_rows = [
            {"row": index, "columns": [name for name in preprocessor.feature_names if name not in row][:10]}
            for index, row in enumerate(normalized_rows)
            if any(name not in row for name in preprocessor.feature_names)
        ]
        if missing_rows:
            raise APIError(
                422,
                "missing_features",
                "Every flow must include all required model features.",
                {"rows": missing_rows[:10], "invalid_row_count": len(missing_rows)},
            )

        frame = pd.DataFrame(normalized_rows)
        repairs = repair_report(frame, preprocessor.feature_names)
        scores = predict_scores(request.app.state.model, preprocessor.transform(frame))
        threshold = payload.threshold
        if threshold is None:
            threshold = float(request.app.state.config["threshold_selected_on_validation"])

        return {
            "threshold": float(threshold),
            "rows_scored": len(frame),
            "repaired": repairs,
            "predictions": [
                {"row": index, "score": float(score), "flagged": bool(score >= threshold)}
                for index, score in enumerate(scores)
            ],
        }

    return app


app = create_app()
