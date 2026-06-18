from fastapi import APIRouter

router = APIRouter(tags=["health"])

# Reference ranges for all tracked lab values
_REFERENCE_RANGES = {
    "hemoglobin":       {"low": 12.0,  "high": 17.5},
    "wbc":              {"low": 4.0,   "high": 11.0},
    "rbc":              {"low": 3.8,   "high": 5.8},
    "platelets":        {"low": 150.0, "high": 400.0},
    "hematocrit":       {"low": 36.0,  "high": 52.0},
    "mcv":              {"low": 80.0,  "high": 100.0},
    "mch":              {"low": 27.0,  "high": 33.0},
    "mchc":             {"low": 31.0,  "high": 37.0},
    "glucose":          {"low": 70.0,  "high": 100.0},
    "hba1c":            {"low": 4.0,   "high": 5.7},
    "creatinine":       {"low": 0.6,   "high": 1.2},
    "bun":              {"low": 7.0,   "high": 20.0},
    "sodium":           {"low": 136.0, "high": 145.0},
    "potassium":        {"low": 3.5,   "high": 5.0},
    "chloride":         {"low": 98.0,  "high": 107.0},
    "bicarbonate":      {"low": 22.0,  "high": 29.0},
    "calcium":          {"low": 8.5,   "high": 10.5},
    "magnesium":        {"low": 1.7,   "high": 2.4},
    "phosphorus":       {"low": 2.5,   "high": 4.5},
    "albumin":          {"low": 3.5,   "high": 5.0},
    "total_protein":    {"low": 6.0,   "high": 8.3},
    "alt":              {"low": 7.0,   "high": 56.0},
    "ast":              {"low": 10.0,  "high": 40.0},
    "alp":              {"low": 44.0,  "high": 147.0},
    "bilirubin_total":  {"low": 0.1,   "high": 1.2},
    "bilirubin_direct": {"low": 0.0,   "high": 0.3},
    "total_cholesterol":{"low": 0.0,   "high": 200.0},
    "ldl":              {"low": 0.0,   "high": 100.0},
    "hdl":              {"low": 40.0,  "high": 999.0},
    "triglycerides":    {"low": 0.0,   "high": 150.0},
    "tsh":              {"low": 0.4,   "high": 4.0},
    "t3":               {"low": 80.0,  "high": 200.0},
    "t4":               {"low": 5.0,   "high": 12.0},
    "crp":              {"low": 0.0,   "high": 1.0},
    "esr":              {"low": 0.0,   "high": 20.0},
    "ferritin":         {"low": 12.0,  "high": 300.0},
    "iron":             {"low": 60.0,  "high": 170.0},
    "tibc":             {"low": 250.0, "high": 370.0},
    "uric_acid":        {"low": 2.4,   "high": 7.0},
    "psa":              {"low": 0.0,   "high": 4.0},
    "inr":              {"low": 0.8,   "high": 1.2},
    "pt":               {"low": 11.0,  "high": 13.5},
    "aptt":             {"low": 25.0,  "high": 35.0},
}


@router.get("/health")
async def health_check():
    from app.core.config import settings
    return {
        "success": True,
        "message": "Service is running",
        "data": {
            "status": "healthy",
            "models_dir": str(settings.models_dir),
            "models_exist": {
                "risk_model": settings.risk_model_path.exists(),
                "label_encoder": settings.label_encoder_path.exists(),
                "feature_schema": settings.feature_schema_path.exists(),
            },
            "groq_configured": settings.groq_api_key_available,
        },
    }


@router.get("/api/status")
async def api_status():
    """Full system status for the frontend dashboard."""
    from app.core.config import settings
    import sys

    models_exist = {
        "risk_model": settings.risk_model_path.exists(),
        "label_encoder": settings.label_encoder_path.exists(),
        "feature_schema": settings.feature_schema_path.exists(),
    }

    # ML model info
    ml_info = {"status": "not_loaded"}
    if models_exist["risk_model"] and models_exist["label_encoder"]:
        try:
            import pickle
            with open(settings.risk_model_path, "rb") as f:
                model = pickle.load(f)
            ml_info = {
                "status": "loaded",
                "type": type(model).__name__,
                "n_features": getattr(model, "n_features_in_", None),
                "n_classes": getattr(model, "n_classes_", None),
                "max_depth": getattr(model, "max_depth", None),
                "n_estimators": getattr(model, "n_estimators", None),
            }
        except Exception as e:
            ml_info = {"status": "error", "error": str(e)}

    # Label encoder info
    le_info = {"status": "not_loaded"}
    if models_exist["label_encoder"]:
        try:
            import pickle
            with open(settings.label_encoder_path, "rb") as f:
                le = pickle.load(f)
            le_info = {"status": "loaded", "classes": list(le.classes_) if hasattr(le, "classes_") else []}
        except Exception as e:
            le_info = {"status": "error", "error": str(e)}

    # Feature schema
    feature_schema = None
    if models_exist["feature_schema"]:
        try:
            import json
            with open(settings.feature_schema_path) as f:
                feature_schema = json.load(f)
        except Exception:
            pass

    # OCR engine
    ocr_status = "unavailable"
    try:
        from paddleocr import PaddleOCR  # noqa
        ocr_status = "ready"
    except ImportError:
        pass

    # Groq LLM
    groq_key_detected = settings.groq_api_key_available
    groq_client_ok = False
    if groq_key_detected:
        try:
            from hackthathon import groq_client
            groq_client_ok = groq_client is not None
        except Exception:
            pass

    return {
        "success": True,
        "message": "Status OK",
        "data": {
            "status": "healthy",
            "ml_model": ml_info,
            "label_encoder": le_info,
            "feature_schema": feature_schema,
            "ocr_engine": ocr_status,
            "groq_client": groq_client_ok,
            "groq_api_key_detected": groq_key_detected,
            "base_dir": str(settings.backend_dir),
            "paths": {
                "models_dir": str(settings.models_dir),
                "uploads_dir": str(settings.uploads_dir),
                "risk_model": str(settings.risk_model_path),
                "label_encoder": str(settings.label_encoder_path),
                "feature_schema": str(settings.feature_schema_path),
            },
        },
    }


@router.get("/api/reference")
async def get_reference():
    """Reference ranges for all tracked lab values."""
    return {
        "success": True,
        "message": "Reference ranges",
        "data": {"reference": _REFERENCE_RANGES},
    }
