from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.schemas.analysis import AnalysisTextRequest
from app.schemas.response import APIResponse
from app.services.analysis_service import run_full_analysis
from app.services.ocr_service import extract_text as ocr_extract
from app.services.pdf_service import extract_text as pdf_extract
import tempfile, os, json

router = APIRouter(prefix="/api/analysis", tags=["analysis"])
extract_router = APIRouter(prefix="/api/extract", tags=["extract"])

# Units for each lab test
LAB_UNITS = {
    "hemoglobin": "g/dL", "wbc": "/µL", "rbc": "M/µL", "platelets": "/µL",
    "hematocrit": "%", "mcv": "fL", "mch": "pg", "mchc": "g/dL",
    "neutrophils": "%", "lymphocytes": "%", "eosinophils": "%", "monocytes": "%",
    "glucose": "mg/dL", "hba1c": "%", "insulin": "µU/mL", "c_peptide": "ng/mL",
    "creatinine": "mg/dL", "bun": "mg/dL", "uric_acid": "mg/dL",
    "egfr": "mL/min/1.73m²", "cystatin_c": "mg/L",
    "cholesterol": "mg/dL", "ldl": "mg/dL", "hdl": "mg/dL",
    "triglycerides": "mg/dL", "vldl": "mg/dL", "apob": "mg/dL",
    "alt": "U/L", "ast": "U/L", "alp": "U/L", "bilirubin": "mg/dL",
    "direct_bilirubin": "mg/dL", "albumin": "g/dL", "ggt": "U/L", "pt_inr": "",
    "tsh": "mIU/L", "t3": "ng/dL", "t4": "µg/dL",
    "free_t4": "ng/dL", "free_t3": "pg/mL",
    "sodium": "mEq/L", "potassium": "mEq/L", "chloride": "mEq/L",
    "calcium": "mg/dL", "magnesium": "mg/dL", "phosphorus": "mg/dL",
    "bicarbonate": "mEq/L", "iron": "µg/dL", "ferritin": "ng/mL",
    "transferrin": "mg/dL", "tibc": "µg/dL", "transferrin_sat": "%",
    "troponin": "ng/mL", "ck_mb": "U/L", "bnp": "pg/mL",
    "ck": "U/L", "ldh": "U/L", "crp": "mg/L", "esr": "mm/hr",
    "procalcitonin": "ng/mL", "il6": "pg/mL",
    "vitamin_d": "ng/mL", "vitamin_b12": "pg/mL", "folate": "ng/mL",
    "cortisol": "µg/dL", "urine_protein": "mg/24hr", "urine_albumin": "mg/g",
}


def _safe_str(obj):
    """Recursively ensure all strings in obj are properly encoded."""
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(obj, dict):
        return {k: _safe_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_str(i) for i in obj]
    return obj


def _build_response_data(result: dict) -> dict:
    """Return fields with names the frontend expects, with units and clean encoding."""
    labs_raw = {k: v for k, v in result["labs"].items() if v is not None}

    # Build enriched labs with value + unit
    labs_with_units = {
        k: {"value": v, "unit": LAB_UNITS.get(k, "")}
        for k, v in labs_raw.items()
    }

    data = {
        "labs": labs_raw,
        "labs_detail": labs_with_units,
        "interp": {k: v for k, v in result["interp"].items() if v != "not_tested"},
        "score_result": result["score_result"],
        "rule_result": result["rule_result"],
        "diseases": result["diseases"],
        "trend_result": result["trend_result"],
        "llm_output": result["llm_output"],
        "patient_id": result.get("patient_id"),
    }
    return _safe_str(data)


def _json_response(payload: dict, status: int = 200):
    """Return JSONResponse with explicit UTF-8 to fix emoji/Arabic encoding."""
    return JSONResponse(
        content=payload,
        status_code=status,
        media_type="application/json; charset=utf-8",
    )


@router.post("/text")
async def analyze_text(req: AnalysisTextRequest):
    result = run_full_analysis(req.lab_text, req.symptoms, req.patient_id, req.language)
    if result is None:
        return _json_response({"success": False, "message": "Empty input text", "errors": ["No text provided"]})
    return _json_response({"success": True, "message": "Analysis complete", "data": _build_response_data(result)})


@router.post("/pdf")
async def analyze_pdf(
    file: UploadFile = File(...),
    patient_id: str = Form(None),
    symptoms: str = Form(""),
    language: str = Form("en"),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return _json_response({"success": False, "message": "Invalid file", "errors": ["Only PDF files accepted"]})

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        extracted = pdf_extract(tmp.name)
        if not extracted.strip():
            return _json_response({"success": False, "message": "No text extracted from PDF", "errors": ["PDF extraction returned empty"]})

        result = run_full_analysis(extracted, symptoms, patient_id, language)
        if result is None:
            return _json_response({"success": False, "message": "Analysis failed", "errors": ["Empty text after extraction"]})

        data = _build_response_data(result)
        data["extracted_text"] = extracted[:500]
        data["extracted_length"] = len(extracted)
        return _json_response({"success": True, "message": "PDF analysis complete", "data": data})
    except Exception as e:
        return _json_response({"success": False, "message": "PDF analysis failed", "errors": [str(e)]})
    finally:
        os.unlink(tmp.name)


@router.post("/image")
async def analyze_image(
    file: UploadFile = File(...),
    patient_id: str = Form(None),
    symptoms: str = Form(""),
    language: str = Form("en"),
):
    allowed = {"png", "jpg", "jpeg", "tiff", "bmp"}
    ext = file.filename.split(".")[-1].lower() if file.filename else ""
    if ext not in allowed:
        return _json_response({"success": False, "message": "Invalid file type", "errors": [f"Allowed: {', '.join(allowed)}"]})

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        extracted = ocr_extract(tmp.name)
        if not extracted.strip():
            return _json_response({"success": False, "message": "No text extracted from image", "errors": ["OCR returned empty"]})

        result = run_full_analysis(extracted, symptoms, patient_id, language)
        if result is None:
            return _json_response({"success": False, "message": "Analysis failed", "errors": ["Empty text after OCR"]})

        data = _build_response_data(result)
        data["extracted_text"] = extracted[:500]
        data["extracted_length"] = len(extracted)
        return _json_response({"success": True, "message": "Image analysis complete", "data": data})
    except Exception as e:
        return _json_response({"success": False, "message": "Image analysis failed", "errors": [str(e)]})
    finally:
        os.unlink(tmp.name)


# ── Extract-only endpoints (two-step flow: extract then analyze separately) ──

@extract_router.post("/pdf")
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return _json_response({"success": False, "message": "Invalid file", "errors": ["Only PDF files accepted"]})

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        text = pdf_extract(tmp.name)
        if not text.strip():
            return _json_response({"success": False, "message": "No text extracted", "errors": ["PDF extraction returned empty"]})

        return _json_response({"success": True, "message": "Text extracted", "data": {"text": text, "length": len(text)}})
    except Exception as e:
        return _json_response({"success": False, "message": "PDF extraction failed", "errors": [str(e)]})
    finally:
        os.unlink(tmp.name)


@extract_router.post("/image")
async def extract_image(file: UploadFile = File(...)):
    allowed = {"png", "jpg", "jpeg", "tiff", "bmp"}
    ext = file.filename.split(".")[-1].lower() if file.filename else ""
    if ext not in allowed:
        return _json_response({"success": False, "message": "Invalid file type", "errors": [f"Allowed: {', '.join(allowed)}"]})

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        text = ocr_extract(tmp.name)
        if not text.strip():
            return _json_response({"success": False, "message": "No text extracted", "errors": ["OCR returned empty"]})

        return _json_response({"success": True, "message": "Text extracted", "data": {"text": text, "length": len(text)}})
    except Exception as e:
        return _json_response({"success": False, "message": "Image extraction failed", "errors": [str(e)]})
    finally:
        os.unlink(tmp.name)
