"""
╔══════════════════════════════════════════════════════════════════════╗
║   AI LAB INTERPRETATION & EARLY RISK AWARENESS SYSTEM  v3.0         ║
║   Hackathon Edition — Full Upgrade                                   ║
║   ⚠️  Decision-support ONLY — NOT a replacement for diagnosis         ║
╚══════════════════════════════════════════════════════════════════════╝

UPGRADE LOG (v2.1 → v3.0):
  ✅ OCR: 3-pass strategy + confidence boosting + number normalizer
  ✅ ML:  Smarter dummy-model detection, calibrated fallback
  ✅ Rules: 15+ new clinical conditions added
  ✅ Features: 10 new ratio/composite features
  ✅ Trends: Per-visit delta tracking + terminal sparkline
  ✅ Report: Color-coded ANSI terminal output, severity badges
  ✅ Patients: Registry with visit history (CSV-based)
  ✅ Safety: All None/division-by-zero/missing-file edge cases hardened
  ✅ LLM: Tighter, safer, more structured prompt
  ✅ Output: Urdu/Arabic RTL-aware formatting
"""

# ─────────────────────────────────────────────
# SECTION 0: IMPORTS (CLEAN + SAFE)
# ─────────────────────────────────────────────

import os
import re
import sys
import json
import joblib
import warnings
import tempfile
import traceback
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────
# OPTIONAL / EXTERNAL DEPENDENCIES SAFETY LAYER
# ─────────────────────────────────────────────

# OpenCV (OCR/Image)
try:
    import cv2
except ImportError:
    cv2 = None
    print("⚠️ OpenCV not installed (cv2 disabled)")

# OCR
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None
    print("⚠️ PaddleOCR not installed")

# ML
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# LLM
try:
    from groq import Groq
except ImportError:
    Groq = None
    print("⚠️ Groq SDK not installed")

# Language tools
try:
    from langdetect import detect
except ImportError:
    detect = None
    print("⚠️ langdetect not installed")

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None
    print("⚠️ deep_translator not installed")

# PDF support
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    fitz = None
    PDF_SUPPORT = False
    print("⚠️ PyMuPDF not installed. PDF support disabled (pip install pymupdf)")

# ─────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SECTION 1: ANSI COLOR HELPERS
# ─────────────────────────────────────────────

class C:
    """ANSI color codes for terminal output."""

    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"

    BG_RED  = "\033[41m"
    BG_YEL  = "\033[43m"
    BG_GRN  = "\033[42m"
    BG_BLU  = "\033[44m"


def colorize(text: str, *codes: str) -> str:
    """
    Safely wraps text with ANSI color codes.
    """
    try:
        return "".join(codes) + str(text) + C.RESET
    except Exception:
        return str(text)


def risk_color(level: str) -> str:
    """
    Maps risk levels to colors safely.
    """
    if not isinstance(level, str):
        return C.RESET

    return {
        "LOW": C.GREEN,
        "MEDIUM": C.YELLOW,
        "HIGH": C.RED,
        "CRITICAL": C.BG_RED + C.WHITE + C.BOLD,
    }.get(level.upper(), C.RESET)


def status_badge(status: str) -> str:
    """
    Returns formatted status badge with color safety.
    """
    if not isinstance(status, str):
        status = "unknown"

    status = status.lower()

    badges = {
        "normal": colorize("  NORMAL  ", C.BG_GRN, C.WHITE, C.BOLD),
        "high": colorize("  HIGH ↑  ", C.BG_RED, C.WHITE, C.BOLD),
        "low": colorize("  LOW ↓  ", C.BG_YEL, C.WHITE, C.BOLD),
        "not_tested": colorize(" NOT TESTED ", C.DIM),
        "unknown_ref": colorize(" UNKNOWN ", C.DIM),
    }

    return badges.get(
        status,
        colorize(f"  {status.upper()}  ", C.DIM)
    )

# ─────────────────────────────────────────────
# SECTION 2: CONFIGURATION & PATHS
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# BASE DIRECTORY (MAKE IT PORTABLE)
# ─────────────────────────────────────────────

BASE_DIR = Path(os.getenv("HEALTH_PROJECT_DIR", r"D:\Health Project"))

# Ensure base directory exists
BASE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# ALL PROJECT PATHS
# ─────────────────────────────────────────────

PATHS = {
    "risk_model":       BASE_DIR / "risk_model.pkl",
    "feature_schema":   BASE_DIR / "feature_schema.json",
    "trend_encoder":    BASE_DIR / "trend_encoder.pkl",
    "label_encoder":    BASE_DIR / "label_encoder.pkl",

    "features_final":   BASE_DIR / "features_dataset_final.csv",
    "features_v2":      BASE_DIR / "features_dataset_v2.csv",
    "lab_trends":       BASE_DIR / "lab_trends_clean.csv",
    "risk_dataset":     BASE_DIR / "risk_dataset.csv",
    "model_input":      BASE_DIR / "model_input_clean.csv",

    "capture_temp":     BASE_DIR / "capture_temp.jpg",

    "patient_registry": BASE_DIR / "patient_registry.csv",
    "visit_history":    BASE_DIR / "visit_history.csv",
}

# ─────────────────────────────────────────────
# .ENV LOADER (secure key management)
# ─────────────────────────────────────────────
from dotenv import load_dotenv
_env_file = Path(".env")
if _env_file.exists():
    load_dotenv(_env_file)
elif (env_example := Path(".env.example")).exists():
    load_dotenv(env_example)

# ─────────────────────────────────────────────
# SECTION 3: CLINICAL REFERENCE RANGES (ENHANCED)
# ─────────────────────────────────────────────

REFERENCE = {
    # Hematology
    "hemoglobin": (12.0, 17.0),
    "wbc": (4500, 11000),
    "platelets": (150000, 450000),
    "rbc": (4.2, 5.8),
    "hematocrit": (36.0, 50.0),
    "mcv": (80.0, 100.0),
    "mch": (27.0, 33.0),
    "mchc": (32.0, 36.0),
    "neutrophils": (40.0, 70.0),
    "lymphocytes": (20.0, 40.0),
    "eosinophils": (1.0, 4.0),
    "monocytes": (2.0, 8.0),

    # Metabolic
    "glucose": (70.0, 140.0),
    "hba1c": (0.0, 5.7),
    "insulin": (2.0, 25.0),
    "c_peptide": (0.5, 2.0),

    # Kidney
    "creatinine": (0.7, 1.3),
    "bun": (7.0, 20.0),
    "uric_acid": (3.5, 7.2),
    "egfr": (60.0, 120.0),
    "cystatin_c": (0.5, 1.0),

    # Lipids
    "cholesterol": (0.0, 200.0),
    "ldl": (0.0, 100.0),
    "hdl": (40.0, 100.0),
    "triglycerides": (0.0, 150.0),
    "vldl": (0.0, 30.0),
    "apob": (0.0, 100.0),

    # Liver
    "alt": (7.0, 56.0),
    "ast": (10.0, 40.0),
    "alp": (44.0, 147.0),
    "bilirubin": (0.1, 1.2),
    "albumin": (3.5, 5.0),
    "ggt": (9.0, 48.0),
    "pt_inr": (0.8, 1.2),
    "direct_bilirubin": (0.0, 0.3),

    # Thyroid
    "tsh": (0.4, 4.0),
    "t3": (80.0, 200.0),
    "t4": (5.1, 14.1),
    "free_t4": (0.8, 1.8),
    "free_t3": (2.3, 4.2),

    # Electrolytes
    "sodium": (136.0, 145.0),
    "potassium": (3.5, 5.0),
    "chloride": (98.0, 106.0),
    "calcium": (8.5, 10.2),
    "magnesium": (1.7, 2.2),
    "phosphorus": (2.5, 4.5),
    "bicarbonate": (22.0, 29.0),

    # Iron
    "iron": (60.0, 170.0),
    "ferritin": (12.0, 300.0),
    "transferrin": (200.0, 360.0),
    "tibc": (250.0, 370.0),
    "transferrin_sat": (20.0, 50.0),

    # Cardiac
    "troponin": (0.0, 0.04),
    "ck_mb": (0.0, 25.0),
    "bnp": (0.0, 100.0),
    "ck": (30.0, 200.0),
    "ldh": (140.0, 280.0),

    # Inflammation
    "crp": (0.0, 10.0),
    "esr": (0.0, 20.0),
    "procalcitonin": (0.0, 0.5),
    "il6": (0.0, 7.0),

    # Vitamins
    "vitamin_d": (30.0, 100.0),
    "vitamin_b12": (200.0, 900.0),
    "folate": (3.0, 17.0),
    "cortisol": (6.0, 23.0),

    # Urine
    "urine_protein": (0.0, 150.0),
    "urine_albumin": (0.0, 30.0),
    "hba1c_mmol": (0.0, 39.0),
}

# ─────────────────────────────────────────────
# SEVERITY SYSTEM
# ─────────────────────────────────────────────

SEVERITY_WEIGHTS = {
    "mild": 10,
    "moderate": 25,
    "severe": 50,
    "critical": 80,
}

# ─────────────────────────────────────────────
# SAFE VALIDATION HELPERS (NEW)
# ─────────────────────────────────────────────

def is_valid_test(test_name: str) -> bool:
    """Check if test exists in reference ranges."""
    return isinstance(test_name, str) and test_name in REFERENCE


def get_reference_range(test_name: str):
    """Safely return reference range."""
    return REFERENCE.get(test_name, None)


def calculate_severity_score(value: float, test_name: str) -> float:
    """
    Returns deviation severity (0–100 scale approx).
    """
    try:
        if test_name not in REFERENCE:
            return 0.0

        low, high = REFERENCE[test_name]

        if value < low:
            return min(100.0, ((low - value) / low) * 100)
        elif value > high:
            return min(100.0, ((value - high) / high) * 100)
        else:
            return 0.0

    except Exception:
        return 0.0

# ─────────────────────────────────────────────
# SECTION 4: INIT SERVICES
# ─────────────────────────────────────────────

print(colorize("Initializing services...Stand by", C.CYAN))

# --------------------------
# GROQ LLM
print(colorize(f"   Groq SDK available: {'YES' if Groq else 'NO'}", C.CYAN if Groq else C.YELLOW))
_groq_api_key = os.getenv("GROQ_API_KEY")
if _groq_api_key:
    print(colorize("   GROQ_API_KEY detected: YES", C.CYAN))
    try:
        groq_client = Groq(api_key=_groq_api_key)
        print(colorize("   Groq LLM connected", C.GREEN))
    except Exception as e:
        groq_client = None
        print(colorize(f"   Groq LLM init failed: {e}", C.YELLOW))
else:
    groq_client = None
    print(colorize("   GROQ_API_KEY detected: NO", C.YELLOW))
    print(colorize("   Groq LLM disabled — set GROQ_API_KEY in .env or environment", C.YELLOW))


# --------------------------
# OCR ENGINE
# --------------------------
try:
    if PaddleOCR is not None:
        # New PaddleOCR 3.x API (show_log removed)
        try:
            ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")
            print(colorize("PaddleOCR ready", C.GREEN))
        except Exception as _e1:
            ocr_engine = None
            print(colorize(f"PaddleOCR init failed: {_e1}", C.YELLOW))
    else:
        ocr_engine = None
except Exception as e:
    ocr_engine = None
    print(colorize(f"PaddleOCR failed: {e}", C.YELLOW))


# --------------------------
# TRANSLATION (SAFE CHECK)
# --------------------------
try:
    _ = GoogleTranslator(source="auto", target="en")
    print(colorize(" Translator ready", C.GREEN))
except Exception as e:
    print(colorize(f" Translator not fully initialized: {e}", C.YELLOW))

# ─────────────────────────────────────────────
# SECTION 5: PATIENT REGISTRY (NEW v3.0)
# ─────────────────────────────────────────────


def load_patient_registry() -> pd.DataFrame:
    path = PATHS["patient_registry"]

    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass

    return pd.DataFrame(columns=[
        "patient_id", "name", "age", "gender",
        "first_visit", "last_visit", "visit_count"
    ])


def save_patient_registry(df: pd.DataFrame) -> None:
    try:
        df.to_csv(PATHS["patient_registry"], index=False)
    except Exception as e:
        print(colorize(f" Could not save registry: {e}", C.YELLOW))


def register_patient(patient_id: str, name: str = "", age: str = "", gender: str = "") -> None:
    """Register or update a patient in the registry."""

    df = load_patient_registry()
    now = datetime.now().strftime("%Y-%m-%d")

    # Ensure safe columns exist
    for col in ["visit_count"]:
        if col not in df.columns:
            df[col] = 0

    mask = df["patient_id"] == patient_id

    if mask.any():
        df.loc[mask, "last_visit"] = now

        # FIX: safe numeric increment
        df.loc[mask, "visit_count"] = (
            df.loc[mask, "visit_count"].fillna(0).astype(int) + 1
        )

        if name:
            df.loc[mask, "name"] = name
        if age:
            df.loc[mask, "age"] = age
        if gender:
            df.loc[mask, "gender"] = gender

    else:
        new_row = pd.DataFrame([{
            "patient_id": patient_id,
            "name": name or "Unknown",
            "age": age or "Unknown",
            "gender": gender or "Unknown",
            "first_visit": now,
            "last_visit": now,
            "visit_count": 1,
        }])

        df = pd.concat([df, new_row], ignore_index=True)

    save_patient_registry(df)


def save_visit(patient_id: str, labs: dict, risk_score: float, risk_level: str) -> None:
    """Append a visit record for trend tracking."""

    path = PATHS["visit_history"]

    row = {
        "patient_id": patient_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "risk_score": risk_score,
        "risk_level": risk_level,
    }

    # clean labs safely
    row.update({k: v for k, v in labs.items() if v is not None})

    try:
        if path.exists():
            existing = pd.read_csv(path)
            combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        else:
            combined = pd.DataFrame([row])

        combined.to_csv(path, index=False)
        print(colorize(f" Visit saved for patient: {patient_id}", C.GREEN))

    except Exception as e:
        print(colorize(f" Could not save visit: {e}", C.YELLOW))


def get_visit_history(patient_id: str) -> pd.DataFrame:
    """Load all visits for a patient."""

    path = PATHS["visit_history"]

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)

        if "patient_id" not in df.columns:
            return pd.DataFrame()

        return df[df["patient_id"] == patient_id].reset_index(drop=True)

    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# SECTION 6: INPUT ENGINE
# ─────────────────────────────────────────────

def sanitize_path(path: str) -> str:
    return path.strip().strip('"').strip("'").strip()


def normalize_ocr_numbers(text: str) -> str:
    """
    NEW v3.0: OCR number normalization pipeline.
    Fixes common OCR errors in medical lab values.
    """

    text = re.sub(r'(?<=[0-9])O(?=[0-9])', '0', text)
    text = re.sub(r'(?<=[0-9])[lI](?=[0-9])', '1', text)
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    text = re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', text)
    text = re.sub(r'[°º]', '', text)

    return text


def preprocess_image_for_ocr(img_path: str) -> str:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if w < 1200:
        scale = 1200 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    denoised = cv2.fastNlMeansDenoising(
        gray, h=10, templateWindowSize=7, searchWindowSize=21
    )

    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, thresh)
    return tmp.name


def preprocess_image_sharpened(img_path: str) -> str:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    sharp = cv2.filter2D(gray, -1, kernel)

    h, w = sharp.shape
    if w < 1200:
        scale = 1200 / w
        sharp = cv2.resize(sharp, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, sharp)
    return tmp.name


def run_paddleocr(img_path: str, min_conf: float = 0.5) -> str:
    if ocr_engine is None:
        return ""

    try:
        result = ocr_engine.ocr(img_path, cls=True)

        if not result or not result[0]:
            return ""

        lines = []

        for line in result[0]:
            if not line:
                continue

            bbox, (text, conf) = line[0], line[1]

            if conf < min_conf:
                continue

            y_center = (bbox[0][1] + bbox[2][1]) / 2
            lines.append((y_center, text))

        lines.sort(key=lambda x: x[0])

        extracted = " ".join(t for _, t in lines)
        return normalize_ocr_numbers(extracted)

    except Exception as e:
        print(colorize(f" OCR engine error: {e}", C.YELLOW))
        return ""


def extract_text_from_image(img_path: str) -> str:
    img_path = sanitize_path(img_path)

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    if ocr_engine is None:
        print(colorize(" OCR engine not available.", C.RED))
        return ""

    print(colorize(f" OCR processing (3-pass): {img_path}", C.CYAN))

    results = {}

    for pass_name, fn in [
        ("adaptive_thresh", preprocess_image_for_ocr),
        ("original", None),
        ("sharpened", preprocess_image_sharpened),
    ]:
        tmp_path = None

        try:
            if fn:
                tmp_path = fn(img_path)
                text = run_paddleocr(tmp_path)
            else:
                text = run_paddleocr(img_path)

            results[pass_name] = text
            print(colorize(f"    Pass [{pass_name}]: {len(text)} chars", C.DIM))

        except Exception as e:
            results[pass_name] = ""
            print(colorize(f"    Pass [{pass_name}] failed: {e}", C.YELLOW))

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    best_pass = max(results, key=lambda k: len(results[k]))
    best_text = results[best_pass]

    print(colorize(f" Best pass: [{best_pass}] → {len(best_text)} chars", C.GREEN))

    if not best_text.strip():
        print(colorize("  All OCR passes returned empty text.", C.YELLOW))

    return best_text


def extract_text_from_pdf(pdf_path: str) -> str:
    pdf_path = sanitize_path(pdf_path)

    if not PDF_SUPPORT:
        print(colorize(" PDF support not available.", C.RED))
        return ""

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(colorize(f" Reading PDF: {pdf_path}", C.CYAN))

    try:
        doc = fitz.open(pdf_path)
        full_text = []

        for i in range(doc.page_count):
            page = doc[i]
            text = page.get_text("text")

            if len(text.strip()) < 50:
                pix = page.get_pixmap(dpi=300)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                pix.save(tmp.name)

                ocr_text = extract_text_from_image(tmp.name)

                try:
                    os.unlink(tmp.name)
                except:
                    pass

                full_text.append(ocr_text)
            else:
                full_text.append(normalize_ocr_numbers(text))

        doc.close()
        return " ".join(full_text)

    except Exception as e:
        print(colorize(f" PDF Error: {e}", C.RED))
        return ""


def capture_from_camera() -> str | None:
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print(colorize("Cannot open camera.", C.RED))
        return None

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nCamera opened. SPACE = Capture | ESC = Exit")

    captured_path = None

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        mx, my = int(w * 0.1), int(h * 0.1)

        cv2.rectangle(frame, (mx, my), (w - mx, h - my), (0, 255, 0), 2)

        cv2.putText(frame, "Align lab report inside box",
                    (mx, my - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, "SPACE: Capture | ESC: Exit",
                    (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("AI Lab Scanner", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if key == 32:
            save_path = str(PATHS["capture_temp"])
            cv2.imwrite(save_path, frame)
            captured_path = save_path
            print(colorize(f"  Captured → {save_path}", C.GREEN))
            break

    cam.release()
    cv2.destroyAllWindows()

    return captured_path

# ─────────────────────────────────────────────
# SECTION 7: UNIFIED PARSER ENGINE
# ─────────────────────────────────────────────
 
LAB_ALIASES = {
    "hemoglobin":       ["hemoglobin", "haemoglobin", "hb", "hgb"],
    "wbc":              ["wbc", "white blood cell", "white blood cells", "white blood count",
                         "leukocytes", "leucocytes", "total wbc", "tlc"],
    "platelets":        ["platelets", "platelet count", "plt", "thrombocytes"],
    "rbc":              ["rbc", "red blood cell", "red blood cells", "erythrocytes"],
    "hematocrit":       ["hematocrit", "haematocrit", "hct", "pcv"],
    "mcv":              ["mcv", "mean corpuscular volume"],
    "mch":              ["mch", "mean corpuscular hemoglobin"],
    "mchc":             ["mchc"],
    "neutrophils":      ["neutrophils", "neutrophil", "pmn", "granulocytes", "seg"],   # NEW
    "lymphocytes":      ["lymphocytes", "lymphocyte", "lymphs"],                        # NEW
    "eosinophils":      ["eosinophils", "eosinophil", "eos"],                           # NEW
    "monocytes":        ["monocytes", "monocyte", "mono"],                              # NEW
    "glucose":          ["fasting blood sugar", "fbs", "blood glucose", "glucose",
                         "blood sugar", "rbs", "random blood sugar", "blood sugar level", "fbg"],
    "hba1c":            ["hba1c", "hba 1c", "glycated hemoglobin", "hemoglobin a1c",
                         "a1c", "glycosylated hemoglobin", "hba1c%"],
    "insulin":          ["insulin", "serum insulin"],
    "c_peptide":        ["c peptide", "c-peptide"],                                     # NEW
    "creatinine":       ["creatinine", "serum creatinine", "s. creatinine", "s creatinine"],
    "bun":              ["bun", "blood urea nitrogen", "urea", "blood urea"],
    "uric_acid":        ["uric acid", "serum uric acid", "s. uric acid"],
    "egfr":             ["egfr", "gfr", "estimated gfr", "estimated glomerular"],
    "cystatin_c":       ["cystatin c", "cystatin-c"],                                   # NEW
    "cholesterol":      ["total cholesterol", "cholesterol", "chol"],
    "ldl":              ["ldl", "ldl cholesterol", "ldl-c", "low density lipoprotein"],
    "hdl":              ["hdl", "hdl cholesterol", "hdl-c", "high density lipoprotein"],
    "triglycerides":    ["triglycerides", "triglyceride", "tg", "trigs"],
    "vldl":             ["vldl", "vldl cholesterol"],
    "apob":             ["apob", "apolipoprotein b", "apo b"],                          # NEW
    "alt":              ["alt", "sgpt", "alanine aminotransferase", "alanine transaminase"],
    "ast":              ["ast", "sgot", "aspartate aminotransferase", "aspartate transaminase"],
    "alp":              ["alp", "alkaline phosphatase", "alk phos"],
    "bilirubin":        ["total bilirubin", "bilirubin", "t. bilirubin", "tbil"],
    "direct_bilirubin": ["direct bilirubin", "conjugated bilirubin", "dbil"],           # NEW
    "albumin":          ["albumin", "serum albumin"],
    "ggt":              ["ggt", "gamma gt", "gamma-glutamyltransferase"],
    "pt_inr":           ["pt inr", "inr", "prothrombin time", "pt/inr"],               # NEW
    "tsh":              ["tsh", "thyroid stimulating hormone"],
    "t3":               ["t3", "triiodothyronine", "total t3"],
    "t4":               ["t4", "thyroxine", "total t4"],
    "free_t4":          ["free t4", "ft4", "free thyroxine"],                          # NEW
    "free_t3":          ["free t3", "ft3", "free triiodothyronine"],                   # NEW
    "sodium":           ["sodium", "na", "serum sodium", "s. sodium"],
    "potassium":        ["potassium", "k", "serum potassium", "s. potassium"],
    "chloride":         ["chloride", "cl", "serum chloride"],
    "calcium":          ["calcium", "ca", "serum calcium", "total calcium"],
    "magnesium":        ["magnesium", "mg", "serum magnesium"],
    "phosphorus":       ["phosphorus", "phosphate", "inorganic phosphate"],             # NEW
    "bicarbonate":      ["bicarbonate", "hco3", "co2 content", "total co2"],           # NEW
    "iron":             ["iron", "serum iron", "fe"],
    "ferritin":         ["ferritin", "serum ferritin"],
    "transferrin":      ["transferrin"],
    "tibc":             ["tibc", "total iron binding capacity"],                        # NEW
    "transferrin_sat":  ["transferrin saturation", "tsat", "iron saturation"],         # NEW
    "troponin":         ["troponin", "troponin i", "troponin t", "trop i", "trop t", "hs troponin"],
    "ck_mb":            ["ck-mb", "ck mb", "creatine kinase mb"],
    "bnp":              ["bnp", "brain natriuretic peptide", "nt-probnp", "pro bnp"],
    "ck":               ["total ck", "ck total", "creatine kinase", "cpk"],           # NEW
    "ldh":              ["ldh", "lactate dehydrogenase"],                              # NEW
    "crp":              ["crp", "c reactive protein", "c-reactive protein", "hs-crp"],
    "esr":              ["esr", "erythrocyte sedimentation rate", "sed rate"],
    "procalcitonin":    ["procalcitonin", "pct"],                                      # NEW
    "il6":              ["il6", "il-6", "interleukin 6", "interleukin-6"],             # NEW
    "vitamin_d":        ["vitamin d", "vit d", "25-oh vitamin d", "25 oh vit d",
                         "cholecalciferol", "25-hydroxyvitamin d"],                    # NEW
    "vitamin_b12":      ["vitamin b12", "vit b12", "cobalamin", "cyanocobalamin",
                         "b12"],                                                        # NEW
    "folate":           ["folate", "folic acid", "vitamin b9", "rbc folate"],          # NEW
    "cortisol":         ["cortisol", "serum cortisol", "morning cortisol"],            # NEW
    "urine_protein":    ["urine protein", "urinary protein", "24h protein"],           # NEW
    "urine_albumin":    ["microalbuminuria", "urine albumin", "urinary albumin",
                         "acr", "albumin creatinine ratio"],                           # NEW
}
 
SCALE_FIX = {
    "wbc":       (1, 1000, 1000),
    "platelets": (1, 1000, 1000),
}
 
 
def parse_labs(text: str) -> dict:
    """
    UPGRADED v3.0: Robust lab parser with extended alias coverage
    and improved concatenated-text splitting.
    """
    # Pre-split concatenated field names
    splitters = [
        "HEMATOLOGY", "DIABETES", "KIDNEY", "LIPID", "LIVER", "THYROID",
        "CARDIAC", "ELECTROLYTE", "SYMPTOMS", "PANEL", "FUNCTION", "PROFILE",
        "VITAMIN", "HORMONE", "INFLAMMATORY", "COAGULATION", "URINE",
        "Hemoglobin", "WBC", "Platelet", "Fasting", "HbA1c", "Serum",
        "Blood Urea", "Total Cholesterol", "LDL", "HDL", "Triglycerides",
        "ALT", "AST", "Bilirubin", "Creatinine", "Glucose", "TSH",
        "Sodium", "Potassium", "Calcium", "Iron", "Ferritin", "CRP", "ESR",
        "Troponin", "BNP", "Vitamin", "Cortisol", "Folate",
    ]
    for s in splitters:
        text = re.sub(rf'(?<=[a-z0-9])({re.escape(s)})', r' \1', text)
 
    # Fix unit+next-lab fusion
    for unit in ["U/L", "mg/dL", "g/dL", "%", "/µL", "/ul", "ng/ml",
                 "nmol/L", "mmol/L", "mIU/L", "pg/mL", "IU/L"]:
        text = re.sub(rf'({re.escape(unit)})([A-Z])', r'\1 \2', text,
                      flags=re.IGNORECASE)
 
    text = text.lower()
    text = normalize_ocr_numbers(text)
    text = re.sub(r'\s+', ' ', text).replace('\n', ' ').replace('\r', ' ')
 
    results = {k: None for k in REFERENCE.keys()}
    NUM  = r'([0-9]+\.?[0-9]*)'
    UNIT = r'(?:\s*(?:g/dl|mg/dl|mg/l|mmol/l|u/l|iu/l|%|/µl|/ul|µu/ml|ng/ml|pg/ml|' \
           r'meq/l|mm/hr|nmol/l|miu/l|iu/ml|cells/µl|x10\^?3|x10\^?6|\w+/\w+|\w+)?)?'
 
    def extract_value(aliases):
        for alias in sorted(aliases, key=len, reverse=True):
            escaped = re.escape(alias)
 
            # Strategy 1: alias [optional words] [sep] number [unit]
            pattern = (
                r'(?<![a-z0-9])' + escaped +
                r'(?:\s+(?:count|level|result|value|test|cholesterol|concentration))?' +
                r'\s*[:\-=\(]?\s*' + NUM + UNIT
            )
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try: return float(m.group(1))
                except: pass
 
            # Strategy 2: alias (sub-alias): number
            pattern2 = (
                r'(?<![a-z0-9])' + escaped +
                r'\s*\([^)]{1,30}\)\s*[:\-=]?\s*' + NUM
            )
            m2 = re.search(pattern2, text, re.IGNORECASE)
            if m2:
                try: return float(m2.group(1))
                except: pass
 
            # Strategy 3 (NEW v3.0): alias … number on same logical line
            pattern3 = (
                r'(?<![a-z0-9])' + escaped +
                r'[^0-9\n]{0,40}' + NUM
            )
            m3 = re.search(pattern3, text, re.IGNORECASE)
            if m3:
                try: return float(m3.group(1))
                except: pass
 
        return None
 
    for key, aliases in LAB_ALIASES.items():
        val = extract_value(aliases)
 
        if val is not None and key in SCALE_FIX:
            min_v, threshold, multiplier = SCALE_FIX[key]
            if min_v <= val < threshold:
                val = val * multiplier
 
        if key in results:
            results[key] = val
 
    parsed = {k: v for k, v in results.items() if v is not None}
    print(colorize(f"  Parsed {len(parsed)} lab values: {list(parsed.keys())}", C.CYAN))
    return results
# ─────────────────────────────────────────────
# EXTRACT PATIENT INFO FROM LAB REPORT TEXT
# ─────────────────────────────────────────────

def extract_patient_name_from_text(text: str) -> str:
    """Extract patient name from lab report text."""
    text = text.lower()
    
    # Common patterns for patient name in lab reports
    patterns = [
        r'patient(?:\s*name)?[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'patient[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'report for[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'mr\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'ms\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'mrs\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            return name.title()
    
    return None


def extract_date_from_text(text: str) -> str:
    """Extract collection/report date from lab report text."""
    text = text.lower()
    
    # Common date patterns
    patterns = [
        r'collected[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'report[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            date_str = match.group(1)
            return date_str
    
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# ─────────────────────────────────────────────
# SECTION 8: INTERPRETATION ENGINE
# ─────────────────────────────────────────────

def interpret_labs(labs: dict) -> dict:
    """Interpret each lab value against reference ranges."""
    results = {}

    for key, val in labs.items():

        if val is None:
            results[key] = "not_tested"
            continue

        if key not in REFERENCE:
            results[key] = "unknown_ref"
            continue

        low, high = REFERENCE[key]

        # Symmetric logic kept but cleaned
        if val < low:
            results[key] = "low"
        elif val > high:
            results[key] = "high"
        else:
            results[key] = "normal"

    return results

# ─────────────────────────────────────────────
# SECTION 9: FEATURE ENGINEERING MODULE
# ─────────────────────────────────────────────

def safe_get(labs: dict, key: str, default: float = 0.0) -> float:
    v = labs.get(key)
    return float(v) if v is not None else default


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return round(num / den, 4) if den != 0 else default


def compute_zscore(val: float, low: float, high: float) -> float:
    mid = (low + high) / 2
    spread = (high - low) / 2
    return round((val - mid) / spread, 4) if spread != 0 else 0.0


def build_feature_vector(labs: dict) -> pd.DataFrame:
    """
    UPGRADED v3.0: Robust feature vector builder.
    """

    features = {}

    # ── Z-SCORES ──
    for key, val in labs.items():
        if val is not None and key in REFERENCE:
            low, high = REFERENCE[key]
            features[f"{key}_zscore"] = compute_zscore(val, low, high)
            features[key] = val
        else:
            features[f"{key}_zscore"] = 0.0
            features[key] = 0.0

    # ── SAFE EXTRACTION ──
    ldl   = safe_get(labs, "ldl")
    hdl   = safe_get(labs, "hdl", 40)
    bun   = safe_get(labs, "bun")
    creat = safe_get(labs, "creatinine", 1)
    alt   = safe_get(labs, "alt")
    ast   = safe_get(labs, "ast")
    chol  = safe_get(labs, "cholesterol")
    trig  = safe_get(labs, "triglycerides")
    glc   = safe_get(labs, "glucose")
    hba1c = safe_get(labs, "hba1c")
    hb    = safe_get(labs, "hemoglobin", 14)
    wbc   = safe_get(labs, "wbc", 7000)
    plt   = safe_get(labs, "platelets", 300000)
    egfr  = safe_get(labs, "egfr", 90)
    iron  = safe_get(labs, "iron", 100)
    fer   = safe_get(labs, "ferritin", 100)
    tibc  = safe_get(labs, "tibc", 300)
    ua    = safe_get(labs, "uric_acid")

    # ── CORE RATIOS ──
    features["ldl_hdl_ratio"]        = safe_div(ldl, hdl)
    features["chol_hdl_ratio"]       = safe_div(chol, hdl)
    features["bun_creatinine_ratio"] = safe_div(bun, creat)
    features["glucose_hba1c_ratio"]  = safe_div(glc, hba1c) if hba1c > 0 else 0.0
    features["ast_alt_ratio"]        = safe_div(ast, alt) if alt > 0 else 0.0

    features["trig_hdl_ratio"]       = safe_div(trig, hdl)
    features["iron_tibc_ratio"]      = safe_div(iron, tibc) if tibc > 0 else 0.0
    features["ast_platelet_ratio"]   = safe_div(ast, plt / 1000) if plt > 0 else 0.0
    features["egfr_creatinine_ratio"]= safe_div(egfr, creat)
    features["uric_acid_creat_ratio"]= safe_div(ua, creat) if creat > 0 else 0.0

    # ── ORGAN FLAGS ──
    features["renal_flag"] = int(
        safe_get(labs, "creatinine") > 1.3 or
        safe_get(labs, "bun") > 20 or
        safe_get(labs, "egfr") < 60
    )

    features["hepatic_flag"] = int(
        safe_get(labs, "alt") > 56 or
        safe_get(labs, "ast") > 40 or
        safe_get(labs, "bilirubin") > 1.2 or
        safe_get(labs, "pt_inr") > 1.2
    )

    features["cardiac_flag"] = int(
        safe_get(labs, "ldl") > 100 or
        safe_get(labs, "cholesterol") > 200 or
        safe_get(labs, "troponin") > 0.04 or
        safe_get(labs, "hdl") < 40
    )

    features["metabolic_flag"] = int(
        safe_get(labs, "glucose") > 140 or
        safe_get(labs, "hba1c") > 6.5
    )

    features["anemia_flag"] = int(
        safe_get(labs, "hemoglobin") < 12 or
        safe_get(labs, "iron") < 60 or
        safe_get(labs, "ferritin") < 12
    )

    features["thyroid_flag"] = int(
        safe_get(labs, "tsh") > 4.0 or safe_get(labs, "tsh") < 0.4
    )

    features["inflammation_flag"] = int(
        safe_get(labs, "crp") > 10 or
        safe_get(labs, "esr") > 20 or
        safe_get(labs, "wbc") > 11000 or
        safe_get(labs, "procalcitonin") > 0.5
    )

    features["electrolyte_flag"] = int(
        not (136 <= safe_get(labs, "sodium") <= 145) or
        not (3.5 <= safe_get(labs, "potassium") <= 5.0)
    )

    features["vitamin_flag"] = int(
        safe_get(labs, "vitamin_d") < 30 or
        safe_get(labs, "vitamin_b12") < 200 or
        safe_get(labs, "folate") < 3.0
    )

    # ── COMPOSITES ──
    features["lipid_index"] = round(
        (ldl * 0.4) + (features["chol_hdl_ratio"] * 0.3) + (trig * 0.0003), 3
    )

    features["metabolic_syndrome_score"] = int(
        (trig > 150) +
        (safe_get(labs, "hdl") < 40) +
        (glc > 100) +
        (chol > 200)
    )

    features["hemogram_score"] = round(
        abs(compute_zscore(hb, 12, 17)) +
        abs(compute_zscore(wbc, 4500, 11000)) +
        abs(compute_zscore(plt, 150000, 450000)),
        3
    )

    features["iron_deficiency_score"] = round(
        (iron < 60) * 2 +
        (fer < 12) * 2 +
        (tibc > 370),
        3
    )

    features["sepsis_risk_score"] = round(
        (safe_get(labs, "procalcitonin") > 0.5) * 3 +
        (safe_get(labs, "crp") > 50) * 2 +
        ((wbc > 12000 or wbc < 4000) * 2),
        3
    )

    return pd.DataFrame([features])
# ─────────────────────────────────────────────
# SECTION 10: TREND ANALYSIS ENGINE
# ─────────────────────────────────────────────

SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


def make_sparkline(values: list, width: int = 8) -> str:
    """Generate ASCII sparkline."""
    if len(values) < 2:
        return "─" * width

    mn, mx = min(values), max(values)
    rng = mx - mn or 1

    bars = [
        SPARKLINE_CHARS[
            min(int((v - mn) / rng * (len(SPARKLINE_CHARS) - 1)),
                len(SPARKLINE_CHARS) - 1)
        ]
        for v in values
    ]

    return "".join(bars)


def analyze_trends(labs: dict, patient_id: str = None) -> dict:
    """
    UPGRADED v3.0: Trend engine with patient history support.
    """

    trend_df = None

    # ── PRIORITY 1: PATIENT HISTORY ──
    if patient_id:
        visit_df = get_visit_history(patient_id)
        if not visit_df.empty:
            trend_df = visit_df

    # ── PRIORITY 2: LEGACY DATA ──
    if trend_df is None:
        path = PATHS["lab_trends"]
        if path.exists():
            try:
                legacy = pd.read_csv(path)
                trend_df = legacy
            except Exception:
                trend_df = None

    if trend_df is None or trend_df.empty:
        return {
            "status": "no_history",
            "trends": {},
            "trend_score": 0,
            "sparklines": {}
        }

    trends = {}
    sparklines = {}
    worsening = 0

    KEY_LABS = [
        "hemoglobin", "glucose", "creatinine", "ldl", "alt",
        "hba1c", "wbc", "cholesterol", "tsh", "crp",
        "potassium", "sodium", "troponin", "uric_acid"
    ]

    for key in KEY_LABS:
        if key not in trend_df.columns or labs.get(key) is None:
            continue

        history = trend_df[key].dropna().tolist()
        if len(history) < 2:
            continue

        all_values = history + [labs[key]]
        sparklines[key] = make_sparkline(all_values)

        baseline = history[-3:] if len(history) >= 3 else history
        prev = float(np.mean(baseline))
        curr = labs[key]

        slope = curr - prev
        pct_change = safe_div(abs(slope), abs(prev)) * 100 if prev != 0 else 0

        low, high = REFERENCE.get(key, (0, 100))

        if curr > high or curr < low:
            direction = "worsening"
        elif slope < 0 and key == "hdl":
            direction = "worsening"
        else:
            direction = "improving" if abs(slope) < abs(prev * 0.1) else "stable"

        if direction == "worsening":
            worsening += 1

        trends[key] = {
            "previous": round(prev, 2),
            "current": round(curr, 2),
            "slope": round(slope, 2),
            "pct_change": round(pct_change, 1),
            "direction": direction,
            "icon": {"worsening": "", "improving": "", "stable": ""}.get(direction, "")
        }

    trend_score = min(worsening * 15, 100)

    return {
        "status": "analyzed",
        "trends": trends,
        "trend_score": trend_score,
        "worsening_count": worsening,
        "sparklines": sparklines
    }

# ─────────────────────────────────────────────
# SECTION 11: ML ENGINE (FIXED v4.1 STABLE)
# ─────────────────────────────────────────────

_FEATURE_SCHEMA: list | None = None


def _save_feature_schema(columns: list):
    global _FEATURE_SCHEMA
    _FEATURE_SCHEMA = columns

    try:
        with open(PATHS["feature_schema"], "w") as f:
            json.dump(columns, f)
        print(colorize(f"  Feature schema saved ({len(columns)})", C.GREEN))
    except Exception as e:
        print(colorize(f" Schema save error: {e}", C.YELLOW))


def _load_feature_schema():
    global _FEATURE_SCHEMA

    if _FEATURE_SCHEMA is not None:
        return _FEATURE_SCHEMA

    try:
        if PATHS["feature_schema"].exists():
            with open(PATHS["feature_schema"], "r") as f:
                _FEATURE_SCHEMA = json.load(f)
                return _FEATURE_SCHEMA
    except Exception as e:
        print(colorize(f" Schema load error: {e}", C.YELLOW))

    return None


# ─────────────────────────────
# TRAINING (FIXED)
# ─────────────────────────────

def train_model():
    print(colorize("\nTraining ML model...", C.CYAN))

    for key in ["features_final", "features_v2", "risk_dataset"]:
        path = PATHS[key]

        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
            print(colorize(f" Dataset: {path.name} ({len(df)})", C.CYAN))

            label_col = next(
                (c for c in ["risk_label", "label", "target", "outcome"]
                 if c in df.columns),
                None
            )

            if not label_col:
                continue

            # X FEATURES (STRICT)
            X = df.drop(columns=[label_col])
            X = X.select_dtypes(include=[np.number]).copy()
            X = X.fillna(0.0)

            #  FORCE SORTED FEATURE ORDER (CRITICAL FIX)
            X = X.reindex(sorted(X.columns), axis=1)

            y = df[label_col]

            # SAVE EXACT FEATURE SCHEMA
            _save_feature_schema(list(X.columns))

            if y.dtype == object:
                le = LabelEncoder()
                y = le.fit_transform(y)
                joblib.dump(le, PATHS["label_encoder"])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            )

            model.fit(X_train, y_train)

            joblib.dump(model, PATHS["risk_model"])

            print(colorize(" Model trained successfully", C.GREEN))
            return model

        except Exception as e:
            print(colorize(f"Training error: {e}", C.RED))

    return None


# ─────────────────────────────
# ML SCORING (FIXED STABLE LOGIC)
# ─────────────────────────────

def get_ml_score(feature_df: pd.DataFrame, model) -> tuple[float, bool]:

    try:
        schema = _load_feature_schema()

        if schema is None:
            print(colorize(" No schema → ML unsafe", C.YELLOW))
            return 0.0, True

        # FORCE ALIGNMENT (CRITICAL FIX)
        feature_df = feature_df.reindex(columns=schema, fill_value=0.0)
        feature_df = feature_df.astype(float)

        if not hasattr(model, "predict_proba"):
            return 0.0, True

        proba = model.predict_proba(feature_df)

        if proba is None or len(proba) == 0:
            return 0.0, True

        n_classes = proba.shape[1]

        if n_classes == 2:
            # Binary: P(high risk) * 100
            risk_score = float(proba[0][1]) * 100
        elif n_classes == 4:
            # Multi-class: CRITICAL=90, HIGH=65, MODERATE=35, LOW=10
            # Class order from LabelEncoder (alphabetical): CRITICAL=0, HIGH=1, LOW=2, MODERATE=3
            class_scores = {0: 90.0, 1: 65.0, 2: 10.0, 3: 35.0}
            risk_score = sum(proba[0][i] * class_scores.get(i, 50.0) for i in range(n_classes))
        else:
            # Generic: weighted by class index
            risk_score = float(np.dot(proba[0], np.linspace(0, 100, n_classes)))

        risk_score = max(0.0, min(100.0, round(float(risk_score), 2)))
        dummy_flag = isinstance(model, str)

        return risk_score, dummy_flag

    except Exception as e:
        print(colorize(f"ML scoring error: {e}", C.RED))
        return 0.0, True
# ─────────────────────────────────────────────
# SECTION 12: RULE ENGINE (CLINICAL LOGIC)
# ─────────────────────────────────────────────

def compute_rule_score(labs: dict, symptoms_text: str = "") -> dict:
    score = 0
    triggered = []

    def add(label: str, points: int):
        nonlocal score
        score += points
        triggered.append({"rule": label, "points": points})

    def V(k):
        return labs.get(k)

    hb = V("hemoglobin")
    wbc = V("wbc")
    plt = V("platelets")
    glc = V("glucose")
    a1c = V("hba1c")
    crt = V("creatinine")
    bun = V("bun")
    egfr = V("egfr")
    chol = V("cholesterol")
    ldl = V("ldl")
    hdl = V("hdl")
    trig = V("triglycerides")
    alt = V("alt")
    ast = V("ast")
    bil = V("bilirubin")
    dbil = V("direct_bilirubin")
    tsh = V("tsh")
    crp = V("crp")
    esr = V("esr")
    trop = V("troponin")
    fer = V("ferritin")
    iron = V("iron")
    tibc = V("tibc")
    na = V("sodium")
    k = V("potassium")
    ca = V("calcium")
    mg = V("magnesium")
    bicar = V("bicarbonate")
    ua = V("uric_acid")
    inr = V("pt_inr")
    alb = V("albumin")
    pct = V("procalcitonin")
    vitd = V("vitamin_d")
    b12 = V("vitamin_b12")
    fol = V("folate")
    urine_alb = V("urine_albumin")
    ck = V("ck")
    bnp = V("bnp")
    neut = V("neutrophils")
    lymph = V("lymphocytes")

    # ── ANEMIA ──
    if hb is not None:
        if hb < 7: add("Critical Anemia", SEVERITY_WEIGHTS["critical"])
        elif hb < 8: add("Severe Anemia", SEVERITY_WEIGHTS["severe"])
        elif hb < 10: add("Moderate Anemia", SEVERITY_WEIGHTS["moderate"])
        elif hb < 12: add("Mild Anemia", SEVERITY_WEIGHTS["mild"])

    # Iron deficiency safety check
    if iron is not None and iron < 60:
        add("Low Iron", SEVERITY_WEIGHTS["mild"])
    if fer is not None and fer < 12:
        add("Low Ferritin", SEVERITY_WEIGHTS["mild"])
    if tibc is not None and tibc > 370:
        add("High TIBC", SEVERITY_WEIGHTS["mild"])

    if iron is not None and fer is not None and hb is not None:
        if iron < 60 and fer < 12 and hb < 12:
            add("Iron Deficiency Anemia", SEVERITY_WEIGHTS["moderate"])

    # ── WBC ──
    if wbc is not None:
        if wbc > 30000:
            add("Critical Leukocytosis", SEVERITY_WEIGHTS["critical"])
        elif wbc > 11000:
            add("Leukocytosis", SEVERITY_WEIGHTS["moderate"])
        elif wbc < 3000:
            add("Leukopenia", SEVERITY_WEIGHTS["mild"])

    # ── PLATELETS ──
    if plt is not None:
        if plt < 20000:
            add("Severe Thrombocytopenia", SEVERITY_WEIGHTS["critical"])
        elif plt < 100000:
            add("Thrombocytopenia", SEVERITY_WEIGHTS["severe"])

    # ── GLUCOSE ──
    if glc is not None:
        if glc > 400:
            add("Critical Hyperglycemia", SEVERITY_WEIGHTS["critical"])
        elif glc > 140:
            add("Hyperglycemia", SEVERITY_WEIGHTS["moderate"])

    if a1c is not None:
        if a1c > 9:
            add("Uncontrolled Diabetes", SEVERITY_WEIGHTS["severe"])
        elif a1c > 6.5:
            add("Diabetes", SEVERITY_WEIGHTS["moderate"])

    # ── KIDNEY ──
    if crt is not None and crt > 1.3:
        add("Renal Dysfunction", SEVERITY_WEIGHTS["moderate"])

    if egfr is not None and egfr < 60:
        add("CKD Risk", SEVERITY_WEIGHTS["moderate"])

    # ── LIVER ──
    if alt is not None and alt > 56:
        add("Elevated ALT", SEVERITY_WEIGHTS["mild"])
    if ast is not None and ast > 40:
        add("Elevated AST", SEVERITY_WEIGHTS["mild"])

    # ── INFLAMMATION ──
    if crp is not None and crp > 10:
        add("Inflammation", SEVERITY_WEIGHTS["mild"])

    # ── ELECTROLYTES ──
    if na is not None and (na < 136 or na > 145):
        add("Sodium Imbalance", SEVERITY_WEIGHTS["mild"])
    if k is not None and (k < 3.5 or k > 5.0):
        add("Potassium Imbalance", SEVERITY_WEIGHTS["mild"])

    # ── SYMPTOMS ──
    if symptoms_text:
        symp = symptoms_text.lower()
        if "chest pain" in symp:
            add("Chest Pain Symptom", 15)
        if "fever" in symp:
            add("Fever Symptom", 5)
        if "fatigue" in symp:
            add("Fatigue Symptom", 5)

    return {
        "score": min(score, 100),
        "triggered_rules": triggered
    }
def map_diseases(labs: dict) -> list:
    """
    UPGRADED v3.1: Safe clinical disease mapping.
    Lab-confirmed only. No inference beyond thresholds.
    """

    diseases = []

    def add_if(condition: bool, name: str):
        if condition:
            diseases.append(name)

    g    = labs.get("glucose")
    a1c  = labs.get("hba1c")
    crt  = labs.get("creatinine")
    bun  = labs.get("bun")
    egfr = labs.get("egfr")
    ldl  = labs.get("ldl")
    hdl  = labs.get("hdl")
    trig = labs.get("triglycerides")
    chol = labs.get("cholesterol")
    hb   = labs.get("hemoglobin")
    alt  = labs.get("alt")
    ast  = labs.get("ast")
    bil  = labs.get("bilirubin")
    tsh  = labs.get("tsh")
    crp  = labs.get("crp")
    wbc  = labs.get("wbc")
    iron = labs.get("iron")
    fer  = labs.get("ferritin")
    trop = labs.get("troponin")
    plt  = labs.get("platelets")
    na   = labs.get("sodium")
    k    = labs.get("potassium")
    inr  = labs.get("pt_inr")
    alb  = labs.get("albumin")
    ua   = labs.get("uric_acid")
    vitd = labs.get("vitamin_d")
    b12  = labs.get("vitamin_b12")
    pct  = labs.get("procalcitonin")
    bnp  = labs.get("bnp")
    urine_alb = labs.get("urine_albumin")
    tibc = labs.get("tibc")

    # ─────────────────────────────
    # Diabetes
    # ─────────────────────────────
    add_if(g is not None and g > 140 and a1c is not None and a1c > 6.5,
           "Type 2 Diabetes Mellitus (Confirmed — Dual Criteria)")

    add_if(g is not None and g > 140 and a1c is None,
           "Hyperglycemia (HbA1c Required for Confirmation)")

    add_if(a1c is not None and 5.7 < a1c <= 6.5,
           "Pre-Diabetes / Impaired Glucose Tolerance")

    # ─────────────────────────────
    # Kidney
    # ─────────────────────────────
    add_if(crt is not None and bun is not None and crt > 1.3 and bun > 20,
           "Chronic Kidney Disease (CKD — Lab Evidence)")

    if egfr is not None and egfr < 60:
        diseases.append(f"Reduced Renal Function (eGFR {float(egfr):.0f} mL/min/1.73m²)")

    add_if(urine_alb is not None and urine_alb > 30,
           "Microalbuminuria (Early Kidney Damage Risk)")

    # ─────────────────────────────
    # Gout
    # ─────────────────────────────
    add_if(ua is not None and ua > 7.2,
           "Hyperuricemia (Gout Risk)")

    # ─────────────────────────────
    # Lipids
    # ─────────────────────────────
    add_if((ldl is not None and ldl > 190) or (chol is not None and chol > 300),
           "Severe Hypercholesterolemia")

    add_if((ldl is not None and 100 < ldl <= 190) or (chol is not None and 200 < chol <= 300),
           "Dyslipidemia / Hypercholesterolemia")

    add_if(trig is not None and trig > 200,
           "Hypertriglyceridemia")

    add_if(hdl is not None and hdl < 40,
           "Low HDL (Cardiovascular Risk)")

    # Metabolic Syndrome
    met_criteria = sum([
        bool(trig and trig > 150),
        bool(hdl and hdl < 40),
        bool(g and g > 100),
        bool(chol and chol > 200),
    ])

    if met_criteria >= 3:
        diseases.append(f"Metabolic Syndrome ({met_criteria}/4 criteria met)")

    # ─────────────────────────────
    # Anemia
    # ─────────────────────────────
    add_if(hb is not None and hb < 7,
           "Severe Anemia (Critical)")

    add_if(hb is not None and 7 <= hb < 12,
           "Anemia (Mild–Moderate)")

    add_if(iron is not None and iron < 60 and fer is not None and fer < 12,
           "Iron Deficiency Anemia")

    add_if(iron is not None and iron < 60 and tibc is not None and tibc > 370,
           "Iron Deficiency Pattern (High TIBC)")

    add_if(b12 is not None and b12 < 200,
           "Vitamin B12 Deficiency (Megaloblastic Risk)")

    # ─────────────────────────────
    # Liver
    # ─────────────────────────────
    add_if((alt and alt > 200) or (ast and ast > 200),
           "Acute Liver Injury / Hepatitis")

    add_if((alt and 56 < alt <= 200) or (ast and 40 < ast <= 200),
           "Liver Dysfunction (Elevated Enzymes)")

    add_if(bil is not None and bil > 3.0,
           "Hyperbilirubinemia / Jaundice")

    add_if(inr is not None and inr > 1.5,
           "Coagulopathy (Liver Synthetic Dysfunction)")

    add_if(alb is not None and alb < 3.0,
           "Hypoalbuminemia")

    # ─────────────────────────────
    # Thyroid
    # ─────────────────────────────
    add_if(tsh is not None and tsh > 4.0,
           "Hypothyroidism")

    add_if(tsh is not None and tsh < 0.4,
           "Hyperthyroidism")

    # ─────────────────────────────
    # Infection / Sepsis
    # ─────────────────────────────
    add_if(wbc is not None and crp is not None and wbc > 11000 and crp > 10,
           "Active Infection / Inflammation")

    add_if(pct is not None and pct > 2.0,
           "Probable Bacterial Sepsis")

    if pct and crp and wbc:
        if pct > 2 and crp > 50 and (wbc > 12000 or wbc < 4000):
            diseases.append("Sepsis (Triple Marker Positive)")

    # ─────────────────────────────
    # Cardiac
    # ─────────────────────────────
    add_if(trop is not None and trop > 0.04,
           "Myocardial Injury / Possible ACS")

    add_if(bnp is not None and bnp > 400,
           "Heart Failure")

    # ─────────────────────────────
    # Electrolytes
    # ─────────────────────────────
    add_if(na is not None and (na < 130 or na > 150),
           "Dysnatremia")

    add_if(k is not None and (k < 3.0 or k > 5.5),
           "Dyskalemia")

    # ─────────────────────────────
    # Vitamins
    # ─────────────────────────────
    add_if(vitd is not None and vitd < 20,
           "Vitamin D Deficiency")

    # ─────────────────────────────
    # Platelets
    # ─────────────────────────────
    add_if(plt is not None and plt < 100000,
           "Thrombocytopenia")

    return list(dict.fromkeys(diseases))


def ensemble_score(
    ml: float,
    rule: float,
    trend: float,
    ml_is_dummy: bool = False
) -> dict:
    """
    UPGRADED v3.1: Stable ensemble scoring system.
    Handles dummy ML fallback safely.
    """

    # ─────────────────────────────
    # Weight selection
    # ─────────────────────────────
    if ml_is_dummy:
        final = (rule * 0.65) + (trend * 0.35)
        weight_note = "Rule 65% + Trend 35% (ML inactive / placeholder)"
    else:
        final = (ml * 0.40) + (rule * 0.30) + (trend * 0.30)
        weight_note = "ML 40% + Rule 30% + Trend 30%"

    final = max(0, min(round(final, 2), 100))

    # ─────────────────────────────
    # Risk categorization
    # ─────────────────────────────
    if final < 25:
        level, icon = "LOW", "🟢"
    elif final < 50:
        level, icon = "MODERATE", "🟡"
    elif final < 75:
        level, icon = "HIGH", "🔴"
    else:
        level, icon = "CRITICAL", "🚨"

    return {
        "final_score": final,
        "level": level,
        "icon": icon,
        "breakdown": {
            "ml_score": round(ml, 2),
            "rule_score": round(rule, 2),
            "trend_score": round(trend, 2),
            "weights": weight_note,
            "ml_is_dummy": ml_is_dummy,
        },
    }

def explain_risk(labs: dict, interp: dict,
                 rule_result: dict, score_result: dict) -> dict:
    """
    SHAP-style explanation — top 5 contributing factors.
    """
    contributors = []

    # ── Rule-based contributions ──
    for rule_item in rule_result.get("triggered_rules", []):
        contributors.append({
            "factor": rule_item.get("rule", "Unknown Rule"),
            "impact": rule_item.get("points", 0),
            "direction": "↑ Risk",
        })

    # ── Lab deviation contributions ──
    for key, status in interp.items():
        if status in ["high", "low"] and labs.get(key) is not None:
            low_ref, high_ref = REFERENCE.get(key, (0, 100))

            try:
                mid = (low_ref + high_ref) / 2
                deviation = abs(labs[key] - mid)
            except Exception:
                deviation = 0

            contributors.append({
                "factor": f"{key.replace('_', ' ').title()} ({status.upper()}: {labs[key]})",
                "impact": round(deviation * 0.1, 1),
                "direction": "↑ Risk" if status == "high" else "↓ Below Normal",
            })

    # ── Sort by impact ──
    contributors.sort(key=lambda x: x["impact"], reverse=True)

    top5 = contributors[:5]

    return {
        "top_factors": top5,
        "total_contributors": len(contributors),
        "highest_impact": top5[0] if top5 else None,
    }


def get_llm_explanation(
    labs: dict,
    interp: dict,
    diseases: list,
    score_result: dict,
    explain_result: dict,
    out_lang: str,
) -> str:
    """
    Tighter, safer medical explanation engine (LLM-based).
    """

    if not groq_client:
        return " LLM unavailable. Please check Groq API key."

    # ── Language rules ──
    lang_instruction = {
        "en": "Respond ONLY in English.",
        "ur": "صرف اردو میں جواب دیں۔ (Respond ONLY in Urdu)",
        "ar": "الرجاء الرد باللغة العربية فقط. (Respond ONLY in Arabic)",
    }.get(out_lang, "Respond ONLY in English.")

    # ── Top contributing factors ──
    top_factors = explain_result.get("top_factors", [])

    factors_text = "\n".join([
        f"  - {f.get('factor', 'Unknown')} [{f.get('direction', '')}, Impact: {f.get('impact', 0)}]"
        for f in top_factors
    ]) or "  None identified"

    # ── Only abnormal labs ──
    abnormal_only = {
        k: f"{v} (value: {labs.get(k)})"
        for k, v in interp.items()
        if v in ["high", "low"] and labs.get(k) is not None
    }

    # ── Prompt ──
    prompt = f"""
You are a STRICT, SAFE medical AI assistant. Your role is clinical decision-support ONLY.

 ABSOLUTE RULES:
1. ONLY analyze provided lab values — no external assumptions
2. NEVER introduce diseases not in CONFIRMED CONDITIONS
3. If data is missing → say "More testing required"
4. NEVER prescribe medications
5. Use cautious language ("suggestive of", "consistent with")
6. {lang_instruction}

══════════════ PATIENT DATA ══════════════
ABNORMAL LAB VALUES:
{json.dumps(abnormal_only, indent=2)}

CONFIRMED CONDITIONS:
{chr(10).join(f"• {d}" for d in diseases) if diseases else "• None identified"}

RISK SCORE:
{score_result.get('final_score', 0)}/100 — {score_result.get('level', '')} {score_result.get('icon', '')}

ML Dummy Mode:
{score_result.get('breakdown', {}).get('ml_is_dummy', False)}

TOP CONTRIBUTING FACTORS:
{factors_text}

══════════════════════════════════════════

Respond EXACTLY in this format:

##  KEY FINDINGS
2–4 sentences summarizing lab findings only.

##  CONFIRMED CONDITIONS
Bullet list ONLY from provided conditions.

##  RISK ASSESSMENT
Explain risk score and main drivers.

##  RECOMMENDATIONS
Non-drug clinical guidance only.

##  MISSING TESTS SUGGESTED
3–5 relevant diagnostic tests.

##  LIFESTYLE ADVICE
Safe lifestyle guidance only.

 DISCLAIMER: AI support only — consult a physician.
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.05,
            max_tokens=1800,
        )
        return response.choices[0].message.content

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"LLM Error: {type(e).__name__}: {e}"

def detect_language(text: str) -> str:
    """
    Detect input language safely.
    """
    try:
        return detect(text) if len(text.split()) >= 3 else "en"
    except Exception:
        return "en"


def translate_text(text: str, target_lang: str) -> str:
    """
    Translate text using GoogleTranslator with fallback safety.
    """
    if not text:
        return text

    if target_lang in ("auto", "en"):
        return text

    try:
        return GoogleTranslator(
            source='auto',
            target=target_lang
        ).translate(text)
    except Exception:
        return text

def severity_bar(score: float, width: int = 20) -> str:
    """
    Visual severity bar for terminal display.
    """
    score = max(0, min(score, 100))  # clamp safely

    filled = int(score / 100 * width)
    bar = "█" * filled + "░" * (width - filled)

    if score < 25:
        return colorize(f"[{bar}]", C.GREEN)
    elif score < 50:
        return colorize(f"[{bar}]", C.YELLOW)
    elif score < 75:
        return colorize(f"[{bar}]", C.RED)
    else:
        return colorize(f"[{bar}]", C.BG_RED + C.WHITE)


def format_report(
    labs: dict,
    interp: dict,
    diseases: list,
    score_result: dict,
    rule_result: dict,
    trend_result: dict,
    explain_result: dict,
    llm_output: str,
    out_lang: str,
    patient_id: str = None,
    patient_name: str = None,  # ADD THIS
    report_date: str = None,   # ADD THIS
) -> str:

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    divider = "─" * 60

    # ── LAB TABLE ──
    lab_lines = []

    for key, val in labs.items():
        if val is None:
            continue

        status = interp.get(key, "unknown")
        ref = REFERENCE.get(key, ("?", "?"))

        badge = status_badge(status)

        lab_lines.append(
            f"  {key.replace('_',' ').upper():<22} "
            f"{colorize(str(val), C.BOLD):<14} "
            f"[Ref: {ref[0]}–{ref[1]}] {badge}"
        )

    lab_table = "\n".join(lab_lines) if lab_lines else "  No labs parsed"

    # ── TREND SECTION ──
    trend_lines = []

    if trend_result.get("status") == "analyzed" and trend_result.get("trends"):
        for lab, t in trend_result["trends"].items():

            spark = trend_result.get("sparklines", {}).get(lab, "")

            dir_color = {
                "worsening": C.RED,
                "improving": C.GREEN,
                "stable": C.CYAN,
            }.get(t.get("direction", ""), C.RESET)

            trend_lines.append(
                f"  {t.get('icon','')} {lab.upper():<18} "
                f"{t.get('previous','?')} → {colorize(str(t.get('current','?')), C.BOLD)}  "
                f"Δ {t.get('slope',0):+.2f} ({t.get('pct_change',0):+.1f}%)  "
                f"{colorize(spark, dir_color)}  "
                f"{colorize(t.get('direction','').upper(), dir_color)}"
            )
    else:
        trend_lines = [""]

    # ── RULE ENGINE ──
    rule_lines = []

    for r in rule_result.get("triggered_rules", [])[:10]:
        pts = r.get("points", 0)

        if pts >= 50:
            col = C.RED + C.BOLD
        elif pts >= 25:
            col = C.YELLOW
        else:
            col = C.DIM

        rule_lines.append(
            f"  {colorize(f'+{pts:>3}', col)} pts  {r.get('rule','Unknown Rule')}"
        )

    # ── SHAP FACTORS ──
    factor_lines = []

    for i, f in enumerate(explain_result.get("top_factors", []), 1):
        factor_lines.append(
            f"  {i}. {f.get('factor','Unknown')}\n"
            f"     {colorize(f.get('direction',''), C.YELLOW)}  |  Impact: {f.get('impact',0)}"
        )

    # ── SCORE DISPLAY ──
    lv = score_result.get("level", "UNKNOWN")
    sc = score_result.get("final_score", 0)

    bar = severity_bar(sc)

    lvl_display = colorize(
        f"  {lv} {score_result.get('icon','')}",
        risk_color(lv) + C.BOLD
    )

    breakdown = score_result.get("breakdown", {})

    report = f"""
{colorize("╔══════════════════════════════════════════════════════════════╗", C.CYAN)}
{colorize("║    AI LAB RISK ANALYSIS REPORT  v3.0", C.CYAN + C.BOLD)}{colorize("                         ║", C.CYAN)}
{colorize("║      Clinical Decision-Support ONLY — Not a Diagnosis       ║", C.CYAN)}
{colorize("╚══════════════════════════════════════════════════════════════╝", C.CYAN)}

{colorize("Generated:", C.DIM)} {report_date or ts}   {colorize(f"Patient: {patient_name or patient_id or 'Anonymous'}", C.DIM)}

{colorize("━━━   LAB VALUES & INTERPRETATION  ━━━━━━━━━━━━━━━━━━━━━━━━━", C.BLUE)}
{lab_table}

{colorize("━━━   TREND ANALYSIS  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", C.BLUE)}
{"".join(trend_lines)}

{colorize("━━━   RISK SCORE  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", C.BLUE)}
  {bar}  {colorize(str(sc), C.BOLD)}/100
{lvl_display}



{colorize("━━━   TOP CONTRIBUTING FACTORS (SHAP-style)  ━━━━━━━━━━━━━━━", C.BLUE)}
{"".join(factor_lines) or "  None identified"}

{colorize("━━━  CONFIRMED CONDITIONS  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", C.BLUE)}
{chr(10).join(colorize(f"  • {d}", C.YELLOW) for d in diseases) or colorize("  • None identified", C.GREEN)}

{colorize("━━━   AI CLINICAL EXPLANATION  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", C.BLUE)}
{llm_output}

{colorize("╔══════════════════════════════════════════════════════════════╗", C.DIM)}
{colorize("║   AI support only — not a medical diagnosis.               ║", C.DIM)}
{colorize("╚══════════════════════════════════════════════════════════════╝", C.DIM)}
"""

    return report

# ─────────────────────────────────────────────
# SECTION 19: MAIN PIPELINE (CLEAN v3.7)
# ─────────────────────────────────────────────

print(colorize("\n Loading ML model + encoder...", C.CYAN))
import joblib

MODEL_PATH = PATHS["risk_model"]
ENCODER_PATH = PATHS["label_encoder"]

try:
    ml_model = joblib.load(MODEL_PATH)

    try:
        label_encoder = joblib.load(ENCODER_PATH)
    except:
        label_encoder = None

    EXPECTED_FEATURES = getattr(
        ml_model,
        "n_features_in_",
        None
    )

except Exception:
    ml_model = None
    label_encoder = None
    EXPECTED_FEATURES = None


# ─────────────────────────────
# SAFE TREND FALLBACK
# ─────────────────────────────

def _safe_trends(labs, patient_id):

    try:
        fn = globals().get("analyze_trends")

        if callable(fn):
            result = fn(labs, patient_id)

            if isinstance(result, dict):
                return result

    except Exception:
        pass

    return {
        "trend_score": 0,
        "status": "none",
        "trends": {},
        "history": None
    }


# ─────────────────────────────
# SAFE ML WRAPPER
# ─────────────────────────────

def _run_ml(feature_df):

    if ml_model is None:
        return 50.0, True

    try:
        return get_ml_score(feature_df, ml_model)

    except Exception:
        return 50.0, True


# ─────────────────────────────
# SAFE FLOAT
# ─────────────────────────────

def _safe_float(v):

    if v is None:
        return 0.0

    try:
        return float(v)
    except:
        return 0.0


# ─────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────

def run_pipeline(
    raw_text: str,
    out_lang: str = "en",
    patient_id: str = None,
    symptoms: str = "",
) -> None:

    print("\n" + colorize("=" * 62, C.CYAN))
    print(colorize("   RUNNING ANALYSIS PIPELINE v3.7", C.CYAN + C.BOLD))
    print(colorize("=" * 62, C.CYAN))

    combined_input = (
        raw_text + " " + symptoms
    ).strip()

    # 1
    print(colorize("\n[1/8] Parsing lab values...", C.CYAN))
    labs = parse_labs(combined_input) or {}
    # ── EXTRACT PATIENT NAME AND DATE FROM LAB REPORT TEXT ──
    patient_name_from_report = extract_patient_name_from_text(combined_input)
    report_date_from_report = extract_date_from_text(combined_input)
    
    # Use extracted name if available
    display_name = patient_name_from_report
    if not display_name and patient_id:
        # Fall back to registered name if available
        df = load_patient_registry()
        patient_row = df[df["patient_id"] == patient_id]
        if not patient_row.empty:
            display_name = patient_row.iloc[0].get("name", None)


    # 2
    print(colorize("[2/8]  Interpreting values...", C.CYAN))
    interp = interpret_labs(labs) or {}

    # 3
    print(colorize("[3/8]  Building feature vector...", C.CYAN))

    feature_df = build_feature_vector(labs)

    try:
        feature_df = feature_df.fillna(0)

        for col in feature_df.columns:
            feature_df[col] = feature_df[col].apply(_safe_float)

    except Exception:
        pass

    # 4
    print(colorize("[4/8]  Computing ML score...", C.CYAN))

    ml_score, ml_dummy = _run_ml(feature_df)

    # 5
    print(colorize("[5/8]  Applying clinical rules...", C.CYAN))

    rule_result = compute_rule_score(
        labs,
        combined_input
    )

    # 6
    print(colorize("[6/8] Analyzing trends...", C.CYAN))

    trend_result = _safe_trends(
        labs,
        patient_id
    )

    trend_score = trend_result.get(
        "trend_score",
        0
    )

    # 7
    print(colorize("[7/8]  Computing ensemble score...", C.CYAN))

    score_result = ensemble_score(
        ml_score,
        rule_result["score"],
        trend_score,
        ml_dummy
    )

    print(
        f"       FINAL: "
        f"{score_result['final_score']}/100 "
        f"— {score_result['level']}"
    )

    # 8
    print(colorize(
        "\n[8/8] Generating LLM explanation...",
        C.CYAN
    ))

    diseases = map_diseases(labs)

    explain_result = explain_risk(
        labs,
        interp,
        rule_result,
        score_result
    )

    llm_output = get_llm_explanation(
        labs,
        interp,
        diseases,
        score_result,
        explain_result,
        out_lang
    )

    report = format_report(
        labs,
        interp,
        diseases,
        score_result,
        rule_result,
        trend_result,
        explain_result,
        llm_output,
        out_lang,
        patient_id,
        display_name,
        report_date_from_report
    )
    

    print(report)

    if patient_id:

        save_visit(
            patient_id,
            labs,
            score_result["final_score"],
            score_result["level"]
        )
# ─────────────────────────────────────────────
# SECTION 20: MAIN LOOP
# ─────────────────────────────────────────────

if __name__ == "__main__":

    LANG_MAP = {"1": "en", "2": "ur", "3": "ar", "4": "auto"}

    while True:
        print("\n" + colorize("╔" + "═" * 44 + "╗", C.CYAN))
        print(colorize("║    AI LAB RISK AWARENESS SYSTEM v3.0    ║", C.CYAN + C.BOLD))
        print(colorize("║   Hackathon Edition — Full Stack Upgrade   ║", C.CYAN))
        print(colorize("╚" + "═" * 44 + "╝", C.CYAN))

        print("\n Output Language:")
        print("  1. English")
        print("  2. Urdu (اردو)")
        print("  3. Arabic (العربية)")
        print("  4. Auto-detect")

        lang_choice = input(colorize("\nSelect language [1-4]: ", C.CYAN)).strip()
        out_lang = LANG_MAP.get(lang_choice, "en")

        print("\nInput Method:")
        print("  1. Text Input")
        print("  2. Image File")
        print("  3. PDF Report")
        print("  4. Camera Scan")
        print("  5. Patient List")
        print("  6. Exit")

        choice = input(colorize("\nSelect [1-6]: ", C.CYAN)).strip()

        # EXIT
        if choice == "6":
            print(colorize("\n Goodbye! Stay healthy.", C.GREEN))
            sys.exit(0)

        # PATIENT LIST
        elif choice == "5":
            df = load_patient_registry()
            if df.empty:
                print(colorize("  No patients registered yet.", C.YELLOW))
            else:
                print(colorize("\n  REGISTERED PATIENTS", C.CYAN + C.BOLD))
                print(df.to_string(index=False))

        # TEXT INPUT
        elif choice == "1":
            print(colorize("\nPaste lab report text:", C.DIM))
            text = input().strip()

            if not text:
                print(colorize(" No text entered.", C.YELLOW))
                continue

            symptoms = input("Symptoms (optional): ").strip()
            patient_id = input("Patient ID (optional): ").strip() or None

            if patient_id:
                name = input("Name (optional): ").strip()
                age = input("Age (optional): ").strip()
                gender = input("Gender (M/F/Other): ").strip()
                register_patient(patient_id, name, age, gender)

            run_pipeline(text, out_lang, patient_id, symptoms)

        # IMAGE INPUT
        elif choice == "2":
            path = input("Image path: ").strip()

            try:
                text = extract_text_from_image(path)
                if not text:
                    print(colorize(" No text extracted.", C.RED))
                    continue

                symptoms = input("Symptoms (optional): ").strip()
                patient_id = input("Patient ID (optional): ").strip() or None

                if patient_id:
                    register_patient(patient_id)

                run_pipeline(text, out_lang, patient_id, symptoms)

            except Exception as e:
                print(colorize(f" Error: {e}", C.RED))

        # PDF INPUT
        elif choice == "3":
            path = input("PDF path: ").strip()

            try:
                text = extract_text_from_pdf(path)
                if not text:
                    print(colorize(" No text extracted.", C.RED))
                    continue

                symptoms = input("Symptoms (optional): ").strip()
                patient_id = input("Patient ID (optional): ").strip() or None

                if patient_id:
                    register_patient(patient_id)

                run_pipeline(text, out_lang, patient_id, symptoms)

            except Exception as e:
                print(colorize(f" Error: {e}", C.RED))

        # CAMERA INPUT
        elif choice == "4":
            img_path = capture_from_camera()

            if not img_path:
                print(colorize(" Camera capture failed.", C.RED))
                continue

            try:
                text = extract_text_from_image(img_path)

                if not text:
                    print(colorize(" No text extracted.", C.RED))
                    continue

                symptoms = input("Symptoms (optional): ").strip()
                patient_id = input("Patient ID (optional): ").strip() or None

                if patient_id:
                    register_patient(patient_id)

                run_pipeline(text, out_lang, patient_id, symptoms)

            except Exception as e:
                print(colorize(f" Error: {e}", C.RED))

        else:
            print(colorize("Invalid choice. Select 1–6.", C.YELLOW))