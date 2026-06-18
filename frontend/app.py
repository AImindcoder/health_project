"""
AI Lab Risk Awareness System — Streamlit UI
Communicates with FastAPI backend via HTTP only.
No imports from hackthathon.py.
"""

import os
import sys
import json
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import numpy as np
import streamlit as st
import requests

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

SESSION_TIMEOUT_KEY = "_last_fetch"


def _fetch(path, method="GET", **kwargs):
    url = urljoin(API_BASE, path)
    try:
        if method == "GET":
            r = requests.get(url, timeout=30, **kwargs)
        elif method == "POST":
            r = requests.post(url, timeout=60, **kwargs)
        elif method == "PUT":
            r = requests.put(url, timeout=30, **kwargs)
        elif method == "DELETE":
            r = requests.delete(url, timeout=30, **kwargs)
        else:
            return {"error": f"Unsupported method: {method}"}
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to {API_BASE}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except requests.exceptions.HTTPError as e:
        try:
            return r.json()
        except Exception:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def _get(path, **kwargs):
    return _fetch(path, "GET", **kwargs)


def _post(path, **kwargs):
    return _fetch(path, "POST", **kwargs)


@st.cache_data(ttl=60)
def _get_status():
    resp = _get("/api/status")
    if "error" not in resp:
        return resp
    # fallback to /health if /api/status not yet available
    h = _get("/health")
    if "error" in h:
        return h
    d = h.get("data", {})
    models_exist = d.get("models_exist", {})
    return {
        "success": True,
        "data": {
            "status": "healthy",
            "ml_model": {"status": "loaded" if models_exist.get("risk_model") else "not_loaded"},
            "label_encoder": {"status": "loaded" if models_exist.get("label_encoder") else "not_loaded"},
            "feature_schema": None,
            "ocr_engine": "unavailable",
            "groq_client": d.get("groq_configured", False),
            "groq_api_key_detected": d.get("groq_configured", False),
            "base_dir": "",
            "paths": {},
        },
    }


@st.cache_data(ttl=300)
def _get_reference():
    return _get("/api/reference")


st.set_page_config(
    page_title="AI Lab Risk Awareness",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

LANG_MAP = {"English": "en", "Urdu": "ur", "Arabic": "ar"}

for key in (
    "analysis_results",
    "extracted_text",
    "current_patient_id",
    "patient_history_df",
):
    if key not in st.session_state:
        st.session_state[key] = None if key != "extracted_text" else ""

status = _get_status()
ref_resp = _get_reference()
REFERENCE = {}
if ref_resp.get("success"):
    raw = ref_resp["data"]["reference"]
    REFERENCE = {k: (v["low"], v["high"]) for k, v in raw.items()}

api_ok = "error" not in status
status_data = status.get("data", {}) if api_ok else {}

with st.sidebar:
    st.markdown("## &#65039; AI Lab Risk")
    st.caption("Clinical Decision-Support Only")

    st.divider()

    st.markdown("###  System Status")

    if not api_ok:
        st.error(f"Backend unreachable: {status.get('error', 'unknown')}")
    else:
        ml = status_data.get("ml_model", {})
        model_loaded = ml.get("status") == "loaded"
        model_status = "&#9989; Loaded" if model_loaded else "&#9888; Not Loaded"
        st.markdown(f"**ML Model:** {model_status}")
        if model_loaded and ml.get("n_features"):
            st.caption(f"  Features: {ml['n_features']}")

        ocr_ok = status_data.get("ocr_engine") == "ready"
        ocr_status = "&#9989; Ready" if ocr_ok else "&#9888; Unavailable"
        st.markdown(f"**OCR Engine:** {ocr_status}")

        groq_ok = status_data.get("groq_client", False)
        groq_key = status_data.get("groq_api_key_detected", False)
        if groq_ok:
            llm_status = "&#9989; Connected"
        elif groq_key:
            llm_status = "&#9888; Init Failed"
        else:
            llm_status = "&#9888; No Key"
        st.markdown(f"**Groq LLM:** {llm_status}")

        schema = status_data.get("feature_schema")
        if schema:
            st.caption(f"Feature schema: {len(schema)} cols")

    st.divider()

    out_lang = st.selectbox(
        "Output Language",
        options=list(LANG_MAP.keys()),
        index=0,
    )
    st.caption(f"LLM will respond in {out_lang}")

    st.divider()
    base_dir = status_data.get("base_dir", "")
    st.caption(f"**BASE_DIR:** `{base_dir}`")
    st.caption(f"**Python:** {sys.version.split()[0]}")


def run_full_analysis(text: str, patient_id: str | None, symptoms: str, lang_code: str):
    combined = (text + " " + symptoms).strip()
    if not combined:
        st.warning("No input text provided.")
        return None

    payload = {"lab_text": combined, "symptoms": "", "patient_id": patient_id, "language": lang_code}
    resp = _post("/api/analysis/text", json=payload)
    if "error" in resp:
        st.error(f"Analysis failed: {resp['error']}")
        return None
    if not resp.get("success"):
        st.error(f"Analysis failed: {resp.get('message', 'unknown error')}")
        return None

    result = resp.get("data")
    if result is None:
        st.error("Analysis returned no data.")
        return None

    # Normalize field names — handle both old API format and new
    if "interpretation" in result and "interp" not in result:
        result["interp"] = result.pop("interpretation")
    if "interp" not in result:
        result["interp"] = {}

    if "score_result" not in result:
        result["score_result"] = {
            "final_score": result.pop("risk_score", 0),
            "level": result.pop("risk_level", "UNKNOWN"),
            "icon": "",
            "breakdown": result.pop("risk_breakdown", {}),
        }

    if "rule_result" not in result:
        result["rule_result"] = {"triggered_rules": result.pop("triggered_rules", [])}

    if "trend_result" not in result:
        result["trend_result"] = result.pop("trends", {"status": "no_history"})

    if "llm_output" not in result:
        result["llm_output"] = result.pop("llm_explanation", "")

    return result


def display_lab_table(labs, interp):
    if not labs:
        st.info("No lab values parsed.")
        return

    rows = []
    for key in REFERENCE:
        val = labs.get(key)
        status = interp.get(key, "not_tested")
        ref = REFERENCE.get(key, ("?", "?"))
        rows.append(
            {
                "Test": key.replace("_", " ").title(),
                "Value": f"{val}" if val is not None else "—",
                "Ref Range": f"{ref[0]}–{ref[1]}",
                "Status": status.replace("_", " ").title(),
            }
        )

    df = pd.DataFrame(rows)
    df = df[df["Value"] != "—"]

    if df.empty:
        st.info("No lab values parsed.")
        return

    def color_status(val):
        colors = {
            "normal": "background-color: #d4edda; color: #155724",
            "high": "background-color: #f8d7da; color: #721c24",
            "low": "background-color: #fff3cd; color: #856404",
            "not tested": "background-color: #e2e3e5; color: #383d41",
            "unknown ref": "background-color: #e2e3e5; color: #383d41",
        }
        return colors.get(val.lower(), "")

    styled = df.style.map(color_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def display_risk_score(score_result):
    score = score_result["final_score"]
    level = score_result["level"]
    icon = score_result.get("icon", "")
    breakdown = score_result.get("breakdown", {})

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if level == "LOW":
            st.success(f"## {icon} {level}")
        elif level == "MODERATE":
            st.warning(f"## {icon} {level}")
        elif level == "HIGH":
            st.error(f"## {icon} {level}")
        else:
            st.error(f"## {icon} {level}")

    with col2:
        st.metric("Risk Score", f"{score}/100")

    with col3:
        weight_note = breakdown.get("weights", "")
        st.markdown(f"**Weights:** {weight_note}")
        st.caption(f"ML: {breakdown.get('ml_score', 0)}  |  "
                   f"Rule: {breakdown.get('rule_score', 0)}  |  "
                   f"Trend: {breakdown.get('trend_score', 0)}")
        if breakdown.get("ml_is_dummy"):
            st.caption("⚠️ ML model is using fallback mode")


def display_triggered_rules(rule_result):
    rules = rule_result.get("triggered_rules", [])
    if not rules:
        st.info("No rules triggered.")
        return

    for r in rules:
        pts = r.get("points", 0)
        label = r.get("rule", "Unknown")
        if pts >= 50:
            st.error(f"**+{pts} pts** — {label}")
        elif pts >= 25:
            st.warning(f"**+{pts} pts** — {label}")
        else:
            st.info(f"+{pts} pts — {label}")


def display_conditions(diseases):
    if not diseases:
        st.success("No conditions identified.")
        return
    for d in diseases:
        st.markdown(f"- {d}")


def display_trends(trend_result):
    if trend_result.get("status") != "analyzed":
        st.info("No trend history available for this patient.")
        return

    trends = trend_result.get("trends", {})
    sparklines = trend_result.get("sparklines", {})

    if not trends:
        st.info("No trend data found.")
        return

    rows = []
    for lab, t in trends.items():
        spark = sparklines.get(lab, "")
        rows.append(
            {
                "Lab": lab.replace("_", " ").title(),
                "Previous": t.get("previous", ""),
                "Current": t.get("current", ""),
                "Δ": t.get("slope", 0),
                "Change %": t.get("pct_change", 0),
                "Direction": t.get("direction", "").title(),
                "Sparkline": spark,
            }
        )

    df = pd.DataFrame(rows)

    def dir_color(val):
        if val.lower() == "worsening":
            return "color: red; font-weight: bold"
        elif val.lower() == "improving":
            return "color: green; font-weight: bold"
        return ""

    styled = df.style.map(dir_color, subset=["Direction"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    worsening = trend_result.get("worsening_count", 0)
    if worsening > 0:
        st.warning(f"{worsening} lab(s) showing worsening trend")


def display_llm_output(llm_output):
    if not llm_output:
        st.info("LLM explanation unavailable.")
        return

    if llm_output.startswith("LLM Error"):
        st.error(llm_output)
        return

    if llm_output.startswith("LLM unavailable"):
        st.warning(llm_output)
        return

    st.markdown(llm_output)


def display_all_results(results):
    if results is None:
        return

    tabs = st.tabs([
        " Lab Values",
        " Risk Assessment",
        " Clinical Rules",
        " Conditions",
        " Trends",
        " AI Explanation",
    ])

    with tabs[0]:
        display_lab_table(results["labs"], results["interp"])

    with tabs[1]:
        display_risk_score(results["score_result"])

    with tabs[2]:
        display_triggered_rules(results["rule_result"])

    with tabs[3]:
        display_conditions(results["diseases"])

    with tabs[4]:
        display_trends(results["trend_result"])

    with tabs[5]:
        display_llm_output(results["llm_output"])


def render_text_input_tab():
    st.markdown("### Paste Lab Report Text")

    text = st.text_area(
        "Lab report text",
        height=200,
        placeholder="Paste lab values here...\n\nExample:\nHemoglobin: 13.2 g/dL\nWBC: 8500 /μL\nGlucose: 110 mg/dL",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        symptoms = st.text_input("Symptoms (optional)", placeholder="e.g. fever, chest pain")
    with col2:
        patient_id = st.text_input("Patient ID (optional)", placeholder="e.g. P001")

    reg_name = reg_age = ""
    reg_gender = ""
    if patient_id:
        with st.expander("Patient Registration (optional)"):
            reg_col1, reg_col2, reg_col3 = st.columns(3)
            with reg_col1:
                reg_name = st.text_input("Name", key="txt_name")
            with reg_col2:
                reg_age = st.text_input("Age", key="txt_age")
            with reg_col3:
                reg_gender = st.selectbox(
                    "Gender", ["", "M", "F", "Other"], key="txt_gender"
                )

    if st.button("Analyze Text", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter lab report text.")
            return

        pid = patient_id.strip() or None

        if pid:
            _post("/api/patients", json={
                "patient_id": pid,
                "name": reg_name.strip(),
                "age": reg_age.strip(),
                "gender": reg_gender,
            })

        lang_code = LANG_MAP.get(out_lang, "en")
        results = run_full_analysis(text, pid, symptoms, lang_code)
        st.session_state.analysis_results = results

    if st.session_state.analysis_results is not None:
        st.divider()
        display_all_results(st.session_state.analysis_results)


def render_pdf_tab():
    st.markdown("### Upload PDF Lab Report")

    pdf_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"], label_visibility="collapsed"
    )

    if pdf_file is not None:
        st.success(f"Uploaded: {pdf_file.name} ({pdf_file.size / 1024:.1f} KB)")

        if st.button("Extract Text from PDF", type="primary", use_container_width=True):
            with st.spinner("Extracting text via OCR..."):
                try:
                    resp = _post("/api/extract/pdf", files={"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")})
                    if "error" in resp:
                        st.error(f"PDF extraction failed: {resp['error']}")
                    elif resp.get("success"):
                        text = resp["data"]["text"]
                        st.session_state.extracted_text = text
                        st.success(f"Extracted {len(text)} characters")
                        with st.expander("View extracted text"):
                            st.text(text[:3000])
                    else:
                        st.error("No text could be extracted from the PDF.")
                except Exception as e:
                    st.error(f"PDF extraction failed: {e}")

    if st.session_state.extracted_text:
        st.divider()
        st.markdown("### Analyze Extracted Text")

        col1, col2 = st.columns(2)
        with col1:
            symptoms = st.text_input("Symptoms (optional)", key="pdf_symptoms")
        with col2:
            patient_id = st.text_input("Patient ID (optional)", key="pdf_pid")

        pdf_reg_name = pdf_reg_age = ""
        pdf_reg_gender = ""
        if patient_id:
            with st.expander("Patient Registration (optional)"):
                r1, r2, r3 = st.columns(3)
                with r1:
                    pdf_reg_name = st.text_input("Name", key="pdf_name")
                with r2:
                    pdf_reg_age = st.text_input("Age", key="pdf_age")
                with r3:
                    pdf_reg_gender = st.selectbox("Gender", ["", "M", "F", "Other"], key="pdf_gender")

        if st.button("Analyze PDF Text", type="primary", use_container_width=True):
            pid = patient_id.strip() or None
            if pid:
                _post("/api/patients", json={
                    "patient_id": pid,
                    "name": pdf_reg_name.strip(),
                    "age": pdf_reg_age.strip(),
                    "gender": pdf_reg_gender,
                })
            lang_code = LANG_MAP.get(out_lang, "en")
            results = run_full_analysis(st.session_state.extracted_text, pid, symptoms, lang_code)
            st.session_state.analysis_results = results

        if st.session_state.analysis_results is not None:
            st.divider()
            display_all_results(st.session_state.analysis_results)


def render_image_tab():
    st.markdown("### Upload Lab Report Image")

    img_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
        label_visibility="collapsed",
    )

    if img_file is not None:
        st.image(img_file, caption=img_file.name, width=400)

        if st.button("Run OCR", type="primary", use_container_width=True):
            with st.spinner("Running 3-pass OCR..."):
                try:
                    mime = f"image/{img_file.name.split('.')[-1].lower()}"
                    resp = _post("/api/extract/image", files={"file": (img_file.name, img_file.getvalue(), mime)})
                    if "error" in resp:
                        st.error(f"OCR failed: {resp['error']}")
                    elif resp.get("success"):
                        text = resp["data"]["text"]
                        st.session_state.extracted_text = text
                        st.success(f"Extracted {len(text)} characters")
                        with st.expander("View extracted text"):
                            st.text(text[:3000])
                    else:
                        st.error("No text could be extracted from the image.")
                except Exception as e:
                    st.error(f"OCR failed: {e}")

    if st.session_state.extracted_text:
        st.divider()
        st.markdown("### Analyze Extracted Text")

        col1, col2 = st.columns(2)
        with col1:
            symptoms = st.text_input("Symptoms (optional)", key="img_symptoms")
        with col2:
            patient_id = st.text_input("Patient ID (optional)", key="img_pid")

        img_reg_name = img_reg_age = ""
        img_reg_gender = ""
        if patient_id:
            with st.expander("Patient Registration (optional)"):
                r1, r2, r3 = st.columns(3)
                with r1:
                    img_reg_name = st.text_input("Name", key="img_name")
                with r2:
                    img_reg_age = st.text_input("Age", key="img_age")
                with r3:
                    img_reg_gender = st.selectbox("Gender", ["", "M", "F", "Other"], key="img_gender")

        if st.button("Analyze Image Text", type="primary", use_container_width=True):
            pid = patient_id.strip() or None
            if pid:
                _post("/api/patients", json={
                    "patient_id": pid,
                    "name": img_reg_name.strip(),
                    "age": img_reg_age.strip(),
                    "gender": img_reg_gender,
                })
            lang_code = LANG_MAP.get(out_lang, "en")
            results = run_full_analysis(st.session_state.extracted_text, pid, symptoms, lang_code)
            st.session_state.analysis_results = results

        if st.session_state.analysis_results is not None:
            st.divider()
            display_all_results(st.session_state.analysis_results)


def render_patient_history_tab():
    st.markdown("### Patient Registry & Visit History")

    col1, col2 = st.columns([3, 1])
    with col1:
        search_id = st.text_input("Patient ID", placeholder="e.g. P001")
    with col2:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    if search_clicked and search_id.strip():
        pid = search_id.strip()
        patients_resp = _get(f"/api/patients/{pid}")
        registry_df = _list_patients_df()
        patient_row = None

        if patients_resp.get("success"):
            info = patients_resp["data"]
            patient_row = info
            st.success(f"Patient found: {pid}")
            i1, i2, i3, i4 = st.columns(4)
            with i1:
                st.metric("Name", info.get("name", "Unknown"))
            with i2:
                st.metric("Age", info.get("age", "Unknown"))
            with i3:
                st.metric("Gender", info.get("gender", "Unknown"))
            with i4:
                st.metric("Visits", int(info.get("visit_count", 0)))
        else:
            st.warning(f"No patient found with ID: {pid}")

        visits_resp = _get(f"/api/patients/{pid}/visits")
        if visits_resp.get("success") and visits_resp["data"].get("visits"):
            visits = visits_resp["data"]["visits"]
            visits_df = pd.DataFrame(visits)
            st.session_state.patient_history_df = visits_df
            st.markdown(f"**Visit History:** {len(visits_df)} visits")

            display_cols = [c for c in ["timestamp", "risk_score", "risk_level"] if c in visits_df.columns]
            if display_cols:
                display_df = visits_df[display_cols].copy()
                if "risk_score" in display_df.columns:
                    display_df["risk_score"] = display_df["risk_score"].round(2)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            if "timestamp" in visits_df.columns and "risk_score" in visits_df.columns:
                chart_df = visits_df[["timestamp", "risk_score"]].dropna().copy()
                chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])
                chart_df = chart_df.sort_values("timestamp").set_index("timestamp")
                if len(chart_df) >= 2:
                    st.subheader("Risk Score Over Time")
                    st.line_chart(chart_df, height=250)

            lab_cols = [c for c in visits_df.columns if c in REFERENCE and c not in ("timestamp", "risk_score", "risk_level", "patient_id")]
            if lab_cols:
                st.subheader("Lab Value Trends")
                selected_lab = st.selectbox("Select lab to chart", lab_cols)
                if selected_lab:
                    lab_df = visits_df[["timestamp", selected_lab]].dropna().copy()
                    if not lab_df.empty and len(lab_df) >= 2:
                        lab_df["timestamp"] = pd.to_datetime(lab_df["timestamp"])
                        lab_df = lab_df.sort_values("timestamp").set_index("timestamp")
                        st.line_chart(lab_df, height=200)

                        ref = REFERENCE.get(selected_lab, (None, None))
                        if ref[0] is not None:
                            st.caption(f"Reference range: {ref[0]} – {ref[1]}")
        else:
            st.session_state.patient_history_df = None

    elif search_clicked:
        st.warning("Please enter a Patient ID.")

    st.divider()
    st.markdown("### All Registered Patients")
    registry_df = _list_patients_df()
    if registry_df is not None and not registry_df.empty:
        st.dataframe(registry_df, use_container_width=True, hide_index=True)
    else:
        st.info("No patients registered yet.")


@st.cache_data(ttl=30)
def _list_patients_df():
    resp = _get("/api/patients")
    if resp.get("success") and resp["data"].get("patients"):
        return pd.DataFrame(resp["data"]["patients"])
    return pd.DataFrame()


def render_debug_tab():
    s = _get_status()
    st.markdown("### System Diagnostics")

    if "error" in s:
        st.error(f"Backend unreachable: {s['error']}")
        return

    d = s.get("data", {})
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**ML Model**")
        ml = d.get("ml_model", {})
        if ml.get("status") == "loaded":
            st.success("Loaded")
            st.json({
                "type": ml.get("type"),
                "n_features": ml.get("n_features"),
                "n_classes": ml.get("n_classes"),
                "max_depth": ml.get("max_depth"),
                "n_estimators": ml.get("n_estimators"),
            })
        else:
            st.error("Not loaded")

        st.markdown("**Label Encoder**")
        le = d.get("label_encoder", {})
        if le.get("status") == "loaded":
            st.success(f"Loaded — classes: {le.get('classes', [])}")
        else:
            st.warning("Not loaded")

        st.markdown("**Feature Schema**")
        schema = d.get("feature_schema")
        if schema:
            st.info(f"{len(schema)} columns")
            with st.expander("View schema"):
                st.json(schema)
        else:
            st.warning("No schema loaded")

    with d2:
        st.markdown("**OCR Engine**")
        if d.get("ocr_engine") == "ready":
            st.success("Ready")
        else:
            st.warning("Unavailable — PaddleOCR not installed or failed to init")

        st.markdown("**Groq LLM**")

        groq_sdk_available = True
        try:
            import groq
        except ImportError:
            groq_sdk_available = False

        api_key_detected = d.get("groq_api_key_detected", False)
        groq_client_initialized = d.get("groq_client", False)
        llm_enabled = groq_client_initialized

        st.markdown("**Status**")
        st.json({
            "Groq SDK Installed": "Yes" if groq_sdk_available else "No",
            "API Key Detected": "Yes" if api_key_detected else "No",
            "Groq Client Initialized": "Yes" if groq_client_initialized else "No",
            "LLM Enabled": "Yes" if llm_enabled else "No",
        })

        st.markdown("**Environment**")
        st.json({
            "BASE_DIR": d.get("base_dir", ""),
            "python_version": sys.version.split()[0],
        })

    st.divider()
    st.markdown("**PATHS**")
    paths = d.get("paths", {})
    st.json(paths)


st.title("  AI Lab Risk Awareness System")
st.caption("Clinical Decision-Support Only — Not a replacement for diagnosis")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⌨️ Text Input",
    "📄 PDF Upload",
    "🖼️ Image Upload",
    "📋 Patient History",
    "🔧 System Debug",
])

with tab1:
    render_text_input_tab()

with tab2:
    render_pdf_tab()

with tab3:
    render_image_tab()

with tab4:
    render_patient_history_tab()

with tab5:
    render_debug_tab()
