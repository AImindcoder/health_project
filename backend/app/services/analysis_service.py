from hackthathon import (
    _run_ml,
    ensemble_score,
    explain_risk,
    groq_client,
    get_llm_explanation,
)


def compute_ml_score(feature_df):
    return _run_ml(feature_df)


def compute_ensemble(ml_score: float, rule_score: float, trend_score: float, ml_is_dummy: bool) -> dict:
    return ensemble_score(ml_score, rule_score, trend_score, ml_is_dummy)


def explain(labs: dict, interp: dict, rule_result: dict, score_result: dict) -> dict:
    return explain_risk(labs, interp, rule_result, score_result)


def llm_explain(labs: dict, interp: dict, diseases: list, score_result: dict, explain_result: dict, lang: str) -> str:
    return get_llm_explanation(labs, interp, diseases, score_result, explain_result, lang)


def run_full_analysis(lab_text: str, symptoms: str, patient_id: str | None, lang_code: str) -> dict | None:
    from hackthathon import (
        parse_labs, interpret_labs, build_feature_vector,
        compute_rule_score, _safe_trends, map_diseases,
    )
    from .patient_service import save_visit_record

    combined = (lab_text + " " + symptoms).strip()
    if not combined:
        return None

    labs = parse_labs(combined) or {}
    interp = interpret_labs(labs) or {}

    feature_df = build_feature_vector(labs)
    feature_df = feature_df.fillna(0).astype(float)

    ml_score, ml_dummy = _run_ml(feature_df)

    rule_result = compute_rule_score(labs, combined)

    trend_result = _safe_trends(labs, patient_id)
    trend_score = trend_result.get("trend_score", 0)

    score_result = ensemble_score(ml_score, rule_result["score"], trend_score, ml_dummy)

    diseases = map_diseases(labs)

    explain_result = explain_risk(labs, interp, rule_result, score_result)

    llm_output = get_llm_explanation(
        labs, interp, diseases, score_result, explain_result, lang_code
    )

    if patient_id:
        from hackthathon import register_patient
        register_patient(patient_id)
        save_visit_record(patient_id, labs, score_result["final_score"], score_result["level"])

    return {
        "labs": labs,
        "interp": interp,
        "diseases": diseases,
        "score_result": score_result,
        "rule_result": rule_result,
        "trend_result": trend_result,
        "explain_result": explain_result,
        "llm_output": llm_output,
        "patient_id": patient_id,
    }
