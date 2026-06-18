from hackthathon import analyze_trends, _safe_trends


def get_trends(labs: dict, patient_id: str | None = None) -> dict:
    return _safe_trends(labs, patient_id)


def analyze(patient_id: str) -> dict:
    return analyze_trends({}, patient_id)
