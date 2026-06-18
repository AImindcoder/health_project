from hackthathon import load_patient_registry, save_visit, get_visit_history
import pandas as pd


def list_patients() -> pd.DataFrame:
    df = load_patient_registry()
    if not df.empty and "age" in df.columns:
        df["age"] = df["age"].astype(str)
    return df


def get_patient(patient_id: str) -> dict | None:
    df = load_patient_registry()
    row = df[df["patient_id"] == patient_id]
    if row.empty:
        return None
    result = row.iloc[0].to_dict()
    for k in ("age", "name", "gender", "patient_id"):
        if k in result:
            result[k] = str(result[k])
    return result


def create_patient(patient_id: str, name: str = "", age: str = "", gender: str = "") -> dict:
    _upsert_patient_direct(patient_id, name, age, gender)
    return get_patient(patient_id)


def _upsert_patient_direct(patient_id: str, name: str, age: str, gender: str):
    """Direct DataFrame upsert that avoids hackthathon's int64 dtype issue."""
    from hackthathon import PATHS, save_patient_registry
    import pandas as _pd
    from datetime import datetime

    p = PATHS["patient_registry"]
    if p.exists():
        df = _pd.read_csv(p, dtype={"age": str, "name": str, "gender": str, "patient_id": str})
    else:
        df = _pd.DataFrame(columns=["patient_id", "name", "age", "gender", "first_visit", "last_visit", "visit_count"])

    now = datetime.now().strftime("%Y-%m-%d")
    mask = df["patient_id"] == patient_id

    if mask.any():
        df.loc[mask, "last_visit"] = now
        df.loc[mask, "visit_count"] = df.loc[mask, "visit_count"].fillna(0).astype(int) + 1
        if name:
            df.loc[mask, "name"] = name
        if age:
            df.loc[mask, "age"] = age
        if gender:
            df.loc[mask, "gender"] = gender
    else:
        new_row = _pd.DataFrame([{
            "patient_id": patient_id,
            "name": name or "Unknown",
            "age": age or "Unknown",
            "gender": gender or "Unknown",
            "first_visit": now,
            "last_visit": now,
            "visit_count": 1,
        }])
        df = _pd.concat([df, new_row], ignore_index=True)

    save_patient_registry(df)


def update_patient(patient_id: str, name: str | None = None, age: str | None = None, gender: str | None = None) -> dict | None:
    existing = get_patient(patient_id)
    if existing is None:
        return None
    new_name = str(name) if name is not None else str(existing.get("name", ""))
    new_age = str(age) if age is not None else str(existing.get("age", ""))
    new_gender = str(gender) if gender is not None else str(existing.get("gender", ""))
    _upsert_patient_direct(patient_id, new_name, new_age, new_gender)
    return get_patient(patient_id)


def delete_patient(patient_id: str) -> bool:
    df = load_patient_registry()
    before = len(df)
    df = df[df["patient_id"] != patient_id]
    if len(df) == before:
        return False
    from hackthathon import save_patient_registry
    save_patient_registry(df)
    return True


def get_visits(patient_id: str) -> pd.DataFrame:
    return get_visit_history(patient_id)


def save_visit_record(patient_id: str, labs: dict, risk_score: float, risk_level: str):
    save_visit(patient_id, labs, risk_score, risk_level)
