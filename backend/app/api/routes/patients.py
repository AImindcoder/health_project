from fastapi import APIRouter, HTTPException
from app.schemas.patient import PatientCreate, PatientUpdate, PatientResponse
from app.schemas.response import APIResponse
from app.services.patient_service import (
    list_patients, get_patient, create_patient,
    update_patient, delete_patient, get_visits,
)
import pandas as pd

router = APIRouter(prefix="/api/patients", tags=["patients"])


def _patient_to_response(row: dict) -> dict:
    return {
        "patient_id": row.get("patient_id", ""),
        "name": row.get("name", ""),
        "age": row.get("age", ""),
        "gender": row.get("gender", ""),
        "first_visit": str(row.get("first_visit", "")),
        "last_visit": str(row.get("last_visit", "")),
        "visit_count": int(row.get("visit_count", 0)),
    }


@router.post("")
async def patient_create(req: PatientCreate):
    result = create_patient(req.patient_id, req.name, req.age, req.gender)
    if result is None:
        return APIResponse(success=False, message="Failed to create patient", errors=["Unknown error"])
    return APIResponse(message="Patient created", data=_patient_to_response(result))


@router.get("")
async def patient_list():
    df = list_patients()
    patients = [_patient_to_response(row.to_dict()) for _, row in df.iterrows()]
    return APIResponse(message=f"Found {len(patients)} patients", data={"patients": patients, "total": len(patients)})


@router.get("/{patient_id}")
async def patient_get(patient_id: str):
    result = get_patient(patient_id)
    if result is None:
        return APIResponse(success=False, message="Patient not found", errors=[f"No patient with ID {patient_id}"])
    return APIResponse(message="Patient found", data=_patient_to_response(result))


@router.put("/{patient_id}")
async def patient_update(patient_id: str, req: PatientUpdate):
    result = update_patient(patient_id, req.name, req.age, req.gender)
    if result is None:
        return APIResponse(success=False, message="Patient not found", errors=[f"No patient with ID {patient_id}"])
    return APIResponse(message="Patient updated", data=_patient_to_response(result))


@router.delete("/{patient_id}")
async def patient_delete(patient_id: str):
    deleted = delete_patient(patient_id)
    if not deleted:
        return APIResponse(success=False, message="Patient not found", errors=[f"No patient with ID {patient_id}"])
    return APIResponse(message="Patient deleted", data={"patient_id": patient_id})


@router.get("/{patient_id}/visits")
async def patient_visits(patient_id: str):
    df = get_visits(patient_id)
    return APIResponse(
        message=f"Found {len(df)} visits",
        data={
            "patient_id": patient_id,
            "total_visits": len(df),
            "visits": df.fillna("").to_dict(orient="records") if not df.empty else [],
        },
    )
