from pydantic import BaseModel, Field


class PatientCreate(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=50)
    name: str = ""
    age: str = ""
    gender: str = ""


class PatientUpdate(BaseModel):
    name: str | None = None
    age: str | None = None
    gender: str | None = None


class PatientResponse(BaseModel):
    patient_id: str
    name: str = ""
    age: str = ""
    gender: str = ""
    first_visit: str = ""
    last_visit: str = ""
    visit_count: int = 0
