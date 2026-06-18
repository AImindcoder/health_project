from pydantic import BaseModel, Field


class AnalysisTextRequest(BaseModel):
    lab_text: str = Field(..., min_length=1, description="Raw lab report text")
    patient_id: str | None = Field(None, description="Optional patient identifier")
    symptoms: str = Field("", description="Optional symptom description")
    language: str = Field("en", description="Output language: en, ur, ar")


class AnalysisPDFResponse(BaseModel):
    extracted_text: str
    character_count: int


class AnalysisImageResponse(BaseModel):
    extracted_text: str
    character_count: int
