import os, sys, io, contextlib

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dev_dir = os.path.join(backend_dir, "..")

sys.path.insert(0, dev_dir)
sys.path.insert(0, backend_dir)

os.environ["HEALTH_PROJECT_DIR"] = backend_dir

with contextlib.redirect_stdout(io.StringIO()):
    import hackthathon

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["data"]["status"] == "healthy"


def test_text_analysis():
    resp = client.post(
        "/api/analysis/text",
        json={"lab_text": "Hemoglobin: 13.2, Glucose: 185, HbA1c: 7.2%", "symptoms": "fatigue"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["data"]["risk_score"] is not None
    assert data["data"]["risk_level"] in ("LOW", "MODERATE", "HIGH")
    assert len(data["data"]["labs"]) > 0


def test_text_analysis_empty():
    resp = client.post(
        "/api/analysis/text",
        json={"lab_text": "", "symptoms": ""},
    )
    assert resp.status_code == 422


def test_create_patient():
    resp = client.post(
        "/api/patients",
        json={"patient_id": "TEST001", "name": "Test", "age": "40", "gender": "F"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["data"]["patient_id"] == "TEST001"


def test_get_patient():
    resp = client.get("/api/patients/TEST001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["data"]["name"] == "Test"


def test_list_patients():
    resp = client.get("/api/patients")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert len(data["data"]["patients"]) >= 1


def test_update_patient():
    resp = client.put(
        "/api/patients/TEST001",
        json={"name": "Updated Name", "age": "41"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["data"]["name"] == "Updated Name"
    assert data["data"]["age"] == "41"


def test_delete_patient():
    resp = client.delete("/api/patients/TEST001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True


def test_get_unknown_patient():
    resp = client.get("/api/patients/DOES_NOT_EXIST")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
