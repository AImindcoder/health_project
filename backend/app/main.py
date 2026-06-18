# NOTE: No streamlit imports allowed in this file
# NOTE: No streamlit imports allowed in any file this imports
import os
import sys
import io
import contextlib
from pathlib import Path

# Force UTF-8 output on all platforms (fixes emoji/Arabic garbling on Windows)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

backend_dir = Path(__file__).parent.parent
os.environ["HEALTH_PROJECT_DIR"] = str(backend_dir)

dev_dir = backend_dir.parent
if str(dev_dir) not in sys.path:
    sys.path.insert(0, str(dev_dir))

with contextlib.redirect_stdout(io.StringIO()):
    import hackthathon

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.routes.health import router as health_router
from app.api.routes.analysis import router as analysis_router, extract_router
from app.api.routes.patients import router as patients_router

app = FastAPI(
    title="AI Lab Risk Awareness API",
    description="Clinical decision-support backend for lab risk analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(analysis_router)
app.include_router(extract_router)
app.include_router(patients_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Internal server error", "errors": [str(exc)]},
    )


@app.on_event("startup")
async def startup():
    os.makedirs(settings.uploads_dir, exist_ok=True)
