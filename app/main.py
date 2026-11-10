import logging
import sys

from fastapi import FastAPI

from app.api.routes import create_router
from app.config import get_settings
from app.services.audit_service import AuditStore
from app.utils import JsonLogFormatter


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonLogFormatter())
root_logger = logging.getLogger()
if not root_logger.handlers:
    root_logger.addHandler(handler)
for existing_handler in root_logger.handlers:
    existing_handler.setFormatter(JsonLogFormatter())
root_logger.setLevel(logging.INFO)

settings = get_settings()
audit_store = AuditStore(settings.database_path)
app = FastAPI(
    title="Jira Release Evidence Agent",
    version="1.0.0",
    description="Evidence-backed software release assessments using Jira Cloud records.",
)
app.include_router(create_router(settings, audit_store))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
