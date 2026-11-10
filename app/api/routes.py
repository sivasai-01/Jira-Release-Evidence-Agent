import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Response, UploadFile
from pydantic import ValidationError

from app.assessment.report_generator import render_markdown
from app.config import Settings
from app.jira.client import JiraClient
from app.jira.importer import DatasetImporter
from app.models.api import (
    AssessmentAccepted,
    AssessmentRequest,
    AssessmentStatus,
    ImportResponse,
)
from app.models.dataset import load_dataset_bytes
from app.models.report import AssessmentReport
from app.services.audit_service import AuditStore
from app.services.execution_service import (
    report_with_usage_summary,
    run_assessment_background,
)


logger = logging.getLogger(__name__)


def create_router(settings: Settings, audit: AuditStore) -> APIRouter:
    router = APIRouter()

    @router.post("/api/import", response_model=ImportResponse)
    def import_dataset(
        file: UploadFile = File(..., description="Dataset JSON file to import into Jira."),
        project_key: str | None = Form(default=None),
    ) -> ImportResponse:
        settings.require_jira()
        if file.filename and not file.filename.casefold().endswith(".json"):
            raise HTTPException(status_code=400, detail="Upload a .json dataset file.")
        content = file.file.read(settings.max_dataset_upload_bytes + 1)
        if len(content) > settings.max_dataset_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Dataset upload exceeds the configured limit of "
                    f"{settings.max_dataset_upload_bytes} bytes."
                ),
            )
        try:
            dataset = load_dataset_bytes(content)
        except (ValidationError, ValueError) as error:
            logger.info("Rejected invalid dataset upload: %s", error)
            raise HTTPException(status_code=422, detail="Upload must contain a valid dataset JSON.") from error
        selected_project_key = project_key or settings.jira_project_key
        if not selected_project_key:
            raise HTTPException(
                status_code=422,
                detail="project_key is required unless JIRA_PROJECT_KEY is configured.",
            )
        try:
            client = JiraClient(
                settings.jira_base_url,
                settings.jira_email,
                settings.jira_api_token.get_secret_value(),
                settings.jira_max_retries,
            )
            try:
                importer = DatasetImporter(
                    client,
                    selected_project_key,
                    dataset,
                    project_name=settings.jira_project_name,
                    project_lead_account_id=settings.jira_project_lead_account_id,
                    link_type=settings.jira_link_type,
                    source_id_field=settings.jira_source_id_field,
                )
                result = importer.run()
            finally:
                client.close()
            settings.reports_dir.mkdir(parents=True, exist_ok=True)
            map_path = settings.reports_dir / "source-issue-map.json"
            map_path.write_text(
                json.dumps(
                    {
                        "project_key": result.project_key,
                        "dataset_label": result.dataset_label,
                        "source_id_to_issue_key": result.source_id_to_issue_key,
                        "errors": result.errors,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except (ValueError, RuntimeError) as error:
            logger.exception("Dataset import failed")
            raise HTTPException(status_code=502, detail=str(error)) from error
        return ImportResponse(**result.__dict__)

    @router.post("/api/assessment/run", response_model=AssessmentAccepted, status_code=202)
    def start_assessment(
        request: AssessmentRequest, background_tasks: BackgroundTasks
    ) -> AssessmentAccepted:
        settings.require_jira()
        project_key = request.project_key or settings.jira_project_key
        dataset_label = request.dataset_label or settings.jira_dataset_label
        assessment_id = str(uuid.uuid4())
        audit.create_assessment(assessment_id)
        background_tasks.add_task(
            run_assessment_background,
            assessment_id=assessment_id,
            project_key=project_key,
            dataset_label=dataset_label,
            target_release=request.target_release,
            requested_target=request.target,
            settings=settings,
            audit=audit,
        )
        return AssessmentAccepted(assessment_id=assessment_id, status="queued")

    @router.get(
        "/api/assessment/{assessment_id}", response_model=AssessmentStatus
    )
    def get_assessment(assessment_id: str) -> AssessmentStatus:
        result = audit.get_assessment(assessment_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Assessment not found")
        if result["report"] is not None:
            result["report"] = report_with_usage_summary(result["report"])
        return AssessmentStatus(**result)

    @router.get("/api/assessment/{assessment_id}/report")
    def get_report(assessment_id: str, format: str = "json") -> Any:
        result = audit.get_assessment(assessment_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Assessment not found")
        if result["status"] != "completed":
            raise HTTPException(
                status_code=409,
                detail=f"Assessment is {result['status']}; report is not available.",
            )
        if result["report"] is None:
            raise HTTPException(status_code=404, detail="Assessment report not found")
        report_data = report_with_usage_summary(result["report"])
        report_path = settings.reports_dir / f"{assessment_id}.md"
        if format.casefold() == "markdown":
            if not report_path.is_file():
                raise HTTPException(status_code=404, detail="Markdown report not found")
            return Response(
                content=render_markdown(AssessmentReport.model_validate(report_data)),
                media_type="text/markdown",
            )
        if format.casefold() != "json":
            raise HTTPException(status_code=400, detail="format must be 'json' or 'markdown'")
        return report_data

    return router
