from pydantic import BaseModel, Field

from app.models.dataset import TargetConfig


class ImportRequest(BaseModel):
    project_key: str | None = None


class ImportResponse(BaseModel):
    project_key: str
    dataset_label: str
    total_records: int
    created: int
    updated: int
    comments_created: int
    links_created: int
    errors: list[str] = Field(default_factory=list)
    source_id_to_issue_key: dict[str, str] = Field(default_factory=dict)


class AssessmentRequest(BaseModel):
    project_key: str | None = None
    dataset_label: str | None = None
    target_release: str
    target: TargetConfig | None = None


class AssessmentAccepted(BaseModel):
    assessment_id: str
    status: str


class AssessmentStatus(BaseModel):
    assessment_id: str
    status: str
    error: str | None = None
    report: dict | None = None
