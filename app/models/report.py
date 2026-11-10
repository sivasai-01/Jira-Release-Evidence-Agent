from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.models.dataset import TargetConfig


class RequirementDecision(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ReleaseDecision(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class EvidenceReference(BaseModel):
    issue_key: str
    comment_id: str
    source_id: str | None = None
    actor: str | None = None
    role: str | None = None
    recorded_at: str | None = None
    quote: str = ""


class RequirementResult(BaseModel):
    requirement_id: str
    summary: str
    mandatory: bool
    acceptance_criteria: str
    decision: RequirementDecision
    explanation: str
    llm_usage: dict[str, Any] = Field(default_factory=dict)
    supporting_evidence: list[EvidenceReference] = Field(default_factory=list)
    applicability: list[dict[str, Any]] = Field(default_factory=list)
    rejected_evidence: list[dict[str, Any]] = Field(default_factory=list)
    outstanding_verification: list[str] = Field(default_factory=list)
    clarification_required: list[str] = Field(default_factory=list)


class AssessmentReport(BaseModel):
    assessment_id: str
    dataset_label: str
    project_key: str
    target: TargetConfig
    overall_decision: ReleaseDecision
    total_requirements: int
    pass_count: int
    fail_count: int
    insufficient_evidence_count: int
    requirements: list[RequirementResult]
    investigation_trace: list[dict[str, Any]]
    execution_time_seconds: float
    llm_model: str
    llm_usage: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    generated_at: str
