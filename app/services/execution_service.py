import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.graph import create_model, investigate_requirement
from app.agent.tools import InvestigationContext
from app.assessment.decision_engine import overall_decision, resolve_requirement
from app.config import Settings
from app.jira.client import JiraClient
from app.jira.issue_service import IssueService
from app.models.dataset import TargetConfig
from app.models.report import (
    AssessmentReport,
    ReleaseDecision,
    RequirementDecision,
)
from app.services.audit_service import AuditStore
from app.utils import adf_to_text, parse_target


logger = logging.getLogger(__name__)
SOURCE_ID = re.compile(r"^\[([^\]]+)\]")
MANDATORY = re.compile(r"\bMandatory:\s*(yes|no)\b", re.IGNORECASE)
ACCEPTANCE = re.compile(r"\bAcceptance criterion:\s*(.+)", re.IGNORECASE)
REQUIREMENT_ID = re.compile(r"(?im)^\s*Requirement ID:\s*([A-Z0-9_-]+)\s*$")
TARGET_FIELDS = ("release", "commit", "service", "environment", "prompt")
LINK_TERM_STOP_WORDS = {
    "acceptance", "criterion", "criteria", "target", "release", "commit",
    "service", "environment", "prompt", "version", "must", "should", "shall",
    "with", "from", "that", "this", "their", "which", "where", "into", "only",
    "each", "than", "more", "less", "the", "and", "for", "are", "has", "have",
    "will", "can", "may", "production", "all", "its",
}


def _record_discovery(
    audit: AuditStore,
    assessment_id: str,
    tool: str,
    arguments: dict[str, Any],
    summary: str,
    started: float,
) -> None:
    audit.record_tool(
        assessment_id, tool, arguments, summary, (time.perf_counter() - started) * 1000
    )


def _target_from_jira(
    manifest_text: str,
    target_release: str,
    requested_target: TargetConfig | None,
) -> TargetConfig:
    manifest_target = parse_target(manifest_text)
    if not manifest_target:
        raise ValueError("The Jira release manifest does not contain a parseable target.")
    if manifest_target.get("release") != target_release:
        raise ValueError("The requested release differs from the Jira manifest target.")
    missing = [
        dimension
        for dimension in TARGET_FIELDS
        if not manifest_target.get(dimension)
    ]
    if missing:
        raise ValueError(
            "The Jira release manifest target is missing required dimensions: "
            + ", ".join(missing)
        )
    target = TargetConfig.model_validate(manifest_target)
    if requested_target is None:
        return target
    requested_values = requested_target.model_dump()
    incomplete = [
        dimension for dimension in TARGET_FIELDS if not requested_values.get(dimension)
    ]
    if incomplete:
        raise ValueError(
            "The requested target configuration is missing required dimensions: "
            + ", ".join(incomplete)
        )
    mismatches = [
        dimension
        for dimension in TARGET_FIELDS
        if requested_values.get(dimension) != manifest_target.get(dimension)
    ]
    if mismatches:
        raise ValueError(
            "The requested target conflicts with the Jira manifest in: "
            + ", ".join(mismatches)
        )
    return target


def _requirement_details(issue: dict[str, Any]) -> dict[str, Any]:
    fields = issue.get("fields", {})
    summary = str(fields.get("summary", ""))
    match = SOURCE_ID.match(summary)
    if not match:
        raise ValueError(f"Requirement issue {issue.get('key')} has no source ID in summary")
    source_id = match.group(1)
    declared_id = REQUIREMENT_ID.search(
        adf_to_text(fields.get("description"))
    )
    requirement_id = declared_id.group(1) if declared_id else ""
    if not requirement_id:
        raise ValueError(f"Requirement {source_id} has no Requirement ID field")
    if requirement_id != source_id:
        raise ValueError(
            f"Requirement source ID {source_id} conflicts with its declared "
            f"Requirement ID {requirement_id}"
        )
    description = adf_to_text(fields.get("description"))
    mandatory_match = MANDATORY.search(description)
    acceptance_match = ACCEPTANCE.search(description)
    if not mandatory_match:
        raise ValueError(f"Requirement {requirement_id} has no mandatory classification")
    if not acceptance_match:
        raise ValueError(f"Requirement {requirement_id} has no acceptance criterion")
    target = parse_target(description)
    missing_target = [key for key in TARGET_FIELDS if not target.get(key)]
    if missing_target:
        raise ValueError(
            f"Requirement {requirement_id} has an incomplete target: "
            + ", ".join(missing_target)
        )
    return {
        "issue_key": issue["key"],
        "requirement_id": requirement_id,
        "summary": summary.split("] ", 1)[-1],
        "mandatory": mandatory_match.group(1).casefold() == "yes",
        "acceptance_criteria": acceptance_match.group(1).strip(),
        "description": description,
        "target": TargetConfig.model_validate(target).model_dump(),
    }


def discover_release_requirements(
    issue_service: IssueService, max_dataset_records: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_issues = issue_service.search_dataset_issues(max_dataset_records)
    manifests = [
        issue
        for issue in dataset_issues
        if issue_service.get_record_kind(issue) == "manifest"
    ]
    if len(manifests) != 1:
        raise ValueError(
            "Expected exactly one Jira issue with record kind 'manifest' in the "
            f"configured dataset scope; found {len(manifests)}."
        )

    manifest = manifests[0]
    issue_by_key = {issue["key"]: issue for issue in dataset_issues}
    linked_keys = {
        (link.get("outwardIssue") or link.get("inwardIssue") or {}).get("key")
        for link in manifest.get("fields", {}).get("issuelinks", [])
    }
    linked_keys.discard(None)
    missing_links = sorted(key for key in linked_keys if key not in issue_by_key)
    if missing_links:
        raise ValueError(
            "The release manifest has linked issues outside the configured dataset "
            f"scope or unavailable in Jira: {', '.join(missing_links)}"
        )
    non_requirement_links = sorted(
        key
        for key in linked_keys
        if issue_service.get_record_kind(issue_by_key[key]) != "requirement"
    )
    if non_requirement_links:
        raise ValueError(
            "The Jira release manifest links records that are not requirements: "
            + ", ".join(non_requirement_links)
        )
    all_requirements = [
        issue for issue in dataset_issues
        if issue_service.get_record_kind(issue) == "requirement"
    ]
    requirements = [issue_by_key[key] for key in sorted(linked_keys)]
    if len(requirements) != 12:
        raise ValueError(
            f"The Jira release manifest must link exactly 12 requirements; "
            f"found {len(requirements)}."
        )
    if {item["key"] for item in all_requirements} != linked_keys:
        raise ValueError(
            "The manifest links do not exactly match the requirement records in "
            "the configured dataset scope."
        )
    source_ids: set[str] = set()
    for requirement in requirements:
        details = _requirement_details(requirement)
        if details["requirement_id"] in source_ids:
            raise ValueError(
                f"Duplicate requirement identifier {details['requirement_id']} "
                "in the Jira dataset scope."
            )
        source_ids.add(details["requirement_id"])
    return manifest, requirements


def execute_assessment(
    *,
    assessment_id: str,
    project_key: str,
    dataset_label: str,
    target_release: str,
    requested_target: TargetConfig | None,
    settings: Settings,
    audit: AuditStore,
) -> AssessmentReport:
    started = time.perf_counter()
    client = JiraClient(
        settings.jira_base_url,
        settings.jira_email,
        settings.jira_api_token.get_secret_value(),
        settings.jira_max_retries,
    )
    try:
        issue_service = IssueService(client, project_key, dataset_label)
        manifest_start = time.perf_counter()
        discovery_args = {
            "project_key": project_key,
            "dataset_label": dataset_label,
            "record_kind": "manifest",
            "max_dataset_records": settings.max_dataset_records,
        }
        try:
            manifest, requirement_issues = discover_release_requirements(
                issue_service, settings.max_dataset_records
            )
        except Exception as error:
            _record_discovery(
                audit,
                assessment_id,
                "search_jira_issues",
                discovery_args,
                f"ERROR: {type(error).__name__}: {error}",
                manifest_start,
            )
            raise
        _record_discovery(
            audit,
            assessment_id,
            "search_jira_issues",
            discovery_args,
            f"Discovered one manifest and {len(requirement_issues)} linked requirements in Jira.",
            manifest_start,
        )
        manifest_text = adf_to_text(manifest.get("fields", {}).get("description"))
        target = _target_from_jira(manifest_text, target_release, requested_target)

        requirements: list[dict[str, Any]] = []
        for requirement_issue in requirement_issues:
            details = _requirement_details(requirement_issue)
            target_mismatches = [
                dimension for dimension in TARGET_FIELDS
                if details["target"].get(dimension) != getattr(target, dimension)
            ]
            if target_mismatches:
                raise ValueError(
                    f"Requirement {details['requirement_id']} target conflicts with "
                    "the Jira release manifest in: " + ", ".join(target_mismatches)
                )
            if not details["mandatory"]:
                raise ValueError(
                    f"Requirement {details['requirement_id']} must be mandatory "
                    "according to the Jira release manifest."
                )
            requirements.append(details)

        settings.require_llm()
        model = create_model(
            settings.llm_provider,
            settings.llm_api_key,
            settings.llm_model,
            settings.max_output_tokens,
            settings.groq_base_url if settings.llm_provider == "groq" else None,
        )
        policy = manifest_text
        model_usage: dict[str, Any] = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }
        total_tool_calls = 0
        assessment_metrics: dict[str, Any] = {
            "llm_calls": 0,
            "llm_retries": 0,
            "llm_prompt_characters": 0,
            "llm_call_details": [],
        }
        requirement_usage: dict[str, dict[str, Any]] = {}
        requirement_results = []

        for requirement in requirements:
            requirement_call_start = len(assessment_metrics["llm_call_details"])
            used_calls = [0]
            call_lock = threading.Lock()
            retrieved_comments: dict[tuple[str, str], str] = {}
            context = InvestigationContext(
                client=client,
                issues=issue_service,
                audit=audit,
                settings=settings,
                assessment_id=assessment_id,
                project_key=project_key,
                dataset_label=dataset_label,
                target=target,
                requirement_id=requirement["requirement_id"],
                acceptance_criteria=requirement["acceptance_criteria"],
                manifest_policy=policy,
                used_calls=used_calls,
                lock=call_lock,
                model_usage=model_usage,
                metrics=assessment_metrics,
                excluded_issue_keys={
                    manifest["key"],
                    *(item["key"] for item in requirement_issues),
                },
                root_issue_key=requirement["issue_key"],
                link_relevance_terms={
                    word
                    for word in re.findall(
                        r"[a-z0-9]+",
                        (
                            requirement["summary"]
                            + " "
                            + requirement["acceptance_criteria"]
                        ).casefold(),
                    )
                    if len(word) >= 4 and word not in LINK_TERM_STOP_WORDS
                },
            )
            context.issue_depth[requirement["issue_key"]] = 0
            retrieved_comments = context.retrieved_comments
            candidate = None
            failure = None
            try:
                candidate = investigate_requirement(context, model, requirement)
            except Exception as error:
                logger.exception(
                    "Agent failed while investigating %s",
                    requirement["requirement_id"],
                )
                audit.record_tool(
                    assessment_id,
                    "agent_reasoning",
                    {"requirement_id": requirement["requirement_id"]},
                    f"ERROR: {type(error).__name__}: {error}",
                    0,
                )
                failure = f"{type(error).__name__}: {error}"
            total_tool_calls += used_calls[0]
            validations = context.validations

            result = resolve_requirement(
                requirement_id=requirement["requirement_id"],
                summary=requirement["summary"],
                mandatory=requirement["mandatory"],
                acceptance_criteria=requirement["acceptance_criteria"],
                candidate=candidate,
                validated_evidence=validations,
                retrieved_comments=retrieved_comments,
                failure_reason=failure,
                investigated_issues=[
                    {
                        "issue_key": key,
                        "source_id": snapshot.get("source_id"),
                        "summary": snapshot.get("summary", ""),
                        "status": snapshot.get("status"),
                    }
                    for key, snapshot in context.issue_snapshots.items()
                ],
            )
            call_details = assessment_metrics["llm_call_details"][
                requirement_call_start:
            ]
            result.llm_usage = {
                **_summarize_llm_calls(call_details),
                "calls_detail": call_details,
            }
            requirement_usage[requirement["requirement_id"]] = result.llm_usage
            requirement_results.append(result)

        pass_count = sum(
            result.decision == RequirementDecision.PASS for result in requirement_results
        )
        fail_count = sum(
            result.decision == RequirementDecision.FAIL for result in requirement_results
        )
        insufficient_count = sum(
            result.decision == RequirementDecision.INSUFFICIENT_EVIDENCE
            for result in requirement_results
        )
        investigation_trace = audit.investigation_trace(assessment_id)
        report = AssessmentReport(
            assessment_id=assessment_id,
            dataset_label=dataset_label,
            project_key=project_key,
            target=target,
            overall_decision=ReleaseDecision(overall_decision(requirement_results)),
            total_requirements=len(requirement_results),
            pass_count=pass_count,
            fail_count=fail_count,
            insufficient_evidence_count=insufficient_count,
            requirements=requirement_results,
            investigation_trace=investigation_trace,
            execution_time_seconds=time.perf_counter() - started,
            llm_model=f"{settings.llm_provider}/{settings.llm_model}",
            llm_usage={
                **_summarize_llm_calls(assessment_metrics["llm_call_details"]),
                "calls": assessment_metrics["llm_call_details"],
                "requirements": requirement_usage,
            },
            metrics={
                "llm_calls": assessment_metrics["llm_calls"],
                "llm_retries": assessment_metrics["llm_retries"],
                "llm_prompt_characters": assessment_metrics[
                    "llm_prompt_characters"
                ],
                "agent_tool_calls": total_tool_calls,
                "max_tool_calls_per_requirement": settings.max_tool_calls,
                "trace_events": len(investigation_trace),
                "jira_api_calls": client.metrics["requests"],
                "jira_retries": client.metrics["retries"],
                "jira_errors": client.metrics["errors"],
            },
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        json_path = settings.reports_dir / f"{assessment_id}.json"
        markdown_path = settings.reports_dir / f"{assessment_id}.md"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        from app.assessment.report_generator import render_markdown

        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        audit.save_report(assessment_id, report.model_dump(mode="json"))
        return report
    finally:
        client.close()


def _summarize_llm_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "calls": len(calls),
        "call_count": len(calls),
        "prompt_characters": sum(
            int(call.get("prompt_characters", 0)) for call in calls
        ),
    }
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        observed = [
            int(call[key])
            for call in calls
            if isinstance(call.get(key), int)
        ]
        result[key] = sum(observed) if observed else None
        result[f"{key}_reported_calls"] = len(observed)
    complete_calls = sum(
        all(
            isinstance(call.get(key), int)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        )
        for call in calls
    )
    result["token_usage_reported_calls"] = complete_calls
    result["token_usage_unavailable_calls"] = len(calls) - complete_calls
    result["token_usage_complete"] = bool(calls) and complete_calls == len(calls)
    result["token_usage_status"] = (
        "complete"
        if result["token_usage_complete"]
        else "partial"
        if complete_calls
        or any(
            result[f"{key}_reported_calls"]
            for key in ("input_tokens", "output_tokens", "total_tokens")
        )
        else "unavailable"
    )
    return result


def report_with_usage_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Rebuild usage aggregates from persisted per-call data, including older reports."""
    updated = dict(report)
    usage = dict(updated.get("llm_usage") or {})
    call_details = usage.get("calls")
    if isinstance(call_details, list):
        usage.update(_summarize_llm_calls(call_details))
        usage["calls"] = call_details
    requirements = []
    for requirement in updated.get("requirements", []):
        updated_requirement = dict(requirement)
        requirement_usage = dict(updated_requirement.get("llm_usage") or {})
        requirement_calls = requirement_usage.get("calls_detail")
        if isinstance(requirement_calls, list):
            requirement_usage.update(_summarize_llm_calls(requirement_calls))
            requirement_usage["calls_detail"] = requirement_calls
            updated_requirement["llm_usage"] = requirement_usage
        requirements.append(updated_requirement)
    if "requirements" in updated:
        updated["requirements"] = requirements
    updated["llm_usage"] = usage
    return updated


def run_assessment_background(
    *,
    assessment_id: str,
    project_key: str,
    dataset_label: str,
    target_release: str,
    requested_target: TargetConfig | None,
    settings: Settings,
    audit: AuditStore,
) -> None:
    audit.set_status(assessment_id, "running")
    try:
        execute_assessment(
            assessment_id=assessment_id,
            project_key=project_key,
            dataset_label=dataset_label,
            target_release=target_release,
            requested_target=requested_target,
            settings=settings,
            audit=audit,
        )
    except Exception as error:
        logger.exception("Assessment %s failed", assessment_id)
        audit.set_status(
            assessment_id,
            "failed",
            error=f"{type(error).__name__}: {error}",
        )
