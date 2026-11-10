from typing import Any

from app.models.dataset import TargetConfig
from app.utils import parse_evidence_metadata, parse_recorded_event


def _normalized(value: str) -> str:
    normalized = value.strip().casefold()
    return {"prod": "production", "preprod": "staging"}.get(normalized, normalized)


def validate_evidence(
    *,
    issue_key: str,
    comment_id: str,
    text: str,
    target: TargetConfig,
    acceptance_criteria: str,
    source_id: str | None = None,
    related_text: str = "",
) -> dict[str, Any]:
    event = parse_recorded_event(text)
    metadata = parse_evidence_metadata(text)
    if related_text:
        related_metadata = parse_evidence_metadata(related_text)
        for dimension, value in related_metadata.items():
            metadata.setdefault(dimension, value)
    expected = target.model_dump(exclude_none=True)
    required = ["release", "commit", "service", "environment"]
    if "prompt" in acceptance_criteria.casefold():
        required.append("prompt")
    missing: list[str] = []
    mismatches: dict[str, dict[str, str]] = {}
    for dimension, expected_value in expected.items():
        actual = metadata.get(dimension)
        if not actual:
            if dimension in required:
                missing.append(dimension)
            continue
        if _normalized(actual) != _normalized(expected_value):
            mismatches[dimension] = {"expected": expected_value, "actual": actual}
    for dimension in required:
        if dimension in expected and dimension not in metadata and dimension not in missing:
            missing.append(dimension)
    return {
        "issue_key": issue_key,
        "comment_id": comment_id,
        "source_id": source_id,
        **event,
        "applicable": not missing and not mismatches,
        "metadata_found": metadata,
        "missing_dimensions": sorted(set(missing)),
        "mismatched_dimensions": mismatches,
        "rejection_reasons": [
            *[f"missing {dimension}" for dimension in sorted(set(missing))],
            *[
                f"{dimension} mismatch: expected {values['expected']}, found {values['actual']}"
                for dimension, values in sorted(mismatches.items())
            ],
        ],
    }
