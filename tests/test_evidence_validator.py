import pytest

from app.assessment.evidence_validator import validate_evidence
from app.models.dataset import TargetConfig
from app.utils import parse_evidence_metadata


TARGET = TargetConfig(
    release="R2.4",
    commit="b7a91f2",
    prompt="p7",
    environment="production",
    service="atlas-assistant",
)


def evidence(**overrides: str) -> str:
    values = {
        "release": "R2.4",
        "commit": "b7a91f2",
        "service": "atlas-assistant",
        "environment": "production",
        "prompt": "p7",
    }
    values.update(overrides)
    return "; ".join(f"{key}: {value}" for key, value in values.items())


@pytest.mark.parametrize(
    ("dimension", "value"),
    [
        ("release", "R2.3"),
        ("commit", "wrong-sha"),
        ("environment", "staging"),
        ("service", "support-widget"),
        ("prompt", "p6"),
    ],
)
def test_rejects_mismatched_target_dimensions(dimension: str, value: str) -> None:
    result = validate_evidence(
        issue_key="TYN-1",
        comment_id="1001",
        text=evidence(**{dimension: value}),
        target=TARGET,
        acceptance_criteria="The deployed prompt p7 must pass.",
    )
    assert not result["applicable"]
    assert result["mismatched_dimensions"][dimension]["actual"] == value


def test_accepts_matching_evidence_with_required_prompt() -> None:
    result = validate_evidence(
        issue_key="TYN-1",
        comment_id="1001",
        text=evidence(),
        target=TARGET,
        acceptance_criteria="The quality evaluation matches the deployed prompt p7.",
    )
    assert result["applicable"]
    assert result["missing_dimensions"] == []


def test_missing_verification_metadata_is_not_applicable() -> None:
    result = validate_evidence(
        issue_key="TYN-1",
        comment_id="1001",
        text="QA will run the test tomorrow.",
        target=TARGET,
        acceptance_criteria="QA verifies the target release.",
    )
    assert not result["applicable"]
    assert set(result["missing_dimensions"]) == {
        "release",
        "commit",
        "service",
        "environment",
    }


def test_prompt_is_required_only_when_criterion_mentions_it() -> None:
    result = validate_evidence(
        issue_key="TYN-1",
        comment_id="1001",
        text=evidence().replace("; prompt: p7", ""),
        target=TARGET,
        acceptance_criteria="The security owner approves production.",
    )
    assert result["applicable"]


def test_related_prompt_bundle_metadata_rejects_wrong_evaluation_version() -> None:
    result = validate_evidence(
        issue_key="TYN-1",
        comment_id="1002",
        text=(
            "Recorded event 2026-11-08T14:00:00+05:30 | actor Priya Nair | role QA Lead\n"
            "EVAL-55 completed: release R2.4; commit b7a91f2; service atlas-assistant; "
            "environment production. Accuracy 96.2 percent across 500 graded questions."
        ),
        target=TARGET,
        acceptance_criteria="The deployed prompt quality evaluation must meet its threshold.",
        related_text="Evaluation EVAL-55 harness configured with prompt bundle p6.",
    )

    assert not result["applicable"]
    assert result["mismatched_dimensions"]["prompt"] == {
        "expected": "p7",
        "actual": "p6",
    }


def test_extracts_target_dimensions_from_json_issue_description() -> None:
    description = (
        'Requirement ID: REQ-10\n'
        'Target: {"release": "R2.4", "commit": "b7a91f2", "prompt": "p7", '
        '"environment": "production", "service": "atlas-assistant"}'
    )

    assert parse_evidence_metadata(description) == {
        "release": "R2.4",
        "commit": "b7a91f2",
        "prompt": "p7",
        "environment": "production",
        "service": "atlas-assistant",
    }


def test_release_checklist_prose_is_not_parsed_as_a_release_value() -> None:
    assert parse_evidence_metadata("Tracking for the release checklist.") == {}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Prompt Source: imported metadata", {}),
        ("Prompt version: p7", {"prompt": "p7"}),
        ("prompt bundle p6", {"prompt": "p6"}),
        ("Ready for release.", {}),
    ],
)
def test_prompt_and_release_metadata_only_accept_actual_values(
    text: str, expected: dict[str, str]
) -> None:
    assert parse_evidence_metadata(text) == expected


def test_qa_event_narrative_extracts_each_target_dimension() -> None:
    text = (
        "Recorded event 2026-11-08T14:00:00+05:30 | actor Priya Nair | role QA Lead\n"
        "PARSE-622 supersedes PARSE-601. Retest on release R2.4; commit b7a91f2; "
        "service atlas-assistant; environment production: 60 of 60 mandatory parsing cases pass."
    )
    result = validate_evidence(
        issue_key="SCRUM-13",
        comment_id="10044",
        text=text,
        target=TARGET,
        acceptance_criteria="The target build must pass all 60 mandatory document parsing cases.",
    )

    assert result["applicable"]
    assert result["metadata_found"] == {
        "release": "R2.4",
        "commit": "b7a91f2",
        "service": "atlas-assistant",
        "environment": "production",
    }
