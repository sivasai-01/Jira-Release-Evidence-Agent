import pytest

from app.assessment.decision_engine import overall_decision, resolve_requirement
from app.assessment.evidence_validator import validate_evidence
from app.models.dataset import TargetConfig
from app.models.report import RequirementDecision, RequirementResult


def result(
    decision: str, mandatory: bool = True, validated_evidence: list[dict] | None = None
) -> RequirementResult:
    verification_text = {
        "PASS": "QA verification passed.",
        "FAIL": "QA verification failed: acceptance criteria not met.",
        "INSUFFICIENT_EVIDENCE": "QA investigation is inconclusive.",
    }[decision]
    return resolve_requirement(
        requirement_id="REQ-01",
        summary="Regression tests",
        mandatory=mandatory,
        acceptance_criteria="All tests must pass.",
        candidate={
            "decision": decision,
            "explanation": "Evaluation based on retrieved evidence.",
            "supporting_evidence": [
                {"issue_key": "TYN-1", "comment_id": "10", "quote": "verified"}
            ],
        },
        validated_evidence=validated_evidence
        or [
            {
                "issue_key": "TYN-1",
                "comment_id": "10",
                "applicable": True,
                "actor": "Test Engineer",
                "role": "QA",
            }
        ],
        retrieved_comments={("TYN-1", "10"): verification_text},
    )


def test_all_mandatory_pass_means_ready() -> None:
    assert overall_decision([result("PASS")]) == "READY"


def test_mandatory_failure_blocks_release() -> None:
    assert overall_decision([result("FAIL")]) == "BLOCKED"


def test_mandatory_missing_proof_needs_review() -> None:
    assert overall_decision([result("INSUFFICIENT_EVIDENCE")]) == "NEEDS_REVIEW"


def test_optional_failure_does_not_block() -> None:
    assert overall_decision([result("FAIL", mandatory=False)]) == "READY"


def test_unvalidated_or_unrelated_comment_cannot_support_decision() -> None:
    resolved = result(
        "PASS",
        validated_evidence=[
            {"issue_key": "TYN-2", "comment_id": "11", "applicable": True}
        ],
    )
    assert resolved.decision == RequirementDecision.INSUFFICIENT_EVIDENCE
    assert resolved.supporting_evidence == []


def test_assessment_model_error_is_insufficient_not_pass() -> None:
    resolved = resolve_requirement(
        requirement_id="REQ-01",
        summary="Regression",
        mandatory=True,
        acceptance_criteria="Pass",
        candidate={"decision": "PASS"},
        validated_evidence=[],
        failure_reason="Gemini quota exhausted",
    )
    assert resolved.decision == RequirementDecision.INSUFFICIENT_EVIDENCE
    assert "Gemini quota exhausted" in resolved.explanation


TARGET = TargetConfig(
    release="R2.4",
    commit="b7a91f2",
    prompt="p7",
    environment="production",
    service="atlas-assistant",
)


def validated_comment(comment_id: str, text: str, criterion: str) -> dict:
    return validate_evidence(
        issue_key="TYN-2",
        comment_id=comment_id,
        text=text,
        target=TARGET,
        acceptance_criteria=criterion,
        source_id="REC-1001",
    )


def candidate(decision: str, comment_id: str) -> dict:
    return {
        "decision": decision,
        "explanation": "Based on the retrieved Jira verification.",
        "supporting_evidence": [
            {"issue_key": "TYN-2", "comment_id": comment_id}
        ],
    }


def test_superseding_qa_retest_replaces_prior_failure() -> None:
    criterion = "All mandatory parsing cases must pass."
    old = (
        "Recorded event 2026-11-06T10:00:00+05:30 | actor Priya Nair | role QA Lead\n"
        "PARSE-601: release R2.4; commit b7a91f2; service atlas-assistant; "
        "environment production. 59 of 60 parsing cases pass; rotated-table case fails."
    )
    retest = (
        "Recorded event 2026-11-08T14:00:00+05:30 | actor Priya Nair | role QA Lead\n"
        "PARSE-622 supersedes PARSE-601. Retest on release R2.4; commit b7a91f2; "
        "service atlas-assistant; environment production: 60 of 60 mandatory parsing cases pass."
    )
    comments = {("TYN-2", "601"): old, ("TYN-2", "622"): retest}
    result = resolve_requirement(
        requirement_id="REQ-01",
        summary="Document parsing",
        mandatory=True,
        acceptance_criteria=criterion,
        candidate=candidate("PASS", "622"),
        validated_evidence=[
            validated_comment("601", old, criterion),
            validated_comment("622", retest, criterion),
        ],
        retrieved_comments=comments,
    )

    assert result.decision == RequirementDecision.PASS
    assert [item.comment_id for item in result.supporting_evidence] == ["622"]
    assert result.rejected_evidence[0]["superseded"]


def test_unresolved_authoritative_conflict_requires_review() -> None:
    criterion = "All mandatory parsing cases must pass."
    failed = (
        "Recorded event 2026-11-06T10:00:00+05:30 | actor Priya Nair | role QA Lead\n"
        "Parsing verification: release R2.4; commit b7a91f2; service atlas-assistant; "
        "environment production. 59 of 60 cases pass; one case failed."
    )
    passed = (
        "Recorded event 2026-11-08T14:00:00+05:30 | actor Alex Chen | role QA Lead\n"
        "Parsing verification: release R2.4; commit b7a91f2; service atlas-assistant; "
        "environment production. 60 of 60 cases pass."
    )
    result = resolve_requirement(
        requirement_id="REQ-01",
        summary="Document parsing",
        mandatory=True,
        acceptance_criteria=criterion,
        candidate=candidate("PASS", "622"),
        validated_evidence=[
            validated_comment("601", failed, criterion),
            validated_comment("622", passed, criterion),
        ],
        retrieved_comments={
            ("TYN-2", "601"): failed,
            ("TYN-2", "622"): passed,
        },
    )

    assert result.decision == RequirementDecision.INSUFFICIENT_EVIDENCE
    assert {item.comment_id for item in result.supporting_evidence} == {"601", "622"}
    assert any("conflicting" in item.casefold() for item in result.outstanding_verification)


def test_aborted_target_run_is_inconclusive_not_a_failure() -> None:
    criterion = "Streaming verification must pass."
    text = (
        "Recorded event 2026-10-22T16:14:00+05:30 | actor Priya Nair | role QA Lead\n"
        "Streaming verification run: release R2.4; commit b7a91f2; "
        "service atlas-assistant; environment production; prompt p7. "
        "Run aborted; rerun scheduled."
    )
    result = resolve_requirement(
        requirement_id="REQ-12",
        summary="Streaming integrity",
        mandatory=True,
        acceptance_criteria=criterion,
        candidate=candidate("FAIL", "1"),
        validated_evidence=[validated_comment("1", text, criterion)],
        retrieved_comments={("TYN-2", "1"): text},
    )

    assert result.decision == RequirementDecision.INSUFFICIENT_EVIDENCE


def test_zero_of_twelve_handoff_and_open_defect_establish_failure() -> None:
    criterion = "Escalated conversations must reach the support queue."
    text = (
        "Recorded event 2026-11-08T14:00:00+05:30 | actor Priya Nair | role QA Lead\n"
        "Handoff test: release R2.4; commit b7a91f2; service atlas-assistant; "
        "environment production. 0 of 12 escalated conversations reached the support "
        "queue; the queue adapter drops them. Defect HND-DEF-4 status: open."
    )
    result = resolve_requirement(
        requirement_id="REQ-08",
        summary="Human handoff",
        mandatory=True,
        acceptance_criteria=criterion,
        candidate=candidate("INSUFFICIENT_EVIDENCE", "1"),
        validated_evidence=[validated_comment("1", text, criterion)],
        retrieved_comments={("TYN-2", "1"): text},
    )

    assert result.decision == RequirementDecision.FAIL


def test_cross_customer_document_exposure_establishes_failure() -> None:
    criterion = (
        "Every isolation case must pass; any confirmed cross-customer document "
        "exposure blocks release."
    )
    text = (
        "Recorded event 2026-11-08T11:00:00+05:30 | actor Priya Nair | role QA Lead\n"
        "Isolation retest: release R2.4; commit b7a91f2; service atlas-assistant; "
        "environment production. Customer ALPHA received document BETA-017, "
        "which belongs to customer BETA. Reopening defect ISO-DEF-12."
    )
    result = resolve_requirement(
        requirement_id="REQ-02",
        summary="Customer isolation",
        mandatory=True,
        acceptance_criteria=criterion,
        candidate=candidate("INSUFFICIENT_EVIDENCE", "1"),
        validated_evidence=[validated_comment("1", text, criterion)],
        retrieved_comments={("TYN-2", "1"): text},
    )

    assert result.decision == RequirementDecision.FAIL


def test_developer_only_claim_cannot_establish_pass() -> None:
    criterion = "The target release must pass regression tests."
    text = (
        "Recorded event 2026-11-08T14:00:00+05:30 | actor Dev Shah | role Developer\n"
        "Regression tests passed: release R2.4; commit b7a91f2; "
        "service atlas-assistant; environment production."
    )
    result = resolve_requirement(
        requirement_id="REQ-01",
        summary="Regression tests",
        mandatory=True,
        acceptance_criteria=criterion,
        candidate=candidate("PASS", "1"),
        validated_evidence=[validated_comment("1", text, criterion)],
        retrieved_comments={("TYN-2", "1"): text},
    )

    assert result.decision == RequirementDecision.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize(
    ("actor", "role", "scope", "expected"),
    [
        ("Asha Rao", "Security Owner", "production", RequirementDecision.PASS),
        ("Asha Rao", "Security Owner", "staging", RequirementDecision.INSUFFICIENT_EVIDENCE),
        ("Ravi Kumar", "Security Owner", "production", RequirementDecision.INSUFFICIENT_EVIDENCE),
    ],
)
def test_security_approval_requires_named_owner_and_production_scope(
    actor: str, role: str, scope: str, expected: RequirementDecision
) -> None:
    criterion = (
        "Security Owner Asha Rao must explicitly approve this release and commit for production."
    )
    text = (
        f"Recorded event 2026-11-08T15:00:00+05:30 | actor {actor} | role {role}\n"
        f"Approved: release R2.4, commit b7a91f2, service atlas-assistant. "
        f"Approval scope: {scope}."
    )
    result = resolve_requirement(
        requirement_id="REQ-06",
        summary="Production security approval",
        mandatory=True,
        acceptance_criteria=criterion,
        candidate=candidate("PASS", "1"),
        validated_evidence=[validated_comment("1", text, criterion)],
        retrieved_comments={("TYN-2", "1"): text},
    )

    assert result.decision == expected
