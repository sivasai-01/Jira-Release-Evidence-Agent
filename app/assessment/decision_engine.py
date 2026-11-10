import re
from typing import Any

from app.models.report import (
    EvidenceReference,
    RequirementDecision,
    RequirementResult,
)

EVIDENCE_IDS = re.compile(r"\b[A-Z][A-Z0-9_-]*-\d+\b")
SUPERSESSION = re.compile(
    r"\b([A-Z][A-Z0-9_-]*-\d+)\s+(?:explicitly\s+)?supersed(?:es|ed)\s+"
    r"(?:the\s+)?([A-Z][A-Z0-9_-]*-\d+)\b",
    re.IGNORECASE,
)
FAILURE_MARKERS = re.compile(
    r"\b(?:failed|failure|fail|exposed|exposure|leaked|leakage|duplicate(?:d)?|"
    r"missing final|exceeded|drop(?:s|ped|ping)?)\b",
    re.IGNORECASE,
)
PASS_MARKERS = re.compile(
    r"\b(?:passed|pass|met the threshold|criteria satisfied|no failures?|"
    r"zero failures?|approved)\b",
    re.IGNORECASE,
)
NON_FAILURE_CONTEXT = re.compile(
    r"\b(?:no failures?|zero failures?|without failures?|no exposure|no duplicates?)\b",
    re.IGNORECASE,
)
INCONCLUSIVE_MARKERS = re.compile(
    r"\b(?:aborted|rerun scheduled|retest scheduled|pending|inconclusive|"
    r"not run|not completed)\b",
    re.IGNORECASE,
)
RESULT_COUNTS = re.compile(
    r"\b(\d+)\s*(?:/|of)\s*(\d+)\b",
    re.IGNORECASE,
)
NUMERIC_LIMIT = re.compile(
    r"\b(at most|no more than|not exceed|less than|below|under|"
    r"at least|no less than|greater than|above|exceed|<=|>=|<|>)\s*"
    r"(USD\s*)?(\d+(?:\.\d+)?)\s*(USD|percent|%|seconds?|ms)?",
    re.IGNORECASE,
)
MEASURED_VALUE = re.compile(
    r"(?P<prefix>USD\s*)?(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<suffix>USD|percent|%|seconds?|ms)?",
    re.IGNORECASE,
)
CROSS_CUSTOMER_ACCESS = re.compile(
    r"\bcustomer\s+([A-Z0-9_-]+)\s+(?:received|accessed|retrieved)\s+"
    r"(?:the\s+)?(?:document|record|file)\s+([A-Z0-9_-]+)"
    r".{0,100}\bbelongs to customer\s+([A-Z0-9_-]+)\b",
    re.IGNORECASE,
)


def _role_is_verifier(role: str | None) -> bool:
    if not role:
        return False
    return bool(
        re.search(
            r"\b(?:qa|quality|security owner|performance|sre|reliability|"
            r"accessibility|test engineer|incident commander|release manager|"
            r"support operations)\b",
            role,
            re.IGNORECASE,
        )
    )


def _numeric_criterion_passes(
    text: str, acceptance_criteria: str
) -> bool | None:
    limit = NUMERIC_LIMIT.search(acceptance_criteria)
    if not limit:
        return None
    operator = limit.group(1).casefold()
    required_unit = (
        "usd"
        if limit.group(2) or (limit.group(4) and limit.group(4).casefold() == "usd")
        else (limit.group(4) or "").casefold()
    )
    threshold = float(limit.group(3))
    for measured in MEASURED_VALUE.finditer(text):
        measured_unit = (
            "usd"
            if measured.group("prefix")
            or (
                measured.group("suffix")
                and measured.group("suffix").casefold() == "usd"
            )
            else (measured.group("suffix") or "").casefold()
        )
        if required_unit and measured_unit != required_unit:
            continue
        value = float(measured.group("value"))
        if operator in {"at most", "no more than", "not exceed", "<=", "below", "under", "less than", "<"}:
            return value <= threshold if operator in {"at most", "no more than", "not exceed", "<="} else value < threshold
        if operator in {"at least", "no less than", ">="}:
            return value >= threshold
        if operator in {"greater than", "above", "exceed", ">"}:
            return value > threshold
    return None


def _is_confirmed_failure(
    text: str, role: str | None, acceptance_criteria: str = ""
) -> bool:
    if not _role_is_verifier(role):
        return False
    numeric_result = _numeric_criterion_passes(text, acceptance_criteria)
    if numeric_result is False:
        return True
    if numeric_result is True:
        return False
    if any(
        int(passed) < int(total)
        for passed, total in RESULT_COUNTS.findall(text)
        if int(total) > 0
    ):
        return True
    if any(
        match.group(1).casefold() != match.group(3).casefold()
        for match in CROSS_CUSTOMER_ACCESS.finditer(text)
    ):
        return True
    open_defect = bool(
        re.search(
            r"\b(?:confirmed|reopened|reopening|open|unclosed)\b.{0,80}\bdefect\b"
            r"|\bdefect\b.{0,80}\b(?:confirmed|reopened|reopening|open|unclosed)\b",
            text,
            re.IGNORECASE,
        )
    )
    if open_defect:
        return True
    if NON_FAILURE_CONTEXT.search(text):
        return False
    return bool(FAILURE_MARKERS.search(text))


def _is_inconclusive(text: str) -> bool:
    return bool(INCONCLUSIVE_MARKERS.search(text))


def _is_confirmed_pass(
    text: str, role: str | None, acceptance_criteria: str = ""
) -> bool:
    if not _role_is_verifier(role) or _is_inconclusive(text):
        return False
    numeric_result = _numeric_criterion_passes(text, acceptance_criteria)
    if numeric_result is not None:
        return numeric_result
    counts = RESULT_COUNTS.findall(text)
    if counts and any(int(passed) < int(total) for passed, total in counts):
        return False
    has_failure = bool(FAILURE_MARKERS.search(text)) and not bool(
        NON_FAILURE_CONTEXT.search(text)
    )
    return bool(PASS_MARKERS.search(text)) and not has_failure


def _required_security_owner(acceptance_criteria: str) -> str | None:
    match = re.search(
        r"\bsecurity owner\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,2}?)"
        r"(?=\s+(?:must|should|shall|to|is|will)\b|[,.;:]|$)",
        acceptance_criteria,
        re.IGNORECASE,
    )
    return match.group(1).casefold() if match else None


def _active_evidence(
    validated_evidence: list[dict[str, Any]],
    retrieved_comments: dict[tuple[str, str], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_reference = {
        (item["issue_key"], str(item["comment_id"])): item
        for item in validated_evidence
    }
    superseded_by: dict[str, set[str]] = {}
    for (issue_key, comment_id), text in retrieved_comments.items():
        validation = by_reference.get((issue_key, str(comment_id)))
        if validation and validation.get("applicable"):
            for newer_id, older_id in SUPERSESSION.findall(text):
                superseded_by.setdefault(older_id.casefold(), set()).add(
                    newer_id.casefold()
                )
    active: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for validation in validated_evidence:
        key = (validation["issue_key"], str(validation["comment_id"]))
        text = retrieved_comments.get(key, "")
        evidence_ids = {item.casefold() for item in EVIDENCE_IDS.findall(text)}
        is_superseded = any(
            old_id in evidence_ids and not (newer_ids & evidence_ids)
            for old_id, newer_ids in superseded_by.items()
        )
        detail = dict(validation)
        reasons = list(detail.get("rejection_reasons", []))
        if is_superseded:
            reasons.append("superseded by a later Jira evidence record")
            detail["rejection_reasons"] = reasons
            detail["superseded"] = True
            rejected.append(detail)
        elif not detail.get("applicable"):
            rejected.append(detail)
        else:
            active.append(detail)
    return active, rejected


def resolve_requirement(
    *,
    requirement_id: str,
    summary: str,
    mandatory: bool,
    acceptance_criteria: str,
    candidate: dict[str, Any] | None,
    validated_evidence: list[dict[str, Any]],
    retrieved_comments: dict[tuple[str, str], str] | None = None,
    failure_reason: str | None = None,
    investigated_issues: list[dict[str, Any]] | None = None,
) -> RequirementResult:
    comments = retrieved_comments or {}
    active, rejected = _active_evidence(validated_evidence, comments)
    eligible = {
        (item["issue_key"], str(item["comment_id"])): item for item in active
    }
    candidate = candidate or {}
    cited = candidate.get("supporting_evidence", [])
    accepted: list[EvidenceReference] = []
    accepted_keys: set[tuple[str, str]] = set()
    for reference in cited:
        issue_key = str(reference.get("issue_key", ""))
        comment_id = str(reference.get("comment_id", ""))
        validation = eligible.get((issue_key, comment_id))
        if validation:
            quote = str(reference.get("quote", ""))
            actual_comment = comments.get((issue_key, comment_id), "")
            if quote and quote not in actual_comment:
                quote = ""
            accepted.append(
                EvidenceReference(
                    issue_key=issue_key,
                    comment_id=comment_id,
                    source_id=validation.get("source_id"),
                    actor=validation.get("actor"),
                    role=validation.get("role"),
                    recorded_at=validation.get("recorded_at"),
                    quote=quote,
                )
            )
            accepted_keys.add((issue_key, comment_id))

    decision_text = str(candidate.get("decision", "INSUFFICIENT_EVIDENCE"))
    try:
        requested_decision = RequirementDecision(decision_text)
    except ValueError:
        requested_decision = RequirementDecision.INSUFFICIENT_EVIDENCE
    authoritative = [
        item
        for item in active
        if _role_is_verifier(item.get("role"))
    ]
    requires_security_owner = "security owner" in acceptance_criteria.casefold()
    if requires_security_owner:
        required_owner = _required_security_owner(acceptance_criteria)
        authoritative = [
            item
            for item in active
            if (
                required_owner is None
                or (item.get("actor") or "").casefold() == required_owner
            )
            and "security owner" in (item.get("role") or "").casefold()
            and re.search(r"\bapproved?\b", comments.get(
                (item["issue_key"], str(item["comment_id"])), ""
            ), re.IGNORECASE)
        ]

    authoritative_keys = {
        (item["issue_key"], str(item["comment_id"])) for item in authoritative
    }
    for item in active:
        key = (item["issue_key"], str(item["comment_id"]))
        if key in authoritative_keys:
            continue
        reason = (
            f"recorded role {item.get('role') or 'unavailable'} is not an "
            "authorized verifier for this requirement"
        )
        if requires_security_owner:
            reason = (
                "security approval requires the named Security Owner and production scope"
            )
        detail = dict(item)
        detail["applicable"] = False
        detail["rejection_reasons"] = [
            *detail.get("rejection_reasons", []),
            reason,
        ]
        rejected.append(detail)
    accepted = [
        item
        for item in accepted
        if (item.issue_key, item.comment_id) in authoritative_keys
    ]
    accepted_keys = {
        (item.issue_key, item.comment_id) for item in accepted
    }

    confirmed_failures = [
        item
        for item in active
        if _is_confirmed_failure(
            comments.get((item["issue_key"], str(item["comment_id"])), ""),
            item.get("role"),
            acceptance_criteria,
        )
    ]
    authoritative_passes = [
        item
        for item in active
        if _is_confirmed_pass(
            comments.get((item["issue_key"], str(item["comment_id"])), ""),
            item.get("role"),
            acceptance_criteria,
        )
    ]
    conflicting_evidence = [
        (failure, passing)
        for failure in confirmed_failures
        for passing in authoritative_passes
        if (failure["issue_key"], str(failure["comment_id"]))
        != (passing["issue_key"], str(passing["comment_id"]))
    ]
    for item in confirmed_failures:
        key = (item["issue_key"], str(item["comment_id"]))
        if key not in accepted_keys:
            accepted.append(
                EvidenceReference(
                    issue_key=key[0],
                    comment_id=key[1],
                    source_id=item.get("source_id"),
                    actor=item.get("actor"),
                    role=item.get("role"),
                    recorded_at=item.get("recorded_at"),
                    quote=comments.get(key, "")[:500],
                )
            )
            accepted_keys.add(key)

    if conflicting_evidence:
        for item in (conflicting_evidence[0][0], conflicting_evidence[0][1]):
            key = (item["issue_key"], str(item["comment_id"]))
            if key not in accepted_keys:
                accepted.append(
                    EvidenceReference(
                        issue_key=key[0],
                        comment_id=key[1],
                        source_id=item.get("source_id"),
                        actor=item.get("actor"),
                        role=item.get("role"),
                        recorded_at=item.get("recorded_at"),
                        quote=comments.get(key, "")[:500],
                    )
                )
                accepted_keys.add(key)
        requested_decision = RequirementDecision.INSUFFICIENT_EVIDENCE
    elif confirmed_failures:
        requested_decision = RequirementDecision.FAIL
    elif not accepted:
        requested_decision = RequirementDecision.INSUFFICIENT_EVIDENCE
    elif requested_decision == RequirementDecision.PASS:
        cited_authority = [
            item
            for item in authoritative
            if (item["issue_key"], str(item["comment_id"])) in accepted_keys
            and not _is_inconclusive(
                comments.get(
                    (item["issue_key"], str(item["comment_id"])), ""
                )
            )
        ]
        if not cited_authority:
            requested_decision = RequirementDecision.INSUFFICIENT_EVIDENCE
    elif requested_decision == RequirementDecision.FAIL:
        cited_failure = any(
            (reference.issue_key, reference.comment_id) in accepted_keys
            and _is_confirmed_failure(
                comments.get((reference.issue_key, reference.comment_id), ""),
                next(
                    (
                        item.get("role")
                        for item in active
                        if item["issue_key"] == reference.issue_key
                        and str(item["comment_id"]) == reference.comment_id
                    ),
                    None,
                ),
                acceptance_criteria,
            )
            for reference in accepted
        )
        if not cited_failure:
            requested_decision = RequirementDecision.INSUFFICIENT_EVIDENCE

    outstanding = [str(item) for item in candidate.get("outstanding_verification", [])]
    clarifications = [str(item) for item in candidate.get("clarification_required", [])]
    explanation = str(candidate.get("explanation", "")).strip()
    if conflicting_evidence:
        explanation = (
            "Applicable authoritative Jira evidence contains unresolved, conflicting "
            "pass and failure results."
        )
        outstanding.append(
            "Obtain an authoritative retest that explicitly resolves the conflicting results."
        )
    elif failure_reason and not confirmed_failures:
        requested_decision = RequirementDecision.INSUFFICIENT_EVIDENCE
        explanation = f"Assessment could not be completed: {failure_reason}"
        outstanding.append("Retry the Jira-backed investigation after resolving the reported failure.")
    elif failure_reason:
        explanation = (
            "Applicable Jira verification establishes a failure. The agent also "
            f"reported an investigation error: {failure_reason}"
        )
    elif requested_decision == RequirementDecision.INSUFFICIENT_EVIDENCE and not authoritative:
        explanation = (
            "The retrieved evidence does not include an applicable, appropriately "
            "qualified verification or approval."
        )
        outstanding.append(
            "Obtain target-scoped verification from an authorized QA or requirement owner."
        )
    elif requested_decision == RequirementDecision.INSUFFICIENT_EVIDENCE:
        explanation = (
            "Applicable evidence was retrieved, but it does not establish a conclusive "
            "pass or demonstrated failure."
        )
        outstanding.append("Resolve the inconclusive verification result.")
    elif not accepted:
        if not explanation:
            explanation = (
                "No cited comment was validated as applicable to the target release configuration."
            )
        if not outstanding:
            outstanding.append(
                "Gather release-, commit-, service- and environment-specific verification evidence."
            )
    elif requested_decision == RequirementDecision.FAIL:
        explanation = (
            "Applicable Jira verification establishes a failure or confirmed open "
            "target-build defect."
        )
    elif not explanation:
        explanation = "Decision is supported by the validated Jira comments listed below."

    explanation_parts: list[str] = []
    if conflicting_evidence:
        explanation_parts.append(
            "INSUFFICIENT_EVIDENCE: current authoritative Jira evidence conflicts; "
            "no superseding retest resolves the results."
        )
    elif failure_reason and not confirmed_failures:
        explanation_parts.append(
            f"INSUFFICIENT_EVIDENCE: the Jira investigation failed ({failure_reason})."
        )
    elif requested_decision == RequirementDecision.PASS:
        explanation_parts.append(
            f"PASS: applicable authoritative Jira evidence supports the criterion: "
            f"{acceptance_criteria}"
        )
    elif requested_decision == RequirementDecision.FAIL:
        explanation_parts.append(
            f"FAIL: applicable authoritative Jira evidence demonstrates a violation "
            f"of the criterion: {acceptance_criteria}"
        )
    elif active:
        explanation_parts.append(
            "INSUFFICIENT_EVIDENCE: applicable evidence was retrieved, but it does "
            "not conclusively establish that the acceptance criterion is met or violated."
        )
    else:
        explanation_parts.append(
            "INSUFFICIENT_EVIDENCE: no applicable, qualified Jira verification "
            "establishes the acceptance criterion."
        )

    reviewed = investigated_issues or []
    if reviewed:
        issue_descriptions = [
            f"{item.get('issue_key')} "
            f"(source {item.get('source_id') or 'unknown'}): "
            f"{item.get('summary', 'no summary')}"
            for item in reviewed[:12]
        ]
        if len(reviewed) > 12:
            issue_descriptions.append(f"{len(reviewed) - 12} more linked Jira records")
        explanation_parts.append(
            "Investigated Jira records: " + "; ".join(issue_descriptions) + "."
        )

    if accepted:
        explanation_parts.append(
            "Applicable evidence: "
            + "; ".join(
                f"{item.issue_key} comment {item.comment_id} "
                f"(source {item.source_id or 'unknown'}, "
                f"{item.role or 'role unavailable'}, actor "
                f"{item.actor or 'unavailable'}, recorded "
                f"{item.recorded_at or 'time unavailable'})"
                for item in accepted[:8]
            )
            + "."
        )
    if rejected:
        rejection_summaries = []
        for item in rejected[:8]:
            reasons = item.get("rejection_reasons") or ["superseded or inapplicable"]
            rejection_summaries.append(
                f"{item.get('issue_key')} comment {item.get('comment_id')}: "
                + ", ".join(reasons)
            )
        if len(rejected) > 8:
            rejection_summaries.append(
                f"{len(rejected) - 8} more rejected comments are itemized in rejected_evidence"
            )
        explanation_parts.append("Rejected evidence: " + "; ".join(rejection_summaries) + ".")
    if failure_reason and confirmed_failures:
        explanation_parts.append(
            f"The investigation also reported an error: {failure_reason}."
        )
    if outstanding:
        explanation_parts.append(
            "Outstanding verification: " + "; ".join(dict.fromkeys(outstanding)) + "."
        )
    if clarifications:
        explanation_parts.append(
            "Clarification required: " + "; ".join(dict.fromkeys(clarifications)) + "."
        )
    explanation = " ".join(explanation_parts)

    applicable_entries = [
        item for item in validated_evidence if item.get("applicable")
    ]
    return RequirementResult(
        requirement_id=requirement_id,
        summary=summary,
        mandatory=mandatory,
        acceptance_criteria=acceptance_criteria,
        decision=requested_decision,
        explanation=explanation,
        supporting_evidence=accepted,
        applicability=applicable_entries,
        rejected_evidence=rejected,
        outstanding_verification=outstanding,
        clarification_required=clarifications,
    )


def overall_decision(requirements: list[RequirementResult]) -> str:
    if not requirements:
        return "NEEDS_REVIEW"
    mandatory = [item for item in requirements if item.mandatory]
    if any(item.decision == RequirementDecision.FAIL for item in mandatory):
        return "BLOCKED"
    if any(
        item.decision == RequirementDecision.INSUFFICIENT_EVIDENCE
        for item in mandatory
    ):
        return "NEEDS_REVIEW"
    return "READY"
