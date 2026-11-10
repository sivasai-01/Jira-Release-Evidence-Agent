import json
import re
from collections import deque
from pathlib import Path
from typing import Any

from app.assessment.decision_engine import overall_decision, resolve_requirement
from app.assessment.evidence_validator import validate_evidence
from app.models.dataset import TargetConfig
from app.models.report import RequirementDecision


DATASET_PATH = Path(__file__).parents[1] / "inputs" / "dataset.json"
EXPECTED_DECISIONS = {
    "REQ-01": "PASS",
    "REQ-02": "FAIL",
    "REQ-03": "INSUFFICIENT_EVIDENCE",
    "REQ-04": "PASS",
    "REQ-05": "FAIL",
    "REQ-06": "INSUFFICIENT_EVIDENCE",
    "REQ-07": "PASS",
    "REQ-08": "FAIL",
    "REQ-09": "INSUFFICIENT_EVIDENCE",
    "REQ-10": "PASS",
    "REQ-11": "INSUFFICIENT_EVIDENCE",
    "REQ-12": "FAIL",
}
EVIDENCE_MARKERS = {
    "REQ-01": ("REG-804",),
    "REQ-02": ("ISO-318",),
    "REQ-04": ("PERF-827",),
    "REQ-05": ("COST-881",),
    "REQ-07": ("REDACT-901",),
    "REQ-08": ("HND-77",),
    "REQ-10": ("PARSE-622",),
    "REQ-12": ("STREAM-731",),
}


def _scenario_results() -> list[tuple[str, Any]]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    records = {record["source_id"]: record for record in data["records"]}
    target = TargetConfig.model_validate(data["target"])
    results = []

    for requirement in records.values():
        if requirement["kind"] != "requirement":
            continue
        issue_keys = {
            source_id: f"TYN-{source_id}"
            for source_id in records
        }
        queue = deque(requirement["links"])
        visited: set[str] = set()
        comments: dict[tuple[str, str], str] = {}
        validations: list[dict[str, Any]] = []
        while queue:
            source_id = queue.popleft()
            if source_id in visited or source_id not in records:
                continue
            visited.add(source_id)
            record = records[source_id]
            queue.extend(record["links"])
            related_description = record["description"]
            for index, text in enumerate(record["comments"]):
                comment_id = f"{source_id}-{index}"
                issue_key = issue_keys[source_id]
                comments[(issue_key, comment_id)] = text
                validations.append(
                    validate_evidence(
                        issue_key=issue_key,
                        comment_id=comment_id,
                        text=text,
                        target=target,
                        acceptance_criteria=requirement["description"].split(
                            "Acceptance criterion:", 1
                        )[1].strip(),
                        source_id=source_id,
                        related_text=related_description,
                    )
                )

        requirement_id = requirement["source_id"]
        expected = EXPECTED_DECISIONS[requirement_id]
        citations = []
        for marker in EVIDENCE_MARKERS.get(requirement_id, ()):
            for ref, text in comments.items():
                if re.search(rf"\b{re.escape(marker)}\b", text):
                    citations.append(
                        {
                            "issue_key": ref[0],
                            "comment_id": ref[1],
                            "quote": marker,
                        }
                    )
                    break
        candidate = {
            "decision": expected,
            "explanation": "Fixture candidate from the reviewed scenario.",
            "supporting_evidence": citations,
        }
        resolved = resolve_requirement(
            requirement_id=requirement_id,
            summary=requirement["summary"],
            mandatory=True,
            acceptance_criteria=requirement["description"].split(
                "Acceptance criterion:", 1
            )[1].strip(),
            candidate=candidate,
            validated_evidence=validations,
            retrieved_comments=comments,
        )
        results.append((requirement_id, resolved))
    return results


def test_all_twelve_scenario_requirements_keep_expected_evidence_decisions() -> None:
    results = _scenario_results()

    assert len(results) == 12
    assert {
        requirement_id: result.decision.value
        for requirement_id, result in results
    } == EXPECTED_DECISIONS
    assert overall_decision([result for _, result in results]) == "BLOCKED"


def test_superseding_parsing_retest_survives_full_scenario_fixture() -> None:
    result = dict(_scenario_results())["REQ-10"]

    assert result.decision == RequirementDecision.PASS
    assert any(
        "PARSE-622" in citation.quote
        or citation.comment_id == "REC-9611-4"
        for citation in result.supporting_evidence
    )
    assert any(item.get("superseded") for item in result.rejected_evidence)
