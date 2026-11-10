import pytest

from app.jira.issue_service import IssueService
from app.services.execution_service import discover_release_requirements


class FakeIssueService:
    def __init__(self, issues: list[dict]) -> None:
        self.issues = issues

    def search_dataset_issues(self, max_results: int) -> list[dict]:
        assert max_results == 500
        return self.issues

    @staticmethod
    def get_record_kind(issue: dict) -> str | None:
        for label in issue["fields"].get("labels", []):
            if label.startswith("record-kind-"):
                return label.removeprefix("record-kind-")
        for line in issue["fields"].get("description", "").splitlines():
            prefix, separator, value = line.partition(":")
            if separator and prefix.strip().casefold() == "record kind":
                return value.strip().casefold()
        return None


def issue(
    key: str,
    source_id: str,
    kind: str,
    *,
    links: list[str] | None = None,
    kind_label: bool = False,
) -> dict:
    labels = ["assessment-scope"]
    if kind_label:
        labels.append(f"record-kind-{kind}")
    description = "" if kind_label else f"Record kind: {kind}"
    if kind == "requirement":
        description += (
            f"\nRequirement ID: {source_id}\n"
            "Mandatory: yes\n"
            "Acceptance criterion: Release tests pass.\n"
            'Target: {"release":"R2.4","commit":"b7a91f2",'
            '"service":"atlas-assistant","environment":"production","prompt":"p7"}'
        )
    return {
        "key": key,
        "fields": {
            "summary": f"[{source_id}] {kind.title()}",
            "description": description,
            "labels": labels,
            "project": {"key": "SCRUM"},
            "issuelinks": [
                {"outwardIssue": {"key": linked_key}} for linked_key in (links or [])
            ],
        },
    }


def test_discovers_manifest_and_requirements_by_kind_not_source_id() -> None:
    requirement_keys = [f"SCRUM-{number}" for number in range(9, 21)]
    manifest = issue(
        "SCRUM-5",
        "RELEASE-CATALOG",
        "manifest",
        links=requirement_keys,
    )
    requirements = [
        issue(
            f"SCRUM-{number}",
            f"ACCEPT-{number - 8:02d}",
            "requirement",
            kind_label=number == 10,
        )
        for number in range(9, 21)
    ]
    requirements.append(issue("SCRUM-21", "HIST-01", "evidence"))

    discovered_manifest, discovered_requirements = discover_release_requirements(
        FakeIssueService([*requirements, manifest]), max_dataset_records=500
    )

    assert discovered_manifest["key"] == "SCRUM-5"
    assert len(discovered_requirements) == 12
    assert {item["key"] for item in discovered_requirements} == set(requirement_keys)


def test_discovery_rejects_missing_or_ambiguous_manifest() -> None:
    service = FakeIssueService([issue("SCRUM-1", "REQ-1", "requirement")])
    with pytest.raises(ValueError, match="found 0"):
        discover_release_requirements(service, max_dataset_records=500)

    service.issues.extend(
        [
            issue("SCRUM-2", "RELEASE-A", "manifest", links=["SCRUM-1"]),
            issue("SCRUM-3", "RELEASE-B", "manifest", links=["SCRUM-1"]),
        ]
    )
    with pytest.raises(ValueError, match="found 2"):
        discover_release_requirements(service, max_dataset_records=500)


def test_discovery_rejects_manifest_links_outside_scope() -> None:
    service = FakeIssueService(
        [
            issue("SCRUM-2", "RELEASE-CATALOG", "manifest", links=["OTHER-1"]),
        ]
    )
    with pytest.raises(ValueError, match="outside the configured dataset scope"):
        discover_release_requirements(service, max_dataset_records=500)


def test_dataset_issue_search_enforces_scope_and_detects_truncation() -> None:
    scoped_issue = issue("SCRUM-5", "CATALOG", "manifest")

    class FakeSearchClient:
        def __init__(self, results: list[dict]) -> None:
            self.results = results
            self.query = ""
            self.max_results = 0

        def search_jira_issues(self, jql: str, **kwargs) -> list[dict]:
            self.query = jql
            self.max_results = kwargs["max_results"]
            return self.results

    client = FakeSearchClient([scoped_issue])
    results = IssueService(client, "SCRUM", "assessment-scope").search_dataset_issues(500)
    assert results == [scoped_issue]
    assert 'project = "SCRUM"' in client.query
    assert 'labels = "assessment-scope"' in client.query
    assert client.max_results == 501

    over_limit = FakeSearchClient([scoped_issue] * 501)
    with pytest.raises(ValueError, match="search limit"):
        IssueService(over_limit, "SCRUM", "assessment-scope").search_dataset_issues(500)
