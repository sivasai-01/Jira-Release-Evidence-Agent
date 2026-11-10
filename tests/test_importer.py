from typing import Any

from app.jira.importer import DatasetImporter
from app.models.dataset import Dataset
from app.utils import adf_to_text


class FakeJira:
    def __init__(self) -> None:
        self.issues: dict[str, dict[str, Any]] = {}
        self.comments: dict[str, list[dict[str, Any]]] = {}
        self.links: set[tuple[str, str]] = set()
        self.next_key = 1

    def ensure_project(self, project_key: str, *_: str) -> dict[str, Any]:
        return {"id": "100", "key": project_key}

    def issue_types(self, _: str) -> list[str]:
        return ["Task", "Story", "Epic"]

    def search_jira_issues(self, jql: str, **_: Any) -> list[dict[str, Any]]:
        return [
            issue
            for issue in self.issues.values()
            if "assessment-test" in jql
            and "assessment-test" in issue["fields"]["labels"]
        ]

    def create_issue(self, fields: dict[str, Any]) -> dict[str, str]:
        key = f"TYN-{self.next_key}"
        self.next_key += 1
        source_label = next(label for label in fields["labels"] if label.startswith("source-id-"))
        issue = {
            "key": key,
            "fields": {
                **fields,
                "status": {"name": "To Do"},
                "project": {"key": "TYN"},
                "issuelinks": [],
            },
            "source_label": source_label,
        }
        self.issues[key] = issue
        self.comments[key] = []
        return {"key": key}

    def update_issue(self, key: str, fields: dict[str, Any]) -> None:
        self.issues[key]["fields"].update(fields)

    def get_jira_issue(self, key: str) -> dict[str, Any]:
        return self.issues[key]

    def get_transitions(self, _: str) -> list[dict[str, Any]]:
        return [{"id": name, "to": {"name": name}} for name in ("To Do", "In Progress", "Done")]

    def transition_issue(self, key: str, transition_id: str) -> None:
        self.issues[key]["fields"]["status"] = {"name": transition_id}

    def get_jira_comments(self, key: str) -> list[dict[str, Any]]:
        return self.comments[key]

    def create_comment(self, key: str, body: dict[str, Any]) -> dict[str, str]:
        comment_id = str(len(self.comments[key]) + 1)
        self.comments[key].append({"id": comment_id, "body": body})
        return {"id": comment_id}

    def create_issue_link(self, _: str, inward_key: str, outward_key: str) -> None:
        self.links.add(tuple(sorted((inward_key, outward_key))))
        self.issues[inward_key]["fields"]["issuelinks"].append(
            {"outwardIssue": {"key": outward_key}}
        )
        self.issues[outward_key]["fields"]["issuelinks"].append(
            {"inwardIssue": {"key": inward_key}}
        )


def sample_dataset() -> Dataset:
    return Dataset.model_validate(
        {
            "dataset_label": "assessment-test",
            "target": {"release": "R2.4"},
            "records": [
                {
                    "source_id": "MANIFEST",
                    "kind": "manifest",
                    "summary": "Manifest",
                    "release": "R2.4",
                    "recorded_status": "To Do",
                    "description": "Target release.",
                    "comments": [
                        "Recorded event 2026-01-01T10:00:00Z | actor QA | role QA\nReviewed."
                    ],
                    "links": ["REQ-01"],
                },
                {
                    "source_id": "REQ-01",
                    "kind": "requirement",
                    "summary": "Regression",
                    "release": "R2.4",
                    "recorded_status": "Done",
                    "description": "Mandatory: yes\nAcceptance criterion: pass all tests",
                    "comments": [],
                    "links": [],
                },
            ],
        }
    )


def test_import_is_repeatable_and_preserves_comments_status_and_links() -> None:
    jira = FakeJira()
    dataset = sample_dataset()
    importer = DatasetImporter(jira, "TYN", dataset)

    first = importer.run()
    second = importer.run()

    assert first.created == 2
    assert first.comments_created == 1
    assert first.links_created == 1
    assert first.errors == []
    assert second.created == 0
    assert second.updated == 2
    assert second.comments_created == 0
    assert second.links_created == 0
    assert first.source_id_to_issue_key == second.source_id_to_issue_key
    assert len(jira.issues) == 2
    assert len(jira.links) == 1
    manifest = next(
        issue
        for issue in jira.issues.values()
        if issue["source_label"] == "source-id-MANIFEST"
    )
    assert "assessment-test" in manifest["fields"]["labels"]
    assert "source-id-MANIFEST" in manifest["fields"]["labels"]
    assert "record-kind-manifest" in manifest["fields"]["labels"]
    assert "[MANIFEST] Manifest" == manifest["fields"]["summary"]
    assert "Source ID: MANIFEST" in adf_to_text(manifest["fields"]["description"])
    requirement = jira.issues[first.source_id_to_issue_key["REQ-01"]]
    assert requirement["fields"]["status"]["name"] == "Done"
    imported_comment = adf_to_text(jira.comments[manifest["key"]][0]["body"])
    assert "2026-01-01T10:00:00Z" in imported_comment
    assert "actor QA | role QA" in imported_comment
    assert "Reviewed." in imported_comment
    assert "[source-comment:MANIFEST:0:" in imported_comment
