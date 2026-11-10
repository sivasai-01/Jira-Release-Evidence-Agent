from typing import Any

from app.jira.client import JiraClient
from app.utils import adf_to_text, jql_literal, source_label


class IssueService:
    def __init__(self, client: JiraClient, project_key: str, dataset_label: str) -> None:
        self.client = client
        self.project_key = project_key
        self.dataset_label = dataset_label

    def search_by_source_id(self, source_id: str) -> list[dict[str, Any]]:
        jql = (
            f"project = {jql_literal(self.project_key)} "
            f"AND labels = {jql_literal(self.dataset_label)} "
            f"AND labels = {jql_literal(source_label(source_id))}"
        )
        return self.client.search_jira_issues(jql, max_results=2)

    def search_dataset_issues(self, max_results: int = 1000) -> list[dict[str, Any]]:
        jql = (
            f"project = {jql_literal(self.project_key)} "
            f"AND labels = {jql_literal(self.dataset_label)} "
            "ORDER BY key ASC"
        )
        issues = self.client.search_jira_issues(
            jql,
            fields=["summary", "description", "issuetype", "status", "labels", "issuelinks", "project"],
            max_results=max_results + 1,
        )
        scoped: list[dict[str, Any]] = []
        for issue in issues:
            fields = issue.get("fields", {})
            if (
                fields.get("project", {}).get("key") != self.project_key
                or self.dataset_label not in fields.get("labels", [])
            ):
                raise ValueError(
                    f"Jira returned issue {issue.get('key')} outside the configured dataset scope."
                )
            scoped.append(issue)
        if len(scoped) > max_results:
            raise ValueError(
                f"Dataset scope reached the configured record search limit ({max_results}); "
                "increase MAX_DATASET_RECORDS to ensure complete discovery."
            )
        return scoped

    @staticmethod
    def get_record_kind(issue: dict[str, Any]) -> str | None:
        labels = issue.get("fields", {}).get("labels", [])
        for label in labels:
            if label.startswith("record-kind-"):
                return label.removeprefix("record-kind-")
        description = adf_to_text(issue.get("fields", {}).get("description"))
        for line in description.splitlines():
            prefix, separator, value = line.partition(":")
            if separator and prefix.strip().casefold() == "record kind":
                return value.strip().casefold()
        return None

    @staticmethod
    def get_source_id(issue: dict[str, Any]) -> str | None:
        labels = issue.get("fields", {}).get("labels", [])
        for label in labels:
            if label.startswith("source-id-"):
                return label.removeprefix("source-id-")
        summary = str(issue.get("fields", {}).get("summary", ""))
        if summary.startswith("[") and "]" in summary:
            return summary[1 : summary.index("]")]
        return None

    def get_scoped_issue(self, issue_key: str) -> dict[str, Any]:
        issue = self.client.get_jira_issue(issue_key)
        fields = issue.get("fields", {})
        project = fields.get("project", {}).get("key")
        labels = fields.get("labels", [])
        if project != self.project_key or self.dataset_label not in labels:
            raise ValueError(
                f"Issue {issue_key} is outside project {self.project_key} "
                f"or dataset scope {self.dataset_label}"
            )
        return issue
