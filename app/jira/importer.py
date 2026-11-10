import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.jira.client import JiraClient
from app.models.dataset import Dataset, DatasetRecord
from app.utils import adf_to_text, jql_literal, source_label, text_to_adf


logger = logging.getLogger(__name__)
SOURCE_MARKER = re.compile(r"\[source-comment:([^\]]+)\]")


@dataclass
class ImportResult:
    project_key: str
    dataset_label: str
    total_records: int
    created: int = 0
    updated: int = 0
    comments_created: int = 0
    links_created: int = 0
    errors: list[str] = field(default_factory=list)
    source_id_to_issue_key: dict[str, str] = field(default_factory=dict)


class DatasetImporter:
    def __init__(
        self,
        client: JiraClient,
        project_key: str,
        dataset: Dataset,
        *,
        project_name: str = "",
        project_lead_account_id: str = "",
        link_type: str = "Relates",
        source_id_field: str = "",
    ) -> None:
        self.client = client
        self.project_key = project_key
        self.dataset = dataset
        self.project_name = project_name
        self.project_lead_account_id = project_lead_account_id
        self.link_type = link_type
        self.source_id_field = source_id_field

    def run(self) -> ImportResult:
        project = self.client.ensure_project(
            self.project_key, self.project_name, self.project_lead_account_id
        )
        issue_types = set(self.client.issue_types(str(project["id"])))
        existing_issues = self.client.search_jira_issues(
            f"project = {jql_literal(self.project_key)} "
            f"AND labels = {jql_literal(self.dataset.dataset_label)}",
            fields=["summary", "description", "status", "labels", "issuelinks"],
            max_results=max(1000, len(self.dataset.records) + 1),
        )
        existing_by_source: dict[str, dict[str, Any]] = {}
        for issue in existing_issues:
            for label in issue.get("fields", {}).get("labels", []):
                if label.startswith("source-id-"):
                    source_id = label.removeprefix("source-id-")
                    if source_id in existing_by_source:
                        raise RuntimeError(
                            f"Multiple Jira issues exist for source ID {source_id}"
                        )
                    existing_by_source[source_id] = issue

        result = ImportResult(
            project_key=self.project_key,
            dataset_label=self.dataset.dataset_label,
            total_records=len(self.dataset.records),
        )
        issue_map: dict[str, str] = {}
        for record in self.dataset.records:
            try:
                existing = existing_by_source.get(record.source_id)
                fields = self._issue_fields(record, self._issue_type(record.kind, issue_types))
                if self.source_id_field:
                    fields[self.source_id_field] = record.source_id
                if existing:
                    key = existing["key"]
                    self.client.update_issue(key, fields)
                    result.updated += 1
                else:
                    created = self.client.create_issue(fields)
                    key = created["key"]
                    result.created += 1
                issue_map[record.source_id] = key
                self._set_recorded_status(key, record.recorded_status)
                result.comments_created += self._import_comments(key, record)
            except Exception as error:
                logger.exception("Could not import dataset record %s", record.source_id)
                result.errors.append(f"{record.source_id}: {error}")

        existing_edges = self._edges_from_issues(existing_issues)
        result.source_id_to_issue_key = issue_map
        for record in self.dataset.records:
            source_key = issue_map.get(record.source_id)
            if not source_key:
                continue
            for linked_source_id in record.links:
                linked_key = issue_map.get(linked_source_id)
                if not linked_key:
                    result.errors.append(
                        f"{record.source_id}->{linked_source_id}: linked issue was not imported"
                    )
                    continue
                edge = tuple(sorted((source_key, linked_key)))
                if edge in existing_edges:
                    continue
                try:
                    self.client.create_issue_link(self.link_type, source_key, linked_key)
                    existing_edges.add(edge)
                    result.links_created += 1
                except Exception as error:
                    logger.exception(
                        "Could not import relationship %s -> %s",
                        record.source_id,
                        linked_source_id,
                    )
                    result.errors.append(
                        f"{record.source_id}->{linked_source_id}: {error}"
                    )

        return result

    def _issue_type(self, kind: str, available: set[str]) -> str:
        preferences = {
            "manifest": ("Epic", "Story", "Task"),
            "requirement": ("Requirement", "Story", "Task"),
            "work": ("Task", "Story", "Work"),
            "evidence": ("Test", "Evidence", "Task"),
        }
        for name in preferences.get(kind, ("Task",)):
            if name in available:
                return name
        if available:
            raise ValueError(f"No supported Jira issue type found; available: {sorted(available)}")
        raise ValueError("Jira project has no available issue types")

    def _issue_fields(self, record: DatasetRecord, issue_type: str) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "project": {"key": self.project_key},
            "issuetype": {"name": issue_type},
            "summary": f"[{record.source_id}] {record.summary}"[:255],
            "description": text_to_adf(
                f"Source ID: {record.source_id}\n"
                f"Record kind: {record.kind}\n"
                f"Recorded status: {record.recorded_status}\n"
                f"Dataset release: {record.release}\n\n"
                f"{record.description}"
            ),
            "labels": [
                self.dataset.dataset_label,
                source_label(record.source_id),
                f"record-kind-{record.kind}",
            ],
        }
        return fields

    def _set_recorded_status(self, issue_key: str, expected: str) -> None:
        issue = self.client.get_jira_issue(issue_key)
        current = (
            issue.get("fields", {}).get("status", {}).get("name", "").casefold()
        )
        if current == expected.casefold():
            return
        transitions = self.client.get_transitions(issue_key)
        match = next(
            (
                transition
                for transition in transitions
                if transition.get("to", {}).get("name", "").casefold()
                == expected.casefold()
            ),
            None,
        )
        if not match:
            raise RuntimeError(
                f"Cannot preserve recorded status {expected!r} for {issue_key}; "
                f"available transitions: "
                f"{[item.get('to', {}).get('name') for item in transitions]}"
            )
        self.client.transition_issue(issue_key, str(match["id"]))

    def _import_comments(self, issue_key: str, record: DatasetRecord) -> int:
        known_markers = {
            marker
            for comment in self.client.get_jira_comments(issue_key)
            for marker in SOURCE_MARKER.findall(adf_to_text(comment.get("body")))
        }
        created = 0
        for index, original in enumerate(record.comments):
            digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:16]
            marker = f"{record.source_id}:{index}:{digest}"
            if marker in known_markers:
                continue
            body = (
                "Original scenario comment (fictional actor and timestamp are preserved "
                "in the text below; Jira author and creation time are importer metadata).\n"
                f"{original}\n[source-comment:{marker}]"
            )
            self.client.create_comment(issue_key, text_to_adf(body))
            known_markers.add(marker)
            created += 1
        return created

    def _edges_from_issues(
        self, issues: list[dict[str, Any]]
    ) -> set[tuple[str, str]]:
        edges: set[tuple[str, str]] = set()
        for issue in issues:
            key = issue["key"]
            for link in issue.get("fields", {}).get("issuelinks", []):
                other = link.get("outwardIssue") or link.get("inwardIssue")
                if other and other.get("key"):
                    edges.add(tuple(sorted((str(key), other["key"]))))
        return edges
