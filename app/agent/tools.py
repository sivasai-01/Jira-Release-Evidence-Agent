import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import StructuredTool, tool

from app.assessment.evidence_validator import validate_evidence
from app.config import Settings
from app.jira.client import JiraApiError, JiraClient
from app.jira.issue_service import IssueService
from app.models.dataset import TargetConfig
from app.services.audit_service import AuditStore
from app.utils import (
    adf_to_text,
    issue_text,
    jql_literal,
    parse_evidence_metadata,
    parse_recorded_event,
)


logger = logging.getLogger(__name__)


class InvestigationLimitError(RuntimeError):
    pass


@dataclass
class InvestigationContext:
    client: JiraClient
    issues: IssueService
    audit: AuditStore
    settings: Settings
    assessment_id: str
    project_key: str
    dataset_label: str
    target: TargetConfig
    requirement_id: str
    acceptance_criteria: str
    manifest_policy: str
    used_calls: list[int]
    lock: threading.Lock
    issue_depth: dict[str, int] = field(default_factory=dict)
    excluded_issue_keys: set[str] = field(default_factory=set)
    root_issue_key: str | None = None
    link_relevance_terms: set[str] = field(default_factory=set)
    retrieved_comments: dict[tuple[str, str], str] = field(default_factory=dict)
    issue_sources: dict[str, str | None] = field(default_factory=dict)
    issue_descriptions: dict[str, str] = field(default_factory=dict)
    issue_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    comments_loaded: set[str] = field(default_factory=set)
    linked_issues_loaded: set[str] = field(default_factory=set)
    linked_issue_results: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    search_cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    validation_cache: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    information_version: int = 0
    validations: list[dict[str, Any]] = field(default_factory=list)
    model_evidence: list[dict[str, Any]] = field(default_factory=list)
    used_context_chars: int = 0
    model_usage: dict[str, int | None] = field(
        default_factory=lambda: {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }
    )
    metrics: dict[str, Any] = field(default_factory=dict)

    def reserve_call(self) -> None:
        with self.lock:
            if self.used_calls[0] >= self.settings.max_tool_calls:
                raise InvestigationLimitError("Assessment Jira-tool call budget exhausted")
            self.used_calls[0] += 1

    def issue_depth_for(self, issue_key: str) -> int:
        return self.issue_depth.get(issue_key, self.settings.max_recursion_depth + 1)

    def register_issue(self, issue: dict[str, Any], depth: int) -> None:
        issue_key = issue["key"]
        self.issue_depth[issue_key] = min(depth, self.issue_depth.get(issue_key, depth))
        self.issue_sources[issue_key] = self.issues.get_source_id(issue)
        fields = issue.get("fields", {})
        description = adf_to_text(fields.get("description"))
        issue_type = fields.get("issuetype", {}).get("name")
        snapshot = {
            "issue_key": issue_key,
            "source_id": self.issue_sources[issue_key],
            "issue_type": issue_type,
            "summary": str(fields.get("summary", "")),
            "description": description,
            "status": fields.get("status", {}).get("name"),
            "links": [
                {
                    "issue_key": (
                        link.get("outwardIssue") or link.get("inwardIssue") or {}
                    ).get("key"),
                    "link_type": link.get("type", {}).get("name", "unknown"),
                    "summary": str(
                        (
                            link.get("outwardIssue")
                            or link.get("inwardIssue")
                            or {}
                        ).get("fields", {}).get("summary", "")
                    ),
                }
                for link in fields.get("issuelinks", [])
                if (link.get("outwardIssue") or link.get("inwardIssue") or {}).get("key")
            ],
        }
        previous = self.issue_snapshots.get(issue_key)
        self.issue_snapshots[issue_key] = snapshot
        self.issue_descriptions[issue_key] = issue_text(issue)
        if previous is None:
            self.information_version += 1
        elif previous != snapshot:
            self.information_version += 1
        item = {
            "type": "issue",
            "issue_key": issue_key,
            "source_id": snapshot["source_id"],
            "issue_type": issue_type,
            "summary": snapshot["summary"],
            "description": description,
            "status": snapshot["status"],
            "links": snapshot["links"],
        }
        for index, existing in enumerate(self.model_evidence):
            if existing.get("type") == "issue" and existing.get("issue_key") == issue_key:
                self.model_evidence[index] = item
                return
        self.model_evidence.append(item)

    def is_relevant_link(self, summary: str, link_type: str) -> bool:
        if any(
            marker in link_type.casefold()
            for marker in ("depend", "block", "is required by")
        ):
            return True
        summary_terms = set(re.findall(r"[a-z0-9]+", summary.casefold()))
        return bool(summary_terms & self.link_relevance_terms)

    def bound_response(self, value: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        if len(encoded) <= self.settings.max_context_chars:
            return value
        marker = "[tool output truncated to configured context limit]"
        content = encoded[: max(0, self.settings.max_context_chars - len(marker))]
        bounded = {"truncated": True, "content": content + marker}
        return bounded

    def log_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        summary: str,
        started: float,
    ) -> None:
        elapsed_ms = (time.perf_counter() - started) * 1000
        issue_key = arguments.get("issue_key")
        trace_arguments = {
            "requirement_id": self.requirement_id,
            **arguments,
        }
        if issue_key in self.issue_sources and self.issue_sources[issue_key]:
            trace_arguments["source_id"] = self.issue_sources[issue_key]
        self.audit.record_tool(
            self.assessment_id, tool_name, trace_arguments, summary, elapsed_ms
        )

    def comments_for_model(self, issue_key: str) -> list[dict[str, str]]:
        return [
            {"comment_id": comment_id, "text": text[:8000]}
            for (key, comment_id), text in self.retrieved_comments.items()
            if key == issue_key
        ]

    def compact_evidence(self) -> dict[str, Any]:
        evidence_terms = (
            "test", "qa", "verif", "validat", "pass", "fail", "defect", "bug",
            "approv", "sign-off", "supersed", "retest", "run ", "evaluation",
            "accuracy", "threshold", "case", "expos", "leak", "duplicate",
            "abort", "inconclusive", "scheduled", "fix", "implement", "prompt",
            "block", "security", "rollback", "handoff", "keyboard", "accessible",
            "parsing", "latency", "cost", "redact",
        )
        stop_words = {
            "acceptance", "criterion", "criteria", "target", "release", "commit",
            "service", "environment", "prompt", "version", "must", "should",
            "shall", "with", "from", "that", "this", "their", "which", "where",
            "into", "only", "each", "than", "more", "less", "the", "and", "for",
            "are", "has", "have", "will", "can", "may",
        }
        criterion_terms = {
            word
            for word in re.findall(
                r"[a-z0-9]+",
                self.acceptance_criteria.casefold(),
            )
            if len(word) >= 4 and word not in stop_words
        }
        validated_by_ref = {
            (item["issue_key"], str(item["comment_id"])): item
            for item in self.validations
        }
        comments: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for (issue_key, comment_id), text in self.retrieved_comments.items():
            ref = (issue_key, str(comment_id))
            if ref in seen:
                continue
            seen.add(ref)
            validation = self.validation_cache.get(ref) or validated_by_ref.get(ref)
            metadata = parse_evidence_metadata(text)
            is_relevant = (
                any(value for value in metadata.values())
                or any(term in text.casefold() for term in evidence_terms)
                or any(term in text.casefold() for term in criterion_terms)
            )
            if not is_relevant:
                continue
            event = parse_recorded_event(text)
            compact: dict[str, Any] = {
                "issue_key": issue_key,
                "source_id": self.issue_sources.get(issue_key),
                "comment_id": str(comment_id),
                "recorded_at": event.get("recorded_at"),
                "actor": event.get("actor"),
                "role": event.get("role"),
                "text": self._comment_excerpt(text, evidence_terms),
                "metadata": metadata,
            }
            if validation is not None:
                compact["applicability"] = {
                    "applicable": validation.get("applicable"),
                    "missing_dimensions": validation.get("missing_dimensions", []),
                    "mismatched_dimensions": validation.get("mismatched_dimensions", {}),
                    "rejection_reasons": validation.get("rejection_reasons", []),
                }
            comments.append(compact)

        relevant_issue_keys = {item["issue_key"] for item in comments}
        for key, snapshot in self.issue_snapshots.items():
            issue_text = " ".join(
                str(snapshot.get(field, ""))
                for field in ("issue_type", "summary", "description")
            ).casefold()
            if (
                any(term in issue_text for term in evidence_terms)
                or any(term in issue_text for term in criterion_terms)
            ):
                relevant_issue_keys.add(key)
        issues = [
            {
                "issue_key": key,
                "source_id": snapshot.get("source_id"),
                "issue_type": snapshot.get("issue_type"),
                "summary": snapshot.get("summary", "")[:240],
                "status": snapshot.get("status"),
                "description_excerpt": self._description_excerpt(
                    snapshot.get("description", "")
                ),
                "links": snapshot.get("links", []),
            }
            for key, snapshot in self.issue_snapshots.items()
            if key in relevant_issue_keys
        ]
        return {"issues": issues, "comments": comments}

    def validate_retrieved_comments(self) -> None:
        for (issue_key, comment_id), text in self.retrieved_comments.items():
            ref = (issue_key, str(comment_id))
            if ref in self.validation_cache:
                continue
            related_text = self.issue_descriptions.get(issue_key, "")
            result = validate_evidence(
                issue_key=issue_key,
                comment_id=str(comment_id),
                text=text,
                target=self.target,
                acceptance_criteria=self.acceptance_criteria,
                source_id=self.issue_sources.get(issue_key),
                related_text=related_text,
            )
            self.validation_cache[ref] = result
            self.validations.append(result)
            self.information_version += 1
            self.audit.record_tool(
                self.assessment_id,
                "validate_evidence",
                {
                    "requirement_id": self.requirement_id,
                    "issue_key": issue_key,
                    "comment_id": str(comment_id),
                    "source_id": self.issue_sources.get(issue_key),
                },
                (
                    "Evidence is applicable."
                    if result["applicable"]
                    else "Evidence is not applicable: "
                    + "; ".join(result["rejection_reasons"])
                ),
                0,
            )

    def record_llm_call(
        self,
        *,
        operation: str,
        prompt_characters: int,
        usage: dict[str, int],
        attempt: int,
        status: str = "completed",
        usage_unavailable_reason: str | None = None,
    ) -> None:
        calls = self.metrics.setdefault("llm_call_details", [])
        call = {
            "requirement_id": self.requirement_id,
            "operation": operation,
            "attempt": attempt,
            "status": status,
            "prompt_characters": prompt_characters,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        if usage_unavailable_reason and not usage:
            call["usage_unavailable_reason"] = usage_unavailable_reason
        calls.append(call)
        self.metrics["llm_prompt_characters"] = (
            self.metrics.get("llm_prompt_characters", 0) + prompt_characters
        )
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if key in usage:
                self.model_usage[key] = (self.model_usage.get(key) or 0) + usage[key]

    @staticmethod
    def _description_excerpt(description: str) -> str:
        selected = [
            line.strip()
            for line in description.splitlines()
            if any(
                term in line.casefold()
                for term in (
                    "acceptance", "criterion", "depend", "blocks", "blocked",
                    "defect", "approval", "target", "release", "commit", "service",
                )
            )
        ]
        return "\n".join(selected[:5])[:700]

    @staticmethod
    def _comment_excerpt(text: str, terms: tuple[str, ...]) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        event = next(
            (line for line in lines if line.casefold().startswith("recorded event ")),
            "",
        )
        relevant = [
            line
            for line in lines
            if line != event
            and any(term in line.casefold() for term in terms)
        ]
        if not relevant:
            return text[:800]
        excerpt = "\n".join(([event] if event else []) + relevant)
        return excerpt[:2400]


def _validate_jql_fragment(query: str) -> None:
    if not query.strip():
        raise ValueError("jql_query cannot be empty")
    if any(ord(character) < 32 for character in query) or ";" in query:
        raise ValueError("jql_query cannot contain control characters or statement separators")
    if "order by" in query.casefold():
        raise ValueError("Provide JQL conditions only; sorting is managed by Jira.")
    depth = 1
    quoted = False
    escaped = False
    for character in query:
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            quoted = not quoted
        elif not quoted and character == "(":
            depth += 1
        elif not quoted and character == ")":
            depth -= 1
            if depth < 1:
                raise ValueError("jql_query cannot escape the enforced project/dataset scope")
    if quoted or depth != 1:
        raise ValueError("jql_query contains unbalanced quotes or parentheses")


def _parse_investigation_arguments(value: str) -> dict[str, Any]:
    if len(value) > 10_000:
        raise ValueError("arguments_json exceeds the 10000-character audit limit")
    candidates = (value, value.replace(r"\"", '"'))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("arguments_json must encode a JSON object")


def build_tools(context: InvestigationContext) -> list[StructuredTool]:
    @tool
    def search_jira_issues(jql_query: str, max_results: int = 20) -> dict[str, Any]:
        """Search Jira with JQL. Project and dataset label are always enforced."""
        started = time.perf_counter()
        arguments = {"jql_query": jql_query, "max_results": max_results}
        try:
            context.reserve_call()
            limit = max(1, min(max_results, context.settings.max_search_results))
            _validate_jql_fragment(jql_query)
            scoped_jql = (
                f"project = {jql_literal(context.project_key)} "
                f"AND labels = {jql_literal(context.dataset_label)} "
                f"AND ({jql_query})"
            )
            cache_key = f"{scoped_jql}\nMAX_RESULTS={limit}"
            if cache_key in context.search_cache:
                cached = context.search_cache[cache_key]
                context.log_tool(
                    "search_jira_issues",
                    arguments,
                    f"Reused cached search result containing {len(cached)} issues.",
                    started,
                )
                return context.bound_response({"issues": cached, "count": len(cached), "cached": True})
            found = context.client.search_jira_issues(
                scoped_jql,
                fields=["summary", "description", "issuetype", "status", "labels", "issuelinks", "project"],
                max_results=limit,
            )
            results = []
            for issue in found:
                context.register_issue(issue, 0)
                fields = issue.get("fields", {})
                results.append(
                    {
                        "issue_key": issue["key"],
                        "source_id": context.issue_sources.get(issue["key"]),
                        "issue_type": fields.get("issuetype", {}).get("name"),
                        "summary": str(fields.get("summary", ""))[:240],
                        "status": fields.get("status", {}).get("name"),
                    }
                )
            context.search_cache[cache_key] = results
            if not found:
                context.information_version += 1
            summary = f"Found {len(results)} issues in the enforced project and dataset scope."
            context.log_tool("search_jira_issues", arguments, summary, started)
            return context.bound_response({"issues": results, "count": len(results)})
        except (JiraApiError, ValueError, InvestigationLimitError) as error:
            context.log_tool("search_jira_issues", arguments, f"ERROR: {error}", started)
            logger.warning("Jira search tool failed: %s", error)
            return context.bound_response({"error": str(error)})

    @tool
    def get_jira_issue(issue_key: str) -> dict[str, Any]:
        """Retrieve a Jira issue description after verifying its project and dataset label."""
        started = time.perf_counter()
        arguments = {"issue_key": issue_key}
        try:
            context.reserve_call()
            depth = context.issue_depth_for(issue_key)
            if depth > context.settings.max_recursion_depth:
                raise InvestigationLimitError(
                    f"Recursion depth limit prevents retrieval of {issue_key}"
                )
            snapshot = context.issue_snapshots.get(issue_key)
            if snapshot is None:
                issue = context.issues.get_scoped_issue(issue_key)
                context.register_issue(issue, depth)
                snapshot = context.issue_snapshots[issue_key]
            result = {
                "issue_key": issue_key,
                "source_id": snapshot.get("source_id"),
                "summary": snapshot.get("summary", "")[:240],
                "status": snapshot.get("status"),
                "description_excerpt": context._description_excerpt(
                    snapshot.get("description", "")
                ),
                "issue_type": snapshot.get("issue_type"),
                "links": snapshot.get("links", []),
            }
            context.log_tool("get_jira_issue", arguments, "Retrieved one in-scope Jira issue.", started)
            return context.bound_response(result)
        except (JiraApiError, ValueError, InvestigationLimitError) as error:
            context.log_tool("get_jira_issue", arguments, f"ERROR: {error}", started)
            logger.warning("Jira issue tool failed: %s", error)
            return context.bound_response({"error": str(error)})

    @tool
    def get_jira_comments(issue_key: str) -> dict[str, Any]:
        """Retrieve every paginated comment on an in-scope Jira issue."""
        started = time.perf_counter()
        arguments = {"issue_key": issue_key}
        try:
            context.reserve_call()
            depth = context.issue_depth_for(issue_key)
            if depth > context.settings.max_recursion_depth:
                raise InvestigationLimitError(
                    f"Recursion depth limit prevents comment retrieval for {issue_key}"
                )
            if issue_key not in context.issue_snapshots:
                issue = context.issues.get_scoped_issue(issue_key)
                context.register_issue(issue, depth)
            priority_terms = (
                "qa",
                "verification",
                "verified",
                "passed",
                "failed",
                "defect",
                "approved",
                "approval",
                "supersed",
                "exposure",
                "aborted",
                "cases",
                "threshold",
                "cost",
                "handoff",
                "retest",
                "completed",
                "reopen",
                "mismatch",
            )
            if issue_key in context.comments_loaded:
                cached_comments = sorted(
                    [
                        {"id": comment_id, "text": text}
                        for (key, comment_id), text in context.retrieved_comments.items()
                        if key == issue_key
                    ],
                    key=lambda comment: sum(
                        term in comment["text"].casefold()
                        for term in priority_terms
                    ),
                    reverse=True,
                )
                comments = [
                    {
                        "comment_id": comment["id"],
                        "text": context._comment_excerpt(
                            comment["text"], priority_terms
                        ),
                    }
                    for comment in cached_comments
                ]
                context.log_tool(
                    "get_jira_comments",
                    arguments,
                    f"Reused {len(comments)} comments already retrieved from Jira.",
                    started,
                )
                return context.bound_response(
                    {
                        "issue_key": issue_key,
                        "comments": comments,
                        "count": len(comments),
                        "cached": True,
                    }
                )
            comments = context.client.get_jira_comments(issue_key)
            context.comments_loaded.add(issue_key)
            context.information_version += 1
            results: list[dict[str, str]] = []
            truncated = False
            comment_texts = [
                (comment, adf_to_text(comment.get("body")))
                for comment in comments
            ]
            context.retrieved_comments.update(
                {
                    (issue_key, str(comment["id"])): text
                    for comment, text in comment_texts
                }
            )
            prioritized = sorted(
                comment_texts,
                key=lambda pair: sum(
                    term in pair[1].casefold() for term in priority_terms
                ),
                reverse=True,
            )
            for comment, text in prioritized:
                comment_id = str(comment["id"])
                excerpt = context._comment_excerpt(text, priority_terms)
                item = {
                    "type": "comment",
                    "issue_key": issue_key,
                    "comment_id": comment_id,
                    "text": excerpt,
                }
                context.model_evidence.append(item)
                results.append(
                    {
                        "comment_id": comment_id,
                        "text": excerpt,
                    }
                )
            summary = (
                f"Retrieved all {len(comments)} Jira comments; returned "
                f"{len(results)} prioritized comments within the context limit."
            )
            context.log_tool("get_jira_comments", arguments, summary, started)
            return context.bound_response(
                {
                    "issue_key": issue_key,
                    "comments": results,
                    "count": len(comments),
                    "truncated_by_context_limit": truncated,
                }
            )
        except (JiraApiError, ValueError, InvestigationLimitError) as error:
            context.log_tool("get_jira_comments", arguments, f"ERROR: {error}", started)
            logger.warning("Jira comments tool failed: %s", error)
            return context.bound_response({"error": str(error)})

    @tool
    def get_linked_issues(issue_key: str) -> dict[str, Any]:
        """Retrieve direct linked Jira issues and dependencies, excluding out-of-scope issues."""
        started = time.perf_counter()
        arguments = {"issue_key": issue_key}
        try:
            context.reserve_call()
            depth = context.issue_depth_for(issue_key)
            if depth >= context.settings.max_recursion_depth:
                raise InvestigationLimitError(
                    f"Recursion depth limit prevents following links from {issue_key}"
                )
            if issue_key in context.linked_issues_loaded:
                results = context.linked_issue_results.get(issue_key, [])
                context.log_tool(
                    "get_linked_issues",
                    arguments,
                    f"Reused cached link list containing {len(results)} issues.",
                    started,
                )
                return context.bound_response({"issues": results, "cached": True})
            issue_snapshot = context.issue_snapshots.get(issue_key)
            if issue_snapshot is None:
                issue = context.issues.get_scoped_issue(issue_key)
                context.register_issue(issue, depth)
                issue_snapshot = context.issue_snapshots[issue_key]
            links = issue_snapshot.get("links", [])
            results: list[dict[str, Any]] = []
            excluded = 0
            for link in links:
                linked_key = link.get("issue_key")
                if not linked_key:
                    continue
                if linked_key in context.excluded_issue_keys:
                    excluded += 1
                    continue
                link_type = str(link.get("link_type") or "unknown")
                linked_summary = str(link.get("summary") or "")
                is_transitive = bool(
                    context.root_issue_key
                    and issue_key != context.root_issue_key
                )
                if (
                    is_transitive
                    and linked_summary
                    and not context.is_relevant_link(linked_summary, link_type)
                ):
                    excluded += 1
                    continue
                linked = context.issue_snapshots.get(linked_key)
                if linked is None:
                    try:
                        linked_issue = context.issues.get_scoped_issue(linked_key)
                    except ValueError:
                        excluded += 1
                        continue
                    linked_summary = str(
                        linked_issue.get("fields", {}).get("summary", "")
                    )
                    if (
                        is_transitive
                        and not context.is_relevant_link(linked_summary, link_type)
                    ):
                        excluded += 1
                        continue
                    context.register_issue(linked_issue, depth + 1)
                    linked = context.issue_snapshots[linked_key]
                results.append(
                    {
                        "issue_key": linked_key,
                        "source_id": linked.get("source_id"),
                        "summary": linked.get("summary", "")[:240],
                        "status": linked.get("status"),
                        "link_type": link_type,
                    }
                )
            context.linked_issues_loaded.add(issue_key)
            context.linked_issue_results[issue_key] = results
            context.information_version += 1
            summary = f"Found {len(results)} in-scope linked issues; excluded {excluded} out-of-scope links."
            context.log_tool("get_linked_issues", arguments, summary, started)
            return context.bound_response(
                {"issues": results, "excluded_out_of_scope": excluded}
            )
        except (JiraApiError, ValueError, InvestigationLimitError) as error:
            context.log_tool("get_linked_issues", arguments, f"ERROR: {error}", started)
            logger.warning("Jira link tool failed: %s", error)
            return context.bound_response({"error": str(error)})

    @tool
    def inspect_requirement(requirement_id: str) -> dict[str, Any]:
        """Retrieve a release requirement and its acceptance criterion from Jira."""
        started = time.perf_counter()
        arguments = {"requirement_id": requirement_id}
        try:
            context.reserve_call()
            known = [
                (key, item)
                for key, item in context.issue_snapshots.items()
                if item.get("source_id") == requirement_id
            ]
            if len(known) == 1:
                issue_key, snapshot = known[0]
            else:
                found = context.issues.search_by_source_id(requirement_id)
                if len(found) != 1:
                    raise ValueError(
                        f"Expected one in-scope requirement {requirement_id}, found {len(found)}"
                    )
                issue = context.issues.get_scoped_issue(found[0]["key"])
                context.register_issue(issue, 0)
                issue_key = issue["key"]
                snapshot = context.issue_snapshots[issue_key]
            result = {
                "issue_key": issue_key,
                "source_id": snapshot.get("source_id"),
                "summary": snapshot.get("summary", "")[:240],
                "description_excerpt": context._description_excerpt(
                    snapshot.get("description", "")
                ),
                "status": snapshot.get("status"),
            }
            context.log_tool("inspect_requirement", arguments, "Retrieved requirement from Jira.", started)
            return context.bound_response(result)
        except (JiraApiError, ValueError, InvestigationLimitError) as error:
            context.log_tool("inspect_requirement", arguments, f"ERROR: {error}", started)
            logger.warning("Requirement inspection tool failed: %s", error)
            return context.bound_response({"error": str(error)})

    @tool
    def validate_evidence(issue_key: str, comment_id: str) -> dict[str, Any]:
        """Check a Jira comment's release, commit, service, environment and required prompt applicability."""
        started = time.perf_counter()
        arguments = {"issue_key": issue_key, "comment_id": comment_id}
        try:
            context.reserve_call()
            text = context.retrieved_comments.get((issue_key, str(comment_id)))
            if text is None:
                raise ValueError(
                    f"Comment {comment_id} has not been retrieved from Jira; call get_jira_comments first"
                )
            ref = (issue_key, str(comment_id))
            if ref in context.validation_cache:
                result = context.validation_cache[ref]
                context.log_tool(
                    "validate_evidence",
                    arguments,
                    "Reused existing applicability validation.",
                    started,
                )
                return context.bound_response(result)
            result = validate_evidence(
                issue_key=issue_key,
                comment_id=str(comment_id),
                text=text,
                target=context.target,
                acceptance_criteria=context.acceptance_criteria,
                source_id=context.issue_sources.get(issue_key),
                related_text="\n".join(
                    sibling_text
                    for (sibling_issue, _), sibling_text in context.retrieved_comments.items()
                    if sibling_issue == issue_key
                ),
            )
            context.validation_cache[ref] = result
            context.validations.append(result)
            context.information_version += 1
            context.log_tool(
                "validate_evidence",
                arguments,
                "Evidence is applicable." if result["applicable"] else "Evidence is not applicable or is incomplete.",
                started,
            )
            return context.bound_response(result)
        except (JiraApiError, ValueError, InvestigationLimitError) as error:
            context.log_tool("validate_evidence", arguments, f"ERROR: {error}", started)
            logger.warning("Evidence validation tool failed: %s", error)
            return context.bound_response({"error": str(error)})

    @tool
    def record_investigation(
        action_summary: str, arguments_json: str, result_summary: str
    ) -> dict[str, str]:
        """Record a concise investigation action and its result in the audit trace."""
        started = time.perf_counter()
        arguments = {
            "action_summary": action_summary,
            "arguments_json": arguments_json,
            "result_summary": result_summary,
        }
        try:
            context.reserve_call()
            parsed_arguments = _parse_investigation_arguments(arguments_json)
            context.audit.record_tool(
                context.assessment_id,
                "agent_investigation_note",
                {"action_summary": action_summary, **parsed_arguments},
                result_summary[:1000],
                (time.perf_counter() - started) * 1000,
            )
            context.log_tool(
                "record_investigation",
                arguments,
                "Investigation note recorded.",
                started,
            )
            return context.bound_response({"status": "recorded"})
        except (ValueError, InvestigationLimitError) as error:
            context.log_tool("record_investigation", arguments, f"ERROR: {error}", started)
            logger.warning("Investigation note was not recorded: %s", error)
            return context.bound_response({"error": str(error)})

    return [
        search_jira_issues,
        get_jira_issue,
        get_jira_comments,
        get_linked_issues,
        inspect_requirement,
        validate_evidence,
        record_investigation,
    ]
