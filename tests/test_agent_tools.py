import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from google.api_core.exceptions import ResourceExhausted
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.graph import (
    RequirementAnalysis,
    _parse_structured_decision,
    _recent_tool_history,
    _tokens,
    _investigate_linked_records,
    create_model,
    investigate_requirement,
    _invoke_with_rate_limit_retry,
)
from app.agent.tools import InvestigationContext, build_tools
from app.agent.tools import _validate_jql_fragment
from app.config import Settings
from app.jira.issue_service import IssueService
from app.models.dataset import TargetConfig
from app.services.audit_service import AuditStore
from app.services.execution_service import _summarize_llm_calls


class FakeJira:
    def __init__(self) -> None:
        self.issue_reads = 0
        self.search_limits: list[int] = []

    def get_jira_issue(self, _: str) -> dict[str, Any]:
        self.issue_reads += 1
        return {
            "key": "TYN-1",
            "fields": {
                "project": {"key": "TYN"},
                "labels": ["scope"],
                "summary": "[REC-1] Evidence",
                "description": "Fetched from Jira",
                "status": {"name": "Done"},
                "issuetype": {"name": "Task"},
                "issuelinks": [],
            },
        }

    def get_jira_comments(self, _: str) -> list[dict[str, Any]]:
        return [
            {
                "id": "555",
                "body": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": "Retrieved over Jira REST."}],
                        }
                    ],
                },
            }
        ]

    def search_jira_issues(
        self, _: str, fields: list[str] | None = None, max_results: int = 20
    ) -> list[dict[str, Any]]:
        self.search_limits.append(max_results)
        return []


def test_jql_fragment_cannot_escape_dataset_scope() -> None:
    _validate_jql_fragment('summary ~ "latency" AND (status = Done)')
    for query in ('summary = "x") OR project = OTHER OR (x', 'summary = "x"; DELETE'):
        try:
            _validate_jql_fragment(query)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe JQL accepted: {query}")


def test_recent_tool_history_keeps_only_complete_latest_call_batch() -> None:
    initial = HumanMessage(content='{"requirement_id":"REQ-01"}')
    old_call = AIMessage(
        content="",
        tool_calls=[{"name": "search_jira_issues", "args": {}, "id": "old"}],
    )
    old_result = ToolMessage(content="old result", tool_call_id="old")
    latest_call = AIMessage(
        content="",
        tool_calls=[{"name": "get_jira_comments", "args": {}, "id": "current"}],
    )
    latest_result = ToolMessage(content="current result", tool_call_id="current")

    compacted = _recent_tool_history(
        [initial, old_call, old_result, latest_call, latest_result]
    )

    assert compacted == [initial, latest_call, latest_result]


def test_recent_tool_history_drops_unmatched_tool_call() -> None:
    initial = HumanMessage(content="requirement")
    call = AIMessage(
        content="",
        tool_calls=[{"name": "get_jira_issue", "args": {}, "id": "unmatched"}],
    )

    assert _recent_tool_history([initial, call]) == [initial]


def test_gemini_singleton_scalar_arrays_are_safely_normalized() -> None:
    raw = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "RequirementAnalysis",
                "args": {
                    "decision": ["PASS"],
                    "explanation": ["QA verified all required cases."],
                    "supporting_evidence": [
                        {"issue_key": "SCRUM-13", "comment_id": "10044", "quote": ""}
                    ],
                    "outstanding_verification": [],
                    "clarification_required": [],
                },
                "id": "decision-call",
            }
        ],
    )

    parsed = _parse_structured_decision(
        {"raw": raw, "parsed": None, "parsing_error": ValueError("wrong scalar shape")}
    )

    assert parsed.decision == "PASS"
    assert parsed.explanation == "QA verified all required cases."
    assert parsed.supporting_evidence[0].issue_key == "SCRUM-13"


def test_gemini_ambiguous_decision_arrays_are_not_normalized() -> None:
    raw = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "RequirementAnalysis",
                "args": {
                    "decision": ["PASS", "FAIL"],
                    "explanation": ["Ambiguous."],
                    "supporting_evidence": [],
                    "outstanding_verification": [],
                    "clarification_required": [],
                },
                "id": "decision-call",
            }
        ],
    )

    with pytest.raises(ValueError):
        _parse_structured_decision(
            {"raw": raw, "parsed": None, "parsing_error": ValueError("bad shape")}
        )


def test_missing_provider_usage_fields_remain_unavailable() -> None:
    assert _tokens(
        SimpleNamespace(usage_metadata={"input_tokens": 127})
    ) == {"input_tokens": 127}


def test_provider_response_metadata_usage_is_captured_without_estimating_totals() -> None:
    response = SimpleNamespace(
        usage_metadata=None,
        response_metadata={
            "token_usage": {
                "prompt_tokens": 200,
                "completion_tokens": 25,
            }
        },
        additional_kwargs={},
    )

    assert _tokens(response) == {"input_tokens": 200, "output_tokens": 25}


def test_provider_usage_model_dump_is_captured() -> None:
    usage = SimpleNamespace(
        model_dump=lambda: {
            "prompt_token_count": 200,
            "candidates_token_count": 25,
            "total_token_count": 225,
        }
    )

    assert _tokens(SimpleNamespace(usage_metadata=usage)) == {
        "input_tokens": 200,
        "output_tokens": 25,
        "total_tokens": 225,
    }


def test_usage_summary_does_not_present_partial_token_data_as_complete() -> None:
    summary = _summarize_llm_calls(
        [
            {
                "prompt_characters": 100,
                "input_tokens": 80,
                "output_tokens": 10,
                "total_tokens": 90,
            },
            {
                "prompt_characters": 50,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            },
        ]
    )

    assert summary == {
        "calls": 2,
        "call_count": 2,
        "prompt_characters": 150,
        "input_tokens": 80,
        "input_tokens_reported_calls": 1,
        "output_tokens": 10,
        "output_tokens_reported_calls": 1,
        "total_tokens": 90,
        "total_tokens_reported_calls": 1,
        "token_usage_reported_calls": 1,
        "token_usage_unavailable_calls": 1,
        "token_usage_complete": False,
        "token_usage_status": "partial",
    }


def test_usage_summary_marks_fully_reported_token_counts_complete() -> None:
    summary = _summarize_llm_calls(
        [
            {
                "prompt_characters": 100,
                "input_tokens": 80,
                "output_tokens": 10,
                "total_tokens": 90,
            },
            {
                "prompt_characters": 50,
                "input_tokens": 30,
                "output_tokens": 5,
                "total_tokens": 35,
            },
        ]
    )

    assert summary["input_tokens"] == 110
    assert summary["output_tokens"] == 15
    assert summary["total_tokens"] == 125
    assert summary["token_usage_complete"] is True
    assert summary["token_usage_status"] == "complete"
    assert summary["token_usage_reported_calls"] == 2
    assert summary["token_usage_unavailable_calls"] == 0


def test_gemini_model_accepts_tool_and_structured_output_schemas() -> None:
    model = create_model("gemini", "offline-validation-key", "gemini-3.8-flash", 512)
    assert model.timeout is None
    structured = model.with_structured_output(RequirementAnalysis, include_raw=True)
    assert structured is not None
    assert "INSUFFICIENT_EVIDENCE" in str(RequirementAnalysis.model_json_schema())


def test_groq_model_accepts_tool_and_structured_output_schemas() -> None:
    model = create_model(
        "groq",
        "offline-validation-key",
        "qwen/qwen3.8-27b",
        512,
        "https://api.groq.com/openai/v1",
    )
    assert model.request_timeout is None
    assert model.model_name == "qwen/qwen3.8-27b"
    assert model.with_structured_output(RequirementAnalysis, include_raw=True) is not None


def test_assessment_is_not_cut_off_by_cumulative_token_usage(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "audit.sqlite3",
        reports_dir=tmp_path,
    )
    jira = FakeJira()
    context = InvestigationContext(
        client=jira,
        issues=IssueService(jira, "TYN", "scope"),
        audit=AuditStore(settings.database_path),
        settings=settings,
        assessment_id="token-budget-test",
        project_key="TYN",
        dataset_label="scope",
        target=TargetConfig(
            release="R2.4",
            commit="abc123",
            environment="production",
            service="atlas",
        ),
        requirement_id="REQ-1",
        acceptance_criteria="Verify the release.",
        manifest_policy="Use only Jira.",
        used_calls=[0],
        lock=threading.Lock(),
        model_usage={
            "input_tokens": 60_001,
            "output_tokens": 0,
            "total_tokens": 60_001,
        },
    )

    class FakeStructuredModel:
        def invoke(self, _: Any) -> dict[str, Any]:
            return {
                "parsed": RequirementAnalysis(
                    decision="INSUFFICIENT_EVIDENCE",
                    explanation="No evidence found.",
                ),
                "raw": AIMessage(content=""),
                "parsing_error": None,
            }

    class FakeModel:
        def bind_tools(self, _: Any) -> "FakeModel":
            return self

        def invoke(self, _: Any) -> AIMessage:
            return AIMessage(content="Investigation complete.")

        def with_structured_output(self, *_: Any, **__: Any) -> FakeStructuredModel:
            return FakeStructuredModel()

    result = investigate_requirement(
        context,
        cast(BaseChatModel, FakeModel()),
        {
            "requirement_id": "REQ-1",
            "issue_key": "TYN-1",
            "acceptance_criteria": "Verify the release.",
        },
    )

    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert context.metrics["llm_calls"] == 2


def test_gemini_rate_limit_retries_after_provider_delay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        database_path=tmp_path / "audit.sqlite3",
        reports_dir=tmp_path,
    )
    jira = FakeJira()
    context = InvestigationContext(
        client=jira,
        issues=IssueService(jira, "TYN", "scope"),
        audit=AuditStore(settings.database_path),
        settings=settings,
        assessment_id="rate-limit-test",
        project_key="TYN",
        dataset_label="scope",
        target=TargetConfig(
            release="R2.4",
            commit="abc123",
            environment="production",
            service="atlas",
        ),
        requirement_id="REQ-1",
        acceptance_criteria="Verify the release.",
        manifest_policy="Use only Jira.",
        used_calls=[0],
        lock=threading.Lock(),
    )
    delays: list[float] = []
    monkeypatch.setattr("app.agent.graph.time.sleep", delays.append)
    calls = 0

    def invoke() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ResourceExhausted(
                "Rate limit exceeded; retry in 8.99414274s",
                errors=[{"retry_delay": {"seconds": 8}}],
            )
        return "success"

    result = _invoke_with_rate_limit_retry(invoke, context, "test")

    assert result == "success"
    assert calls == 2
    assert delays == [8.99414274]
    assert context.metrics["llm_calls"] == 2
    assert context.metrics["llm_retries"] == 1
    assert context.metrics["llm_prompt_characters"] == 0
    assert [call["status"] for call in context.metrics["llm_call_details"]] == [
        "rate_limited",
        "completed",
    ]
    assert all(call["input_tokens"] is None for call in context.metrics["llm_call_details"])
    assert context.metrics["llm_call_details"][0]["usage_unavailable_reason"] == (
        "Rate-limited before a completion response returned token usage."
    )
    assert context.metrics["llm_call_details"][1]["usage_unavailable_reason"] == (
        "Provider response did not include token-usage metadata."
    )
    trace = context.audit.investigation_trace("rate-limit-test")
    assert trace[0]["tool"] == "llm_rate_limit_retry"
    assert trace[0]["arguments"]["retry_delay_seconds"] == 8.99414274


def test_non_rate_limit_llm_errors_are_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        database_path=tmp_path / "audit.sqlite3",
        reports_dir=tmp_path,
    )
    jira = FakeJira()
    context = InvestigationContext(
        client=jira,
        issues=IssueService(jira, "TYN", "scope"),
        audit=AuditStore(settings.database_path),
        settings=settings,
        assessment_id="non-rate-limit-test",
        project_key="TYN",
        dataset_label="scope",
        target=TargetConfig(
            release="R2.4",
            commit="abc123",
            environment="production",
            service="atlas",
        ),
        requirement_id="REQ-1",
        acceptance_criteria="Verify the release.",
        manifest_policy="Use only Jira.",
        used_calls=[0],
        lock=threading.Lock(),
    )
    delays: list[float] = []
    monkeypatch.setattr("app.agent.graph.time.sleep", delays.append)

    with pytest.raises(ValueError, match="invalid request"):
        _invoke_with_rate_limit_retry(
            lambda: (_ for _ in ()).throw(ValueError("invalid request")),
            context,
            "test",
        )

    assert delays == []
    assert context.metrics["llm_calls"] == 1


def test_agent_comment_tool_reads_scoped_jira_not_dataset_json(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "audit.sqlite3",
        reports_dir=tmp_path,
    )
    audit = AuditStore(settings.database_path)
    jira = FakeJira()
    issue_service = IssueService(jira, "TYN", "scope")
    context = InvestigationContext(
        client=jira,
        issues=issue_service,
        audit=audit,
        settings=settings,
        assessment_id="assessment-test",
        project_key="TYN",
        dataset_label="scope",
        target=TargetConfig(
            release="R2.4",
            commit="abc123",
            environment="production",
            service="atlas",
        ),
        requirement_id="REQ-1",
        acceptance_criteria="Verify the release.",
        manifest_policy="Use only Jira.",
        used_calls=[0],
        lock=threading.Lock(),
    )
    context.issue_depth["TYN-1"] = 0
    comments_tool = next(tool for tool in build_tools(context) if tool.name == "get_jira_comments")

    result = comments_tool.invoke({"issue_key": "TYN-1"})

    assert result["comments"] == [
        {"comment_id": "555", "text": "Retrieved over Jira REST."}
    ]
    assert context.retrieved_comments[("TYN-1", "555")] == "Retrieved over Jira REST."
    assert audit.investigation_trace("assessment-test")[0]["tool"] == "get_jira_comments"


def test_cached_issue_retrieval_logs_both_calls_without_new_information(
    tmp_path: Path,
) -> None:
    settings = Settings(database_path=tmp_path / "audit.sqlite3", reports_dir=tmp_path)
    audit = AuditStore(settings.database_path)
    jira = FakeJira()
    context = InvestigationContext(
        client=jira,
        issues=IssueService(jira, "TYN", "scope"),
        audit=audit,
        settings=settings,
        assessment_id="cached-issue-test",
        project_key="TYN",
        dataset_label="scope",
        target=TargetConfig(
            release="R2.4",
            commit="abc123",
            environment="production",
            service="atlas",
        ),
        requirement_id="REQ-1",
        acceptance_criteria="Verify the release.",
        manifest_policy="Use only Jira.",
        used_calls=[0],
        lock=threading.Lock(),
    )
    issue_tool = next(tool for tool in build_tools(context) if tool.name == "get_jira_issue")
    context.issue_depth["TYN-1"] = 0

    issue_tool.invoke({"issue_key": "TYN-1"})
    after_first = context.information_version
    issue_tool.invoke({"issue_key": "TYN-1"})

    assert jira.issue_reads == 1
    assert context.information_version == after_first
    assert len(audit.investigation_trace("cached-issue-test")) == 2


def test_linked_evidence_crawl_fetches_direct_records_without_transitive_noise(
    tmp_path: Path,
) -> None:
    settings = Settings(
        database_path=tmp_path / "audit.sqlite3",
        reports_dir=tmp_path,
        max_recursion_depth=2,
    )

    def jira_issue(key: str, links: list[str]) -> dict[str, Any]:
        source_id = "REQ-5" if key == "TYN-5" else f"REC-{key[-1]}"
        return {
            "key": key,
            "fields": {
                "project": {"key": "TYN"},
                "labels": ["scope"],
                "summary": {
                    "TYN-1": "[REQ-1] Handoff delivery",
                    "TYN-2": "[REC-2] Handoff implementation",
                    "TYN-3": "[REC-3] Handoff QA",
                    "TYN-4": "[REC-4] Handoff delivery defect",
                    "TYN-5": "[REQ-5] Billing console regression",
                    "TYN-6": "[REC-6] Billing console regression",
                }.get(key, f"[{source_id}] {key} evidence"),
                "description": "Verification record.",
                "status": {"name": "Done"},
                "issuetype": {"name": "Task"},
                "issuelinks": [
                    {
                        "type": {"name": "Relates"},
                        "outwardIssue": {
                            "key": linked,
                            "fields": {
                                "summary": {
                                    "TYN-2": "[REC-2] Handoff implementation",
                                    "TYN-3": "[REC-3] Handoff QA",
                                    "TYN-4": "[REC-4] Handoff delivery defect",
                                    "TYN-5": "[REQ-5] Billing console regression",
                                    "TYN-6": "[REC-6] Billing console regression",
                                }.get(linked, linked)
                            },
                        },
                    }
                    for linked in links
                ],
            },
        }

    class GraphJira(FakeJira):
        def __init__(self) -> None:
            super().__init__()
            self.issue_read_keys: list[str] = []
            self.issues = {
                "TYN-1": jira_issue("TYN-1", ["TYN-2", "TYN-3"]),
                "TYN-2": jira_issue("TYN-2", ["TYN-4", "TYN-5", "TYN-6"]),
                "TYN-3": jira_issue("TYN-3", []),
                "TYN-4": jira_issue("TYN-4", []),
                "TYN-5": jira_issue("TYN-5", []),
                "TYN-6": jira_issue("TYN-6", []),
            }
            self.comments = {
                key: [
                    {
                        "id": f"comment-{key}",
                        "body": "Retrieved over Jira REST.",
                    }
                ]
                for key in self.issues
            }

        def get_jira_issue(self, issue_key: str) -> dict[str, Any]:
            self.issue_reads += 1
            self.issue_read_keys.append(issue_key)
            return self.issues[issue_key]

        def get_jira_comments(self, issue_key: str) -> list[dict[str, Any]]:
            return self.comments[issue_key]

    jira = GraphJira()
    audit = AuditStore(settings.database_path)
    context = InvestigationContext(
        client=jira,
        issues=IssueService(jira, "TYN", "scope"),
        audit=audit,
        settings=settings,
        assessment_id="linked-crawl-test",
        project_key="TYN",
        dataset_label="scope",
        target=TargetConfig(
            release="R2.4",
            commit="abc123",
            environment="production",
            service="atlas",
        ),
        requirement_id="REQ-1",
        acceptance_criteria="Verify the release.",
        manifest_policy="Use only Jira.",
        used_calls=[0],
        lock=threading.Lock(),
        root_issue_key="TYN-1",
        link_relevance_terms={"handoff", "delivery"},
    )
    context.issue_depth["TYN-1"] = 0
    context.excluded_issue_keys.add("TYN-5")

    _investigate_linked_records(context, build_tools(context), "TYN-1")

    assert context.comments_loaded == {"TYN-1", "TYN-2", "TYN-3", "TYN-4"}
    assert context.linked_issues_loaded == {"TYN-1", "TYN-2", "TYN-3"}
    assert set(context.issue_snapshots) == {
        "TYN-1", "TYN-2", "TYN-3", "TYN-4"
    }
    trace = audit.investigation_trace("linked-crawl-test")
    crawl = next(item for item in trace if item["tool"] == "linked_evidence_crawl")
    assert "Inspected comments and linked records for 4 Jira issues" in crawl[
        "result_summary"
    ]
    assert "TYN-5" not in jira.issue_read_keys
    assert "TYN-6" not in jira.issue_read_keys
    assert "TYN-4" in jira.issue_read_keys


def test_search_cache_includes_result_limit(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "audit.sqlite3", reports_dir=tmp_path)
    jira = FakeJira()
    context = InvestigationContext(
        client=jira,
        issues=IssueService(jira, "TYN", "scope"),
        audit=AuditStore(settings.database_path),
        settings=settings,
        assessment_id="search-cache-test",
        project_key="TYN",
        dataset_label="scope",
        target=TargetConfig(
            release="R2.4",
            commit="abc123",
            environment="production",
            service="atlas",
        ),
        requirement_id="REQ-1",
        acceptance_criteria="Verify the release.",
        manifest_policy="Use only Jira.",
        used_calls=[0],
        lock=threading.Lock(),
    )
    search_tool = next(
        tool for tool in build_tools(context) if tool.name == "search_jira_issues"
    )

    search_tool.invoke({"jql_query": 'summary ~ "test"', "max_results": 1})
    search_tool.invoke({"jql_query": 'summary ~ "test"', "max_results": 2})

    assert jira.search_limits == [1, 2]


def test_compact_evidence_retains_mismatched_citation_and_scenario_actor(
    tmp_path: Path,
) -> None:
    settings = Settings(database_path=tmp_path / "audit.sqlite3", reports_dir=tmp_path)
    jira = FakeJira()
    context = InvestigationContext(
        client=jira,
        issues=IssueService(jira, "TYN", "scope"),
        audit=AuditStore(settings.database_path),
        settings=settings,
        assessment_id="compact-evidence-test",
        project_key="TYN",
        dataset_label="scope",
        target=TargetConfig(
            release="R2.4",
            commit="abc123",
            environment="production",
            service="atlas",
        ),
        requirement_id="REQ-1",
        acceptance_criteria="Verify the release.",
        manifest_policy="Use only Jira.",
        used_calls=[0],
        lock=threading.Lock(),
    )
    context.register_issue(jira.get_jira_issue("TYN-1"), depth=0)
    text = (
        "Recorded event 2026-11-08T14:00:00+05:30 | actor Priya Nair | role QA Lead\n"
        "Verification failed on release R2.4; commit wrong-sha; service atlas; "
        "environment production."
    )
    validation = {
        "issue_key": "TYN-1",
        "comment_id": "555",
        "applicable": False,
        "missing_dimensions": [],
        "mismatched_dimensions": {"commit": {"expected": "abc123", "actual": "wrong-sha"}},
        "rejection_reasons": ["commit does not match target"],
    }
    context.retrieved_comments[("TYN-1", "555")] = text
    context.validation_cache[("TYN-1", "555")] = validation
    context.validations.append(validation)

    compacted = context.compact_evidence()

    assert compacted["comments"][0]["comment_id"] == "555"
    assert compacted["comments"][0]["actor"] == "Priya Nair"
    assert compacted["comments"][0]["role"] == "QA Lead"
    assert compacted["comments"][0]["recorded_at"] == "2026-11-08T14:00:00+05:30"
    assert compacted["comments"][0]["applicability"]["applicable"] is False
    assert compacted["comments"][0]["applicability"]["mismatched_dimensions"]["commit"][
        "actual"
    ] == "wrong-sha"


def test_compact_evidence_omits_non_evidentiary_comments_after_validation(
    tmp_path: Path,
) -> None:
    settings = Settings(database_path=tmp_path / "audit.sqlite3", reports_dir=tmp_path)
    jira = FakeJira()
    context = InvestigationContext(
        client=jira,
        issues=IssueService(jira, "TYN", "scope"),
        audit=AuditStore(settings.database_path),
        settings=settings,
        assessment_id="compact-irrelevant-test",
        project_key="TYN",
        dataset_label="scope",
        target=TargetConfig(
            release="R2.4",
            commit="abc123",
            environment="production",
            service="atlas",
        ),
        requirement_id="REQ-1",
        acceptance_criteria="The deployed keyboard handoff must pass QA verification.",
        manifest_policy="Use only Jira.",
        used_calls=[0],
        lock=threading.Lock(),
    )
    context.register_issue(jira.get_jira_issue("TYN-1"), depth=0)
    context.retrieved_comments[("TYN-1", "generic")] = (
        "Recorded event 2026-11-03T09:00:00+05:30 | actor Sam Patel | role Developer\n"
        "Picked this up. Will update once the change is in review."
    )
    context.retrieved_comments[("TYN-1", "evidence")] = (
        "Recorded event 2026-11-08T14:00:00+05:30 | actor Priya Nair | role QA Lead\n"
        "KBD-210: release R2.3; commit 42d09ce; service atlas; environment production. "
        "Keyboard handoff: 18 of 18 cases pass."
    )
    for comment_id, text in context.retrieved_comments.items():
        validation = {
            "issue_key": comment_id[0],
            "comment_id": comment_id[1],
            "applicable": False,
            "missing_dimensions": [],
            "mismatched_dimensions": {},
            "rejection_reasons": [],
        }
        context.validation_cache[comment_id] = validation
        context.validations.append(validation)

    compacted = context.compact_evidence()

    assert [item["comment_id"] for item in compacted["comments"]] == ["evidence"]


def test_record_investigation_accepts_escaped_json_and_logs_valid_event(
    tmp_path: Path,
) -> None:
    settings = Settings(database_path=tmp_path / "audit.sqlite3", reports_dir=tmp_path)
    audit = AuditStore(settings.database_path)
    jira = FakeJira()
    context = InvestigationContext(
        client=jira,
        issues=IssueService(jira, "TYN", "scope"),
        audit=audit,
        settings=settings,
        assessment_id="escaped-audit-test",
        project_key="TYN",
        dataset_label="scope",
        target=TargetConfig(
            release="R2.4",
            commit="abc123",
            environment="production",
            service="atlas",
        ),
        requirement_id="REQ-1",
        acceptance_criteria="Verify the release.",
        manifest_policy="Use only Jira.",
        used_calls=[0],
        lock=threading.Lock(),
    )
    note_tool = next(
        tool for tool in build_tools(context) if tool.name == "record_investigation"
    )

    result = note_tool.invoke(
        {
            "action_summary": "Inspect Jira issue",
            "arguments_json": r'{\"issue_key\": \"TYN-12\"}',
            "result_summary": "Issue comments retrieved.",
        }
    )

    assert result == {"status": "recorded"}
    trace = audit.investigation_trace("escaped-audit-test")
    assert trace[0]["arguments"]["issue_key"] == "TYN-12"
    assert trace[1]["tool"] == "record_investigation"


def test_record_investigation_logs_invalid_arguments_without_corrupting_audit(
    tmp_path: Path,
) -> None:
    settings = Settings(database_path=tmp_path / "audit.sqlite3", reports_dir=tmp_path)
    audit = AuditStore(settings.database_path)
    jira = FakeJira()
    context = InvestigationContext(
        client=jira,
        issues=IssueService(jira, "TYN", "scope"),
        audit=audit,
        settings=settings,
        assessment_id="invalid-audit-test",
        project_key="TYN",
        dataset_label="scope",
        target=TargetConfig(
            release="R2.4",
            commit="abc123",
            environment="production",
            service="atlas",
        ),
        requirement_id="REQ-1",
        acceptance_criteria="Verify the release.",
        manifest_policy="Use only Jira.",
        used_calls=[0],
        lock=threading.Lock(),
    )
    note_tool = next(
        tool for tool in build_tools(context) if tool.name == "record_investigation"
    )

    result = note_tool.invoke(
        {
            "action_summary": "Bad note",
            "arguments_json": "{not-json",
            "result_summary": "This note cannot be parsed.",
        }
    )

    assert "error" in result
    trace = audit.investigation_trace("invalid-audit-test")
    assert len(trace) == 1
    assert trace[0]["tool"] == "record_investigation"
    assert trace[0]["result_summary"].startswith("ERROR:")
