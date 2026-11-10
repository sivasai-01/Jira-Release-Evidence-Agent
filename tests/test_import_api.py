import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes
from app.config import Settings
from app.jira.importer import ImportResult
from app.services.audit_service import AuditStore


VALID_DATASET = {
    "dataset_label": "upload-test",
    "target": {"release": "R2.4"},
    "records": [
        {
            "source_id": "MANIFEST",
            "kind": "manifest",
            "summary": "Manifest",
            "release": "R2.4",
            "recorded_status": "To Do",
            "description": "Target release.",
            "comments": [],
            "links": [],
        }
    ],
}


def test_import_endpoint_accepts_uploaded_json_file(
    tmp_path: Path, monkeypatch
) -> None:
    settings = Settings(
        jira_base_url="https://jira.example.test",
        jira_email="user@example.test",
        jira_api_token="test-token",
        jira_project_key="TYN",
        google_api_key="test-key",
        reports_dir=tmp_path,
        database_path=tmp_path / "audit.sqlite3",
    )
    captured: dict = {}

    class FakeJiraClient:
        def __init__(self, *_: object) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeImporter:
        def __init__(self, client, project_key, dataset, **kwargs) -> None:
            captured["project_key"] = project_key
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs

        def run(self) -> ImportResult:
            return ImportResult(
                project_key=captured["project_key"],
                dataset_label=captured["dataset"].dataset_label,
                total_records=len(captured["dataset"].records),
                created=1,
                source_id_to_issue_key={"MANIFEST": "TYN-1"},
            )

    monkeypatch.setattr(routes, "JiraClient", FakeJiraClient)
    monkeypatch.setattr(routes, "DatasetImporter", FakeImporter)
    test_app = FastAPI()
    test_app.include_router(
        routes.create_router(settings, AuditStore(settings.database_path))
    )
    client = TestClient(test_app)

    response = client.post(
        "/api/import",
        files={
            "file": (
                "dataset.json",
                json.dumps(VALID_DATASET).encode("utf-8"),
                "application/json",
            )
        },
        data={"project_key": "UPLOAD"},
    )

    assert response.status_code == 200
    assert response.json()["project_key"] == "UPLOAD"
    assert response.json()["dataset_label"] == "upload-test"
    assert response.json()["source_id_to_issue_key"] == {"MANIFEST": "TYN-1"}
    assert captured["dataset"].records[0].source_id == "MANIFEST"
    saved_mapping = json.loads((tmp_path / "source-issue-map.json").read_text("utf-8"))
    assert saved_mapping["source_id_to_issue_key"] == {"MANIFEST": "TYN-1"}


def test_import_endpoint_rejects_invalid_json_and_wrong_extension(
    tmp_path: Path,
) -> None:
    settings = Settings(
        jira_base_url="https://jira.example.test",
        jira_email="user@example.test",
        jira_api_token="test-token",
        jira_project_key="TYN",
        reports_dir=tmp_path,
        database_path=tmp_path / "audit.sqlite3",
    )
    audit = AuditStore(settings.database_path)
    router_app = FastAPI()
    router_app.include_router(routes.create_router(settings, audit))
    client = TestClient(router_app)

    invalid_json = client.post(
        "/api/import",
        files={"file": ("dataset.json", b"{bad json", "application/json")},
    )
    wrong_extension = client.post(
        "/api/import",
        files={"file": ("dataset.txt", b"{}", "application/json")},
    )

    assert invalid_json.status_code == 422
    assert wrong_extension.status_code == 400


def test_import_endpoint_enforces_upload_size_limit(tmp_path: Path) -> None:
    settings = Settings(
        jira_base_url="https://jira.example.test",
        jira_email="user@example.test",
        jira_api_token="test-token",
        jira_project_key="TYN",
        max_dataset_upload_bytes=1024,
        reports_dir=tmp_path,
        database_path=tmp_path / "audit.sqlite3",
    )
    router_app = FastAPI()
    router_app.include_router(
        routes.create_router(settings, AuditStore(settings.database_path))
    )
    client = TestClient(router_app)

    response = client.post(
        "/api/import",
        files={"file": ("dataset.json", b"x" * 1025, "application/json")},
    )

    assert response.status_code == 413


def test_assessment_report_api_returns_usage_citations_and_trace(tmp_path: Path) -> None:
    settings = Settings(
        reports_dir=tmp_path,
        database_path=tmp_path / "audit.sqlite3",
    )
    audit = AuditStore(settings.database_path)
    audit.create_assessment("assessment-report-test")
    saved_report = {
        "assessment_id": "assessment-report-test",
        "dataset_label": "scope",
        "project_key": "TYN",
        "target": {
            "release": "R2.4",
            "commit": "abc123",
            "environment": "production",
            "service": "atlas",
        },
        "overall_decision": "READY",
        "total_requirements": 1,
        "pass_count": 1,
        "fail_count": 0,
        "insufficient_evidence_count": 0,
        "requirements": [
            {
                "requirement_id": "REQ-01",
                "summary": "Test requirement",
                "mandatory": True,
                "acceptance_criteria": "Must pass.",
                "decision": "PASS",
                "explanation": "Verified using target evidence.",
                "supporting_evidence": [
                    {"issue_key": "TYN-12", "comment_id": "345"}
                ],
                "rejected_evidence": [
                    {
                        "issue_key": "TYN-13",
                        "comment_id": "346",
                        "rejection_reasons": ["commit mismatch"],
                    }
                ],
                "llm_usage": {
                    "calls_detail": [
                        {
                            "prompt_characters": 100,
                            "input_tokens": 80,
                            "output_tokens": 10,
                            "total_tokens": 90,
                        }
                    ]
                },
            }
        ],
        "investigation_trace": [
            {
                "tool": "get_jira_comments",
                "duration_ms": 10,
                "result_summary": "Retrieved evidence comments.",
            }
        ],
        "execution_time_seconds": 12.5,
        "llm_model": "gemini/test-model",
        "generated_at": "2026-11-10T00:00:00+00:00",
        "llm_usage": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "calls": [
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
            ],
        },
        "metrics": {"llm_calls": 2, "llm_prompt_characters": 150},
    }
    audit.save_report("assessment-report-test", saved_report)
    (tmp_path / "assessment-report-test.md").write_text("stale report", encoding="utf-8")
    app = FastAPI()
    app.include_router(routes.create_router(settings, audit))
    client = TestClient(app)

    report_response = client.get("/api/assessment/assessment-report-test/report")
    status_response = client.get("/api/assessment/assessment-report-test")
    markdown_response = client.get(
        "/api/assessment/assessment-report-test/report?format=markdown"
    )

    assert report_response.status_code == 200
    usage = report_response.json()["llm_usage"]
    assert usage["input_tokens"] == 80
    assert usage["output_tokens"] == 10
    assert usage["total_tokens"] == 90
    assert usage["token_usage_status"] == "partial"
    assert usage["token_usage_reported_calls"] == 1
    assert usage["token_usage_unavailable_calls"] == 1
    assert report_response.json()["requirements"][0]["llm_usage"]["input_tokens"] == 80
    assert report_response.json()["requirements"][0]["supporting_evidence"][0][
        "issue_key"
    ] == "TYN-12"
    assert report_response.json()["investigation_trace"][0]["tool"] == "get_jira_comments"
    assert status_response.status_code == 200
    assert status_response.json()["report"]["llm_usage"]["token_usage_status"] == "partial"
    assert markdown_response.status_code == 200
    assert "Reported input tokens: 80" in markdown_response.text
    assert "Token usage coverage: 1 / 2 calls (partial)" in markdown_response.text
    assert "TYN-13" in markdown_response.text
    assert "commit mismatch" in markdown_response.text
