import random
import time
from typing import Any
from urllib.parse import quote

import httpx


class JiraApiError(RuntimeError):
    def __init__(self, status_code: int, message: str, response_body: str = "") -> None:
        self.status_code = status_code
        self.response_body = response_body[:2000]
        super().__init__(f"Jira API returned HTTP {status_code}: {message}")


class JiraClient:
    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        max_retries: int = 5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.metrics = {"requests": 0, "retries": 0, "errors": 0}
        self.http = httpx.Client(
            base_url=f"{self.base_url}/rest/api/3",
            auth=(email, api_token),
            timeout=None,
            headers={"Accept": "application/json"},
        )

    def close(self) -> None:
        self.http.close()

    def _request(
        self, method: str, path: str, *, params: dict | None = None, json: Any = None
    ) -> Any:
        for attempt in range(self.max_retries + 1):
            self.metrics["requests"] += 1
            try:
                response = self.http.request(method, path, params=params, json=json)
            except httpx.TransportError:
                if attempt >= self.max_retries:
                    self.metrics["errors"] += 1
                    raise
                self.metrics["retries"] += 1
                time.sleep(min(30.0, 0.5 * (2**attempt) + random.random()))
                continue
            if response.status_code == 429 or response.status_code in {500, 502, 503, 504}:
                if attempt < self.max_retries:
                    self.metrics["retries"] += 1
                    retry_after = response.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.replace(".", "", 1).isdigit()
                        else min(30.0, 0.5 * (2**attempt) + random.random())
                    )
                    time.sleep(delay)
                    continue
            if not response.is_success:
                self.metrics["errors"] += 1
                raise JiraApiError(
                    response.status_code,
                    response.reason_phrase,
                    response.text,
                )
            if response.status_code == 204 or not response.content:
                return None
            return response.json()
        raise RuntimeError("Jira retry loop ended unexpectedly")

    def ensure_project(
        self,
        project_key: str,
        project_name: str = "",
        lead_account_id: str = "",
    ) -> dict[str, Any]:
        try:
            return self._request("GET", f"/project/{quote(project_key)}")
        except JiraApiError as error:
            if error.status_code != 404:
                raise
        if not project_name or not lead_account_id:
            raise ValueError(
                f"Jira project {project_key} does not exist. Set "
                "JIRA_PROJECT_NAME and JIRA_PROJECT_LEAD_ACCOUNT_ID to create it."
            )
        return self._request(
            "POST",
            "/project",
            json={
                "key": project_key,
                "name": project_name,
                "projectTypeKey": "software",
                "projectTemplateKey": "com.pyxis.greenhopper.jira:gh-scrum-template",
                "leadAccountId": lead_account_id,
                "assigneeType": "UNASSIGNED",
            },
        )

    def search_jira_issues(
        self, jql: str, fields: list[str] | None = None, max_results: int = 100
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        token: str | None = None
        while True:
            payload: dict[str, Any] = {
                "jql": jql,
                "fields": fields or ["summary", "description", "issuetype", "status", "labels", "issuelinks"],
                "maxResults": min(max_results - len(issues), 100),
            }
            if token:
                payload["nextPageToken"] = token
            page = self._request("POST", "/search/jql", json=payload)
            issues.extend(page.get("issues", []))
            token = page.get("nextPageToken")
            if page.get("isLast", not token) or not token or len(issues) >= max_results:
                break
        return issues

    def get_jira_issue(self, issue_key: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/issue/{quote(issue_key)}",
            params={"fields": "summary,description,issuetype,status,labels,issuelinks,project"},
        )

    def get_jira_comments(self, issue_key: str) -> list[dict[str, Any]]:
        comments: list[dict[str, Any]] = []
        start_at = 0
        while True:
            page = self._request(
                "GET",
                f"/issue/{quote(issue_key)}/comment",
                params={"startAt": start_at, "maxResults": 100},
            )
            comments.extend(page.get("comments", []))
            start_at += len(page.get("comments", []))
            if start_at >= page.get("total", 0) or not page.get("comments"):
                return comments

    def create_issue(self, fields: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/issue", json={"fields": fields})

    def update_issue(self, issue_key: str, fields: dict[str, Any]) -> None:
        self._request("PUT", f"/issue/{quote(issue_key)}", json={"fields": fields})

    def get_transitions(self, issue_key: str) -> list[dict[str, Any]]:
        result = self._request("GET", f"/issue/{quote(issue_key)}/transitions")
        return result.get("transitions", [])

    def transition_issue(self, issue_key: str, transition_id: str) -> None:
        self._request(
            "POST",
            f"/issue/{quote(issue_key)}/transitions",
            json={"transition": {"id": transition_id}},
        )

    def create_comment(self, issue_key: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST", f"/issue/{quote(issue_key)}/comment", json={"body": body}
        )

    def create_issue_link(
        self, link_type: str, inward_key: str, outward_key: str
    ) -> None:
        self._request(
            "POST",
            "/issueLink",
            json={
                "type": {"name": link_type},
                "inwardIssue": {"key": inward_key},
                "outwardIssue": {"key": outward_key},
            },
        )

    def issue_types(self, project_id: str) -> list[str]:
        result = self._request(
            "GET", "/issuetype/project", params={"projectId": project_id}
        )
        return [item["name"] for item in result]
