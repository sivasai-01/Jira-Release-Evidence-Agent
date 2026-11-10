import httpx

from app.jira.client import JiraClient


def test_retries_rate_limit_and_transport_failures(monkeypatch) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, request=request)
        if attempts == 2:
            raise httpx.ConnectError("temporary network failure", request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    monkeypatch.setattr("app.jira.client.time.sleep", lambda _: None)
    client = JiraClient("https://jira.invalid", "user", "token", max_retries=3)
    client.http.close()
    client.http = httpx.Client(
        base_url="https://jira.invalid/rest/api/3",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert client._request("GET", "/probe") == {"ok": True}
        assert attempts == 3
    finally:
        client.close()


def test_jira_client_has_no_request_deadline() -> None:
    client = JiraClient("https://jira.invalid", "user", "token")
    try:
        assert client.http.timeout.connect is None
        assert client.http.timeout.read is None
        assert client.http.timeout.write is None
        assert client.http.timeout.pool is None
    finally:
        client.close()
