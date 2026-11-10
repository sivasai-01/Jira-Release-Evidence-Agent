import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS assessments (
                    assessment_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    error TEXT,
                    report_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS investigation_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    assessment_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    result_summary TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def create_assessment(self, assessment_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO assessments VALUES (?, ?, ?, ?, ?, ?)",
                (assessment_id, "queued", None, None, now, now),
            )

    def set_status(
        self, assessment_id: str, status: str, *, error: str | None = None
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE assessments SET status=?, error=?, updated_at=? WHERE assessment_id=?",
                (status, error, now, assessment_id),
            )

    def save_report(self, assessment_id: str, report: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE assessments SET status='completed', error=NULL, report_json=?, "
                "updated_at=? WHERE assessment_id=?",
                (json.dumps(report, ensure_ascii=False), now, assessment_id),
            )

    def get_assessment(self, assessment_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM assessments WHERE assessment_id=?", (assessment_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "assessment_id": row["assessment_id"],
            "status": row["status"],
            "error": row["error"],
            "report": json.loads(row["report_json"]) if row["report_json"] else None,
        }

    def record_tool(
        self,
        assessment_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result_summary: str,
        duration_ms: float,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO investigation_events
                (assessment_id, tool_name, arguments_json, result_summary, duration_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    assessment_id,
                    tool_name,
                    json.dumps(arguments, ensure_ascii=False, default=str),
                    result_summary[:1000],
                    duration_ms,
                    now,
                ),
            )

    def investigation_trace(self, assessment_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT tool_name, arguments_json, result_summary, duration_ms, created_at "
                "FROM investigation_events WHERE assessment_id=? ORDER BY id",
                (assessment_id,),
            ).fetchall()
        return [
            {
                "tool": row["tool_name"],
                "arguments": json.loads(row["arguments_json"]),
                "result_summary": row["result_summary"],
                "duration_ms": row["duration_ms"],
                "timestamp": row["created_at"],
            }
            for row in rows
        ]
