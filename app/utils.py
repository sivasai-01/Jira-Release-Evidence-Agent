import json
import logging
import re
from datetime import datetime, timezone
from typing import Any


def adf_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    if value.get("type") == "text":
        return str(value.get("text", ""))
    pieces: list[str] = []
    for child in value.get("content", []):
        child_text = adf_to_text(child)
        if child.get("type") in {"paragraph", "heading", "blockquote", "listItem"}:
            if pieces and pieces[-1] != "\n":
                pieces.append("\n")
        pieces.append(child_text)
    return "".join(pieces).strip()


def text_to_adf(text: str) -> dict[str, Any]:
    paragraphs = text.splitlines() or [""]
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": line}] if line else [],
            }
            for line in paragraphs
        ],
    }


def parse_target(text: str) -> dict[str, str | None]:
    match = re.search(r"(?im)^\s*Target\s*:\s*", text)
    if not match:
        return {}
    try:
        target, _ = json.JSONDecoder().raw_decode(text[match.end() :].lstrip())
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(target, dict):
        return {}
    return {str(key): str(value) if value is not None else None for key, value in target.items()}


EVIDENCE_DIMENSIONS = {
    "release": r"\brelease\s*[:=]?\s*([A-Za-z0-9._-]+)",
    "commit": r"\bcommit\s*[:=]?\s*([A-Za-z0-9._-]+)",
    "service": r"\bservice\s*[:=]?\s*([A-Za-z0-9._-]+)",
    "environment": r"\b(?:environment|approval\s+scope)\s*[:=]?\s*([A-Za-z0-9._-]+)",
    "prompt": r"\bprompt(?:\s+(?:version|bundle))?\s*[:=]?\s*(p\d+)\b",
}


NON_VALUE_TOKENS = {
    "and",
    "build",
    "checklist",
    "for",
    "from",
    "is",
    "of",
    "on",
    "review",
    "scope",
    "sign-off",
    "the",
    "to",
    "version",
}


def parse_evidence_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {
        key: value
        for key, value in parse_target(text).items()
        if value is not None and key in EVIDENCE_DIMENSIONS
    }
    for name, pattern in EVIDENCE_DIMENSIONS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).rstrip(".,;)")
            if value and value.casefold() not in NON_VALUE_TOKENS:
                metadata[name] = value
    return metadata


RECORDED_EVENT = re.compile(
    r"^\s*Recorded event\s+([^|]+)\|\s*actor\s+([^|]+)\|\s*role\s+([^\r\n]+)\s*(?:\r?\n|$)",
    re.IGNORECASE | re.MULTILINE,
)


def parse_recorded_event(text: str) -> dict[str, str | None]:
    match = RECORDED_EVENT.search(text)
    if not match:
        return {"recorded_at": None, "actor": None, "role": None}
    return {
        "recorded_at": match.group(1).strip(),
        "actor": match.group(2).strip(),
        "role": match.group(3).strip(),
    }


def source_label(source_id: str) -> str:
    return f"source-id-{source_id}"


def jql_literal(value: str) -> str:
    if any(ord(character) < 32 for character in value):
        raise ValueError("JQL values cannot contain control characters")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def issue_text(issue: dict[str, Any]) -> str:
    fields = issue.get("fields", {})
    return "\n".join(
        part
        for part in (
            str(fields.get("summary", "")),
            adf_to_text(fields.get("description")),
        )
        if part
    )
