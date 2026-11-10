import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    release: str
    commit: str | None = None
    prompt: str | None = None
    environment: str | None = None
    service: str | None = None


class DatasetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    kind: str
    summary: str
    release: str
    recorded_status: str
    description: str
    comments: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_label: str
    target: TargetConfig
    records: list[DatasetRecord]
    note: str | None = None

    @model_validator(mode="after")
    def validate_references(self) -> "Dataset":
        ids = [record.source_id for record in self.records]
        if len(set(ids)) != len(ids):
            raise ValueError("Dataset source_id values must be unique")
        known = set(ids)
        missing = [
            f"{record.source_id}->{linked_id}"
            for record in self.records
            for linked_id in record.links
            if linked_id not in known
        ]
        if missing:
            raise ValueError(f"Dataset contains unresolved links: {', '.join(missing)}")
        return self


def load_dataset(path: Path) -> Dataset:
    with path.open("r", encoding="utf-8") as dataset_file:
        raw: Any = json.load(dataset_file)
    return Dataset.model_validate(raw)


def load_dataset_bytes(content: bytes) -> Dataset:
    return Dataset.model_validate_json(content)
