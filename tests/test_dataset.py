from pathlib import Path

from app.models.dataset import load_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_full_dataset_schema_and_relationship_integrity() -> None:
    dataset = load_dataset(ROOT / "inputs" / "dataset.json")
    assert len(dataset.records) == 400
    assert sum(len(record.comments) for record in dataset.records) == 2059
    assert sum(len(record.links) for record in dataset.records) == 76
    assert sum(record.kind == "requirement" for record in dataset.records) == 12
    assert dataset.target.release == "R2.4"
