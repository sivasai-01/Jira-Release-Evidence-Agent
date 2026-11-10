import json
from pathlib import Path

import pytest

from app.services.execution_service import save_assessment_report_files


def test_new_report_becomes_final_and_previous_final_moves_to_development(
    tmp_path: Path,
) -> None:
    save_assessment_report_files(
        tmp_path, "first", '{"assessment_id":"first"}', "# First report"
    )

    save_assessment_report_files(
        tmp_path, "second", '{"assessment_id":"second"}', "# Second report"
    )

    final_dir = tmp_path / "final"
    development_dir = tmp_path / "development"
    assert json.loads((final_dir / "second.json").read_text("utf-8")) == {
        "assessment_id": "second"
    }
    assert (final_dir / "second.md").read_text("utf-8") == "# Second report"
    assert json.loads((development_dir / "first.json").read_text("utf-8")) == {
        "assessment_id": "first"
    }
    assert (development_dir / "first.md").read_text("utf-8") == "# First report"
    assert sorted(path.name for path in final_dir.iterdir()) == [
        "second.json",
        "second.md",
    ]


def test_report_storage_does_not_overwrite_archived_report_names(
    tmp_path: Path,
) -> None:
    save_assessment_report_files(
        tmp_path, "first", '{"assessment_id":"first"}', "# First report"
    )
    development_dir = tmp_path / "development"
    (development_dir / "first.json").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_assessment_report_files(
            tmp_path, "second", '{"assessment_id":"second"}', "# Second report"
        )

    assert (tmp_path / "final" / "first.json").is_file()
    assert (development_dir / "first.json").read_text("utf-8") == "preserve"
