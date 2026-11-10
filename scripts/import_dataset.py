import argparse
import json
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.jira.client import JiraClient
from app.jira.importer import DatasetImporter
from app.models.dataset import load_dataset
from app.utils import JsonLogFormatter


def main() -> int:
    parser = argparse.ArgumentParser(description="Import dataset records into Jira Cloud.")
    parser.add_argument("--project-key", help="Jira project key (defaults to JIRA_PROJECT_KEY).")
    parser.add_argument("--dataset", type=Path, help="Dataset JSON path (defaults to inputs/dataset.json).")
    args = parser.parse_args()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonLogFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    settings = get_settings()
    settings.require_jira()
    dataset_path = args.dataset.resolve() if args.dataset else settings.dataset_path
    dataset = load_dataset(dataset_path)
    client = JiraClient(
        settings.jira_base_url,
        settings.jira_email,
        settings.jira_api_token.get_secret_value(),
        settings.jira_max_retries,
    )
    try:
        result = DatasetImporter(
            client,
            args.project_key or settings.jira_project_key,
            dataset,
            project_name=settings.jira_project_name,
            project_lead_account_id=settings.jira_project_lead_account_id,
            link_type=settings.jira_link_type,
            source_id_field=settings.jira_source_id_field,
        ).run()
    finally:
        client.close()

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = settings.reports_dir / "source-issue-map.json"
    mapping_path.write_text(
        json.dumps(
            {
                "project_key": result.project_key,
                "dataset_label": result.dataset_label,
                "source_id_to_issue_key": result.source_id_to_issue_key,
                "errors": result.errors,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "project_key": result.project_key,
                "dataset_label": result.dataset_label,
                "total_records": result.total_records,
                "created": result.created,
                "updated": result.updated,
                "comments_created": result.comments_created,
                "links_created": result.links_created,
                "source_mapping": str(mapping_path),
                "errors": result.errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
