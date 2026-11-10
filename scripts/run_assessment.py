import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.models.dataset import TargetConfig
from app.services.audit_service import AuditStore
from app.services.execution_service import execute_assessment
from app.utils import JsonLogFormatter


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Jira-backed release assessment.")
    parser.add_argument("--project-key", help="Jira project key (defaults to JIRA_PROJECT_KEY).")
    parser.add_argument("--dataset-label", help="Dataset Jira label.")
    parser.add_argument("--release", required=True, help="Target release, e.g. R2.4.")
    parser.add_argument("--commit", help="Target commit, required when overriding the manifest.")
    parser.add_argument("--service", help="Target service, required when overriding the manifest.")
    parser.add_argument("--environment", help="Target environment, required when overriding the manifest.")
    parser.add_argument("--prompt", help="Target prompt version.")
    args = parser.parse_args()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonLogFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    settings = get_settings()
    settings.require_jira()
    settings.require_llm()

    override_values = (args.commit, args.service, args.environment, args.prompt)
    if any(value is not None for value in override_values):
        if not all((args.commit, args.service, args.environment)):
            parser.error("--commit, --service, and --environment must be supplied together")
        target = TargetConfig(
            release=args.release,
            commit=args.commit,
            service=args.service,
            environment=args.environment,
            prompt=args.prompt,
        )
    else:
        target = None

    assessment_id = str(uuid.uuid4())
    audit = AuditStore(settings.database_path)
    audit.create_assessment(assessment_id)
    audit.set_status(assessment_id, "running")
    try:
        report = execute_assessment(
            assessment_id=assessment_id,
            project_key=args.project_key or settings.jira_project_key,
            dataset_label=args.dataset_label or settings.jira_dataset_label,
            target_release=args.release,
            requested_target=target,
            settings=settings,
            audit=audit,
        )
    except Exception as error:
        audit.set_status(
            assessment_id, "failed", error=f"{type(error).__name__}: {error}"
        )
        logging.getLogger(__name__).exception("Assessment %s failed", assessment_id)
        return 1

    print(
        json.dumps(
            {
                "assessment_id": assessment_id,
                "overall_decision": report.overall_decision.value,
                "json_report": str(settings.reports_dir / f"{assessment_id}.json"),
                "markdown_report": str(settings.reports_dir / f"{assessment_id}.md"),
                "execution_time_seconds": report.execution_time_seconds,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
