from app.models.report import AssessmentReport


def render_markdown(report: AssessmentReport) -> str:
    target = report.target
    metric = lambda value: value if value is not None else "unavailable"
    lines = [
        f"# Release readiness: {target.release}",
        "",
        f"- **Overall decision:** {report.overall_decision.value}",
        f"- **Jira project / dataset:** {report.project_key} / `{report.dataset_label}`",
        f"- **Commit:** {target.commit or 'not specified'}",
        f"- **Service:** {target.service or 'not specified'}",
        f"- **Environment:** {target.environment or 'not specified'}",
        f"- **Prompt:** {target.prompt or 'not specified'}",
        f"- **Requirements:** {report.total_requirements}",
        f"- **PASS / FAIL / INSUFFICIENT_EVIDENCE:** {report.pass_count} / "
        f"{report.fail_count} / {report.insufficient_evidence_count}",
        f"- **Execution time:** {report.execution_time_seconds:.2f} seconds",
        f"- **Model:** {report.llm_model}",
        "",
        "## Requirement decisions",
    ]
    for requirement in report.requirements:
        lines.extend(
            [
                "",
                f"### {requirement.requirement_id} — {requirement.summary}",
                f"- **Mandatory:** {'yes' if requirement.mandatory else 'no'}",
                f"- **Decision:** {requirement.decision.value}",
                f"- **LLM calls / prompt characters:** "
                f"{requirement.llm_usage.get('calls', 'unavailable')} / "
                f"{requirement.llm_usage.get('prompt_characters', 'unavailable')}",
                f"- **Reported LLM input / output tokens:** "
                f"{metric(requirement.llm_usage.get('input_tokens'))} / "
                f"{metric(requirement.llm_usage.get('output_tokens'))}",
                f"- **Acceptance criterion:** {requirement.acceptance_criteria}",
                f"- **Explanation:** {requirement.explanation}",
                "- **Supporting Jira comments:**",
            ]
        )
        if requirement.supporting_evidence:
            lines.extend(
                f"  - `{ref.issue_key}` comment `{ref.comment_id}`"
                + (f": “{ref.quote}”" if ref.quote else "")
                for ref in requirement.supporting_evidence
            )
        else:
            lines.append("  - None validated.")
        if requirement.rejected_evidence:
            lines.append("- **Rejected / inapplicable Jira comments:**")
            for detail in requirement.rejected_evidence:
                reasons = "; ".join(detail.get("rejection_reasons", []))
                lines.append(
                    f"  - `{detail.get('issue_key', 'unknown')}` comment "
                    f"`{detail.get('comment_id', 'unknown')}`: "
                    f"{reasons or 'not applicable to this requirement'}"
                )
        if requirement.applicability:
            lines.append("- **Applicability:**")
            for detail in requirement.applicability:
                lines.append(
                    f"  - `{detail['issue_key']}` comment `{detail['comment_id']}`: "
                    f"{detail.get('metadata_found', {})}"
                )
        if requirement.outstanding_verification:
            lines.append("- **Outstanding verification:**")
            lines.extend(f"  - {item}" for item in requirement.outstanding_verification)
        if requirement.clarification_required:
            lines.append("- **Clarification required:**")
            lines.extend(f"  - {item}" for item in requirement.clarification_required)
    lines.extend(["", "## Investigation trace"])
    if report.investigation_trace:
        for event in report.investigation_trace:
            lines.append(
                f"- `{event['tool']}` ({event['duration_ms']:.0f} ms): "
                f"{event['result_summary']}"
            )
    else:
        lines.append("- No Jira investigation tools were executed.")
    lines.extend(
        [
            "",
            "## Model usage",
            f"- Reported input tokens: {metric(report.llm_usage.get('input_tokens'))}",
            f"- Reported output tokens: {metric(report.llm_usage.get('output_tokens'))}",
            f"- Reported total tokens: {metric(report.llm_usage.get('total_tokens'))}",
            f"- Token usage coverage: "
            f"{report.llm_usage.get('token_usage_reported_calls', 0)} / "
            f"{report.metrics.get('llm_calls', 0)} calls "
            f"({report.llm_usage.get('token_usage_status', 'unavailable')})",
            f"- LLM calls: {report.metrics.get('llm_calls', 'unavailable')}",
            f"- Prompt characters: {report.metrics.get('llm_prompt_characters', 'unavailable')}",
            "",
        ]
    )
    return "\n".join(lines)
