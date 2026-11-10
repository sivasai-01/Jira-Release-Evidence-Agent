# Development Iterations

This document describes the capability milestones represented by the current
implementation and its regression tests. It is a functional summary, not a
dated commit-by-commit history.

## Capability milestones

### 1. Dataset and configuration foundation

Define validated dataset, target, request, and report schemas; load settings
from environment variables; and provide a JSON fixture for offline tests.

**Covered by:** `tests/test_config.py`, `tests/test_dataset.py`.

### 2. Jira access and repeatable import

Add a Jira REST client and an importer that maps source records to Jira issues,
preserves source identifiers, imports comments and relationships, and avoids
duplicating existing records on reruns. Persist the source-ID-to-issue-key map
for reference.

**Covered by:** `tests/test_jira_client.py`, `tests/test_importer.py`.

### 3. API and assessment lifecycle

Expose dataset import and assessment endpoints. Track queued, running,
completed, and failed assessment states, and retain reports and investigation
events in SQLite.

**Covered by:** `tests/test_import_api.py`.

### 4. Jira requirement discovery and evidence investigation

Discover the release manifest and its linked requirements in Jira, then
investigate requirements with a bounded LangGraph agent and structured Jira
tools. Follow relevant linked records while limiting calls, recursion, and
context.

**Covered by:** `tests/test_requirement_discovery.py`,
`tests/test_agent_tools.py`.

### 5. Evidence validation and decision policy

Validate evidence scope and authority independently of the language model.
Handle target mismatches, failed or inconclusive results, explicit retests,
unresolved contradictions, and requirement-level decision aggregation.

**Covered by:** `tests/test_evidence_validator.py`,
`tests/test_decision_engine.py`.

### 6. Assessment reports and regression validation

Produce structured JSON and readable Markdown reports with decision
explanations, Jira citations, rejected evidence, investigation traces, and
reported provider usage. Add release-scenario coverage for the end-to-end
assessment behavior.

**Covered by:** `tests/test_release_scenario.py` and the broader test suite.
The current offline validation was **74 passed**; tests use fakes and fixtures
and do not call Jira or an LLM provider.

## Report organization

Report files are grouped by their role in the submitted assessment:

- `reports/final/` contains the designated final R2.4 report pair:
  `22fab5b9-7683-4911-81e9-7596d5168053.json` and
  `22fab5b9-7683-4911-81e9-7596d5168053.md`.
- `reports/development/` contains the other generated JSON and Markdown
  assessment runs for R2.4. These are retained as development and comparison
  artifacts rather than discarded.
- The SQLite database and `source-issue-map.json` remain directly under
  `reports/`.

When an assessment completes, its JSON and Markdown pair is written to
`reports/final/`. Any previous report files there are moved to
`reports/development/`, so the newest completed run is the final pair and prior
runs remain available. The database and issue map stay at the `reports/` root.

## Assessment outcome

The final report covers 12 mandatory requirements and records 4 `PASS`,
4 `FAIL`, and 4 `INSUFFICIENT_EVIDENCE` decisions. The overall decision is
`BLOCKED`. This reflects the evidence-based policy: confirmed mandatory
failures block release, and unresolved evidence gaps are not treated as passes.
