# Yellow Network – Jira Release Evidence Agent

## 1. Project Overview

The Jira Release Evidence Agent assesses release readiness using evidence stored in Jira Cloud. It imports the supplied dataset into a configured Jira project, discovers requirements from the Jira release manifest, investigates linked Jira issues and comments, validates evidence against the requested target, and generates a structured JSON report and readable Markdown report.

The dataset JSON is the import source only. The assessment agent retrieves evidence through Jira REST API tools and does not send the complete dataset to the LLM.

The supplied dataset contains 400 records and 2,059 scenario comments: one release manifest, 12 mandatory requirements, 185 work records, and 202 evidence records. The manifest links the release requirements and carries assessment policy.

## 2. Architecture and Technology Stack

- **Python 3.11+** and **FastAPI** for the application and API.
- **LangGraph** for the bounded, tool-using investigation workflow.
- **Google Gemini** or **Groq-compatible chat models** for tool selection and structured evidence analysis.
- **Jira Cloud REST API v3** for issue, comment, link, and dataset access.
- **LangChain structured tools** for Jira search, issue/comment retrieval, link traversal, requirement inspection, evidence validation, and investigation audit.
- **Pydantic** for configuration, dataset, API, and report schemas.
- **SQLite** for assessment state and investigation traces.
- **Pytest** for offline tests using fixtures and fakes.

All direct dependencies are pinned in [`requirements.txt`](requirements.txt). The configuration template is [`.env.example`](.env.example).

### Data and evidence preservation

The importer maps each source record to a Jira issue, selects an available kind-appropriate issue type, stores the source ID in a stable summary prefix and label, and applies the dataset label to every imported issue. It imports comments with the fictional actor, role, timestamp, and a source marker in their text; these are not represented as Jira's native author or creation time. Source relationships are imported using the configured Jira link type, `Relates` by default, because the dataset does not identify a relationship type. The importer supports reruns without duplicate issues, comments, or links and writes the source-ID-to-issue-key map to `reports/source-issue-map.json`.

## 3. Setup and Installation

### Prerequisites

- Python 3.11 or later.
- Jira Cloud access with permissions to create/update project issues, comments, and links (or an existing configured project).
- A Jira API token.
- An API key for the selected Gemini or Groq-compatible model provider.

### Install

From the project root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set:

- `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`
- `JIRA_PROJECT_KEY` and, if the importer must create the project, `JIRA_PROJECT_NAME` and `JIRA_PROJECT_LEAD_ACCOUNT_ID`
- `LLM_PROVIDER` and the selected provider key (`GOOGLE_API_KEY` for Gemini or `GROQ_API_KEY` for Groq)

Gemini is the default provider. Its model is configured by `GOOGLE_MODEL`; Groq-compatible settings are `GROQ_BASE_URL` and `GROQ_MODEL`. The legacy `GEMINI_API_KEY` variable is accepted. Optional investigation limits and retries are also configurable in `.env.example`. Never commit `.env` or API credentials.

## 4. Jira Dataset Import

The default dataset is [`inputs/dataset.json`](inputs/dataset.json). Import it into the configured Jira project with:

```powershell
python scripts\import_dataset.py
```

Override the project or dataset file when needed:

```powershell
python scripts\import_dataset.py --project-key SCRUM --dataset inputs\dataset.json
```

The importer validates and loads the dataset, creates or uses the project, imports issues/comments/relationships/statuses, and prints an import summary. It writes `reports/source-issue-map.json`. The importer exits unsuccessfully and reports errors if a record, comment, relationship, or recorded status cannot be imported. Rerunning the import updates matching source-ID issues and avoids duplicating source-marked comments and links.

The API also supports dataset upload through `POST /api/import` as a multipart upload with field `file`; `project_key` is optional if `JIRA_PROJECT_KEY` is configured. Upload size is limited by `MAX_DATASET_UPLOAD_BYTES` (10 MB by default).

## 5. Running the Assessment Agent

Run the CLI assessment for R2.4:

```powershell
python scripts\run_assessment.py --release R2.4
```

The project key and dataset label default to `JIRA_PROJECT_KEY` and `JIRA_DATASET_LABEL`. The target configuration is read from the Jira manifest and checked against the requested release. Each completed assessment becomes the final report pair in `reports/final/{assessment_id}.json` and `reports/final/{assessment_id}.md`. Any existing report pair in `reports/final/` is moved to `reports/development/` before the new pair is published. The assessment database and investigation trace remain under the configured report/database paths.

To override the target, supply `--commit`, `--service`, and `--environment` together; `--prompt` is optional:

```powershell
python scripts\run_assessment.py --release R2.5 `
  --commit commit-sha --service atlas-assistant `
  --environment production --prompt p8
```

Run the API server with:

```powershell
python -m uvicorn app.main:app --reload
```

API endpoints:

- `GET /health` — health check.
- `POST /api/import` — import a dataset upload.
- `POST /api/assessment/run` — queue an assessment. Example JSON request body:

  ```json
  {
    "project_key": "SCRUM",
    "dataset_label": "tyn-release-assessment-v2",
    "target_release": "R2.4"
  }
  ```

- `GET /api/assessment/{assessment_id}` — assessment status and report when complete.
- `GET /api/assessment/{assessment_id}/report` — JSON report; add `?format=markdown` for Markdown.

GET requests do not have a request payload. Use the assessment ID returned by `POST /api/assessment/run` in these URLs. For the submitted report ID:

```powershell
$assessmentId = "22fab5b9-7683-4911-81e9-7596d5168053"

# Check assessment status (and report details if completed)
curl.exe "http://127.0.0.1:8000/api/assessment/$assessmentId"

# Retrieve the structured JSON report
curl.exe "http://127.0.0.1:8000/api/assessment/$assessmentId/report"

# Retrieve the readable Markdown report
curl.exe "http://127.0.0.1:8000/api/assessment/$assessmentId/report?format=markdown"
```

The report endpoint returns `409` until the assessment status is `completed`.

## 6. Assessment Workflow and Decision Logic

1. Discover the dataset-scoped Jira release manifest and its linked requirements.
2. Read each requirement and its acceptance criterion from Jira.
3. Use a LangGraph agent to dynamically investigate relevant Jira issues, comments, and linked evidence with bounded tool calls and recursion.
4. Validate evidence against the target release, commit, service, environment, and prompt version when required.
5. Apply deterministic checks for evidence authority, superseding retests, contradictory results, failures, defects, and security approval scope. Jira status and developer implementation claims are not verification.
6. Record tool calls, arguments, result summaries, and timings in the investigation trace. The report includes exact Jira citations, applicability information, rejected evidence, model usage where reported, and execution metrics.

Each requirement receives exactly one decision:

- **PASS** — applicable, qualified evidence establishes the acceptance criterion.
- **FAIL** — applicable evidence establishes a failure or policy violation.
- **INSUFFICIENT_EVIDENCE** — evidence is missing, inapplicable, aborted, contradictory, or inconclusive.

Overall aggregation is deterministic: a mandatory `FAIL` yields `BLOCKED`; absent a mandatory failure, any mandatory `INSUFFICIENT_EVIDENCE` yields `NEEDS_REVIEW`; `READY` requires all mandatory requirements to pass.

## 7. Sample Assessment Results

The submitted actual Jira-backed R2.4 report is available in both formats:

- [Release-readiness report (Markdown)](reports/final/22fab5b9-7683-4911-81e9-7596d5168053.md)
- [Release-readiness report (JSON)](reports/final/22fab5b9-7683-4911-81e9-7596d5168053.json)

The JSON report contains the full **370-event investigation trace**, including Jira tool calls, arguments, results, and durations. The Markdown report includes a readable investigation-trace summary and, for each requirement, its decision, explanation, supporting Jira issue/comment citations, rejected evidence, applicability, and outstanding verification.

### Final R2.4 result

Target: release `R2.4`, commit `b7a91f2`, service `atlas-assistant`, environment `production`, prompt `p7`; Jira project `SCRUM`; dataset label `tyn-release-assessment-v2`.

| Requirement | Decision | Supporting Jira evidence / reason |
|---|---|---|
| REQ-01 | PASS | SCRUM-320 comment 11578 |
| REQ-02 | FAIL | SCRUM-310 comment 11526 |
| REQ-03 | INSUFFICIENT_EVIDENCE | SCRUM-218 comment 11068 is p7 but for the wrong commit/staging; SCRUM-30 comment 10127 uses p6, and no fully target-scoped p7 result was established |
| REQ-04 | PASS | SCRUM-154 comment 10747 |
| REQ-05 | FAIL | SCRUM-228 comment 11116 |
| REQ-06 | INSUFFICIENT_EVIDENCE | SCRUM-184 comment 10897 is staging-only; SCRUM-123 comment 10593 is production sign-off by a role other than the named Security Owner |
| REQ-07 | PASS | SCRUM-400 comment 12037 |
| REQ-08 | FAIL | SCRUM-80 comment 10378 |
| REQ-09 | INSUFFICIENT_EVIDENCE | Conflicting authoritative QA results remain unresolved |
| REQ-10 | PASS | SCRUM-13 comment 10044; successful PARSE-622 superseding retest |
| REQ-11 | INSUFFICIENT_EVIDENCE | SCRUM-106 comment 10508 is historical; SCRUM-263 comments 11290–11294 lack target dimensions, so no target-scoped QA verification was established |
| REQ-12 | FAIL | SCRUM-322 comment 11648 |

**Overall: BLOCKED** — 12 mandatory requirements; 4 `PASS`, 4 `FAIL`, and 4 `INSUFFICIENT_EVIDENCE`.

Measured metrics from this report:

- Execution time: **213.39 seconds**
- Model: `gemini/gemini-3.5-flash-lite`
- LLM calls: **45**
- Input tokens: **82,028**
- Output tokens: **4,025**
- Total tokens: **86,053**
- Token usage coverage: **45/45 calls (complete)**
- Prompt characters: **216,680**
- Jira API calls: **84**

## 8. Challenges Faced and Solutions

- **Quota and model variability:** Multiple providers were tested. Provider rate limits, quota exhaustion, availability, and latency motivated bounded retries that honor provider retry delays and explicit reporting when usage metadata is unavailable.
- **Repeated LLM context and token use:** LangGraph history and repeated evidence could inflate prompts. The workflow retains only the current requirement/target and the latest complete tool-call batch, caches repeated Jira reads and validations, deduplicates evidence, and compacts final decision context without removing citation or applicability details.
- **Evidence scope and authority:** A matching release label alone is insufficient. Validation checks all applicable target dimensions; deterministic logic distinguishes QA verification from implementation claims, handles superseding and conflicting evidence, and requires the named Security Owner and production scope for security approval.
- **Linked-record coverage:** Traversal follows relevant Jira relationships and dependencies while filtering unrelated transitive records. Tool calls and their results remain auditable.
- **Provider token reporting:** Usage is captured from provider response metadata where available, aggregated from per-call data, and marked unavailable rather than estimated when no usage is returned.

## 9. Design Choices, Known Limitations, and AI Assistance

### Design Choices, Known Limitations, Challenges Faced, and AI Assistance
- The solution was designed using Python, LangGraph, Jira REST APIs, and LLM-based agents to dynamically investigate release requirements, linked Jira issues, comments, dependencies, and verification evidence. Instead of passing the complete JSON dataset directly to the LLM, all records were imported into a dedicated Jira project, and the agent retrieves relevant information dynamically through Jira tools. This ensures that assessment decisions are based on Jira records and their relationships.
- During development and testing, we experimented with multiple LLM models, including Gemini 3.8 Flash, Gemini 3.5 Flash Lite, and Qwen 3.8 27B through Groq. We encountered API quota exhaustion, rate-limit errors, model availability issues, and inconsistent response latency. To improve reliability, we implemented retry mechanisms with controlled waiting periods, configurable timeouts, token limits, and error handling for model and Jira API calls.
- One of the major challenges was **high token consumption and repeated context transmission across LangGraph execution cycles**. The agent was repeatedly sending conversation history, manifest information, and previously retrieved evidence to the LLM, resulting in excessive input tokens and longer execution times. We optimized the workflow by reducing repeated conversation history, avoiding unnecessary context transmission, deduplicating retrieved comments and evidence, and improving token-usage tracking across LLM calls. These changes reduced the latest assessment run to 86,053 total tokens across 45 LLM calls, with usage successfully captured for all 45 calls.
- Another major challenge was **accurate evidence validation against the complete target configuration**. Initially, evidence could appear relevant based on its release number while referring to an incorrect commit, service, environment, or deployed prompt version. We strengthened the validation process to check all applicable target dimensions: release, commit, service, environment, and prompt version. This helped identify cases where evaluations used an older prompt version or evidence belonged to staging or a different commit rather than the target production deployment.
- We also encountered challenges in distinguishing verified results from developer implementation claims, Jira issue statuses, and approval statements. We refined the investigation and decision logic to prioritize applicable QA verification results, identify explicitly superseded evidence, detect unresolved contradictory results, and validate approval authority and scope. For example, a later QA retest explicitly superseding an earlier failed result is considered authoritative, whereas equally authoritative contradictory results without a resolution remain `INSUFFICIENT_EVIDENCE`. Similarly, a security approval for staging cannot be treated as production approval, and an approval from an unauthorized role cannot satisfy a security requirement.
- Another challenge involved **investigating deeply linked Jira records without overlooking relevant dependencies or introducing unrelated evidence**. We improved the Jira investigation tools and linked-issue traversal so that the agent could discover related issues, inspect comments, follow evidence references, and validate applicability before using evidence in a decision. We also improved investigation trace generation to capture actual tool calls, Jira issue and comment references, retrieved evidence, rejected evidence, and validation outcomes.
- During report validation, we identified issues involving incomplete prompt-version extraction, missing or incorrectly scoped verification evidence, and inconsistent token-usage reporting. We refined the parsing and validation logic, improved provider-specific token usage extraction, and added regression tests to prevent previously identified issues from recurring. The test suite currently has 74 passing tests, including regression coverage for report storage, agent tools, and token accounting.
- The final assessment evaluates all 12 mandatory requirements and produces four `PASS`, four `FAIL`, and four `INSUFFICIENT_EVIDENCE` decisions, resulting in an overall `BLOCKED` status. This outcome is intentional: the agent does not assume readiness when mandatory requirements have confirmed failures or unresolved evidence gaps.
- AI assistance, including GitHub Copilot and ChatGPT, was used throughout development for code generation, debugging, prompt engineering, architecture refinement, optimization, and test-case design. AI-generated suggestions were reviewed and incorporated into the implementation where appropriate. Assessment decisions are grounded in Jira evidence, with deterministic evidence validation and aggregation logic used to reduce unsupported LLM conclusions.
- **Known Limitations:** The solution depends on third-party LLM availability, API quotas, rate limits, and response latency. Investigations involving large numbers of linked issues and comments can still increase execution time and token consumption. Ambiguous, contradictory, or incomplete Jira evidence may require human review. The system deliberately uses `INSUFFICIENT_EVIDENCE` rather than assuming a requirement has passed. The assessment is also limited to evidence available within the imported Jira dataset and does not independently verify external systems or production environments.

## 10. Testing and Validation

Run the automated test suite from the project root:

```powershell
python -m pytest -q
```

The latest validation completed with **74 passed**. Tests use local fixtures and fakes; they do not call Jira, Gemini, or Groq. A live assessment requires valid Jira and LLM credentials and is measured separately.

## 11. Future Improvements

- Add a durable external queue for multi-worker API deployments.
- Add richer relationship types to the source dataset/import mapping instead of representing every source link as `Relates`.
- Add a human review workflow for ambiguous or conflicting evidence.
- Continue evaluating model/provider reliability and reducing prompt size while preserving evidence coverage and auditability.
