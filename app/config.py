from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    jira_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("JIRA_BASE_URL", "JIRA_URL"),
    )
    jira_email: str = ""
    jira_api_token: SecretStr = SecretStr("")
    jira_project_key: str = ""
    jira_project_name: str = ""
    jira_project_lead_account_id: str = ""
    jira_dataset_label: str = "tyn-release-assessment-v2"
    jira_link_type: str = "Relates"
    jira_source_id_field: str = ""
    jira_max_retries: int = Field(default=5, ge=0, le=10)

    google_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    google_model: str = "gemini-3.8-flash"
    llm_provider: Literal["gemini", "groq"] = "gemini"
    llm_rate_limit_retries: int = Field(default=3, ge=0, le=10)
    groq_api_key: SecretStr = SecretStr("")
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "qwen/qwen3.8-27b"
    max_tool_calls: int = Field(
        default=160,
        ge=1,
        le=500,
        description="Maximum Jira/agent tool calls allowed for each requirement.",
    )
    max_recursion_depth: int = Field(default=3, ge=0, le=10)
    max_context_chars: int = Field(default=30000, ge=1000, le=100000)
    max_output_tokens: int = Field(default=4096, ge=256, le=32768)
    max_search_results: int = Field(default=100, ge=1, le=1000)
    max_dataset_records: int = Field(default=1000, ge=1, le=5000)
    max_dataset_upload_bytes: int = Field(default=10_000_000, ge=1024, le=100_000_000)
    database_path: Path = ROOT_DIR / "reports" / "assessment.sqlite3"
    reports_dir: Path = ROOT_DIR / "reports"
    dataset_path: Path = ROOT_DIR / "inputs" / "dataset.json"

    def require_jira(self) -> None:
        missing = [
            name
            for name, value in (
                ("JIRA_BASE_URL", self.jira_base_url),
                ("JIRA_EMAIL", self.jira_email),
                ("JIRA_API_TOKEN", self.jira_api_token.get_secret_value()),
                ("JIRA_PROJECT_KEY", self.jira_project_key),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing Jira configuration: {', '.join(missing)}")

    @property
    def llm_api_key(self) -> str:
        if self.llm_provider == "groq":
            return self.groq_api_key.get_secret_value()
        return self.google_api_key.get_secret_value()

    @property
    def llm_model(self) -> str:
        if self.llm_provider == "groq":
            return self.groq_model
        return self.google_model

    def require_llm(self) -> None:
        if not self.llm_api_key:
            variable = "GROQ_API_KEY" if self.llm_provider == "groq" else "GOOGLE_API_KEY"
            raise ValueError(
                f"Missing {self.llm_provider} configuration: {variable}"
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
