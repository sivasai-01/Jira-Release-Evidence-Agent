from pathlib import Path

from app.config import Settings


def test_gemini_is_the_default_llm_provider() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "gemini"
    assert settings.llm_model == "gemini-3.8-flash"


def test_existing_local_env_variable_names_are_supported(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "JIRA_URL=https://jira.example.test\n"
        "JIRA_EMAIL=reviewer@example.test\n"
        "JIRA_API_TOKEN=test-token\n"
        "GEMINI_API_KEY=test-gemini-key\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.jira_base_url == "https://jira.example.test"
    assert settings.google_api_key.get_secret_value() == "test-gemini-key"


def test_groq_configuration_is_loaded_from_environment_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=groq\n"
        "GROQ_API_KEY=test-groq-key\n"
        "GROQ_BASE_URL=https://api.groq.com/openai/v1\n"
        "GROQ_MODEL=qwen/qwen3.8-27b\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.llm_provider == "groq"
    assert settings.llm_api_key == "test-groq-key"
    assert settings.llm_model == "qwen/qwen3.8-27b"
    assert settings.groq_base_url == "https://api.groq.com/openai/v1"
