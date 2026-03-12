"""Application configuration via Pydantic BaseSettings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # OpenAI
    openai_api_key: str = ""

    # LLM
    llm_model: str = "gpt-4o-mini"

    # Game
    max_turns: int = 20
    token_budget_per_agent: int = 8000

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"


# Singleton instance — import this everywhere
settings = Settings()
