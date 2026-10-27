from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    model: str = Field("claude-sonnet-5-5", validation_alias="CLAUDE_MODEL")
    small_model: str = "claude-haiku-5-5"
    max_tokens: int = 1024
    max_history_turns: int = 30

settings = Settings()
