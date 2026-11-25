from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str
    ANTHROPIC_API_KEY: str

    QDRANT_URL: str = "http://qdrant:6333"
    COLLECTION_NAME: str = "Amazon-items-collection-01"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


config = Config()
