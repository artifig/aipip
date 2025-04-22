# Pydantic models for configuration will go here 
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class ProviderKeys(BaseSettings):
    """API Keys for various AI providers, loaded from environment variables."""
    # Use SecretStr to prevent accidental logging of keys
    # Order: Anthropic, Google, OpenAI
    anthropic_api_key: Optional[SecretStr] = Field(None, alias='ANTHROPIC_API_KEY')
    google_api_key: Optional[SecretStr] = Field(None, alias='GOOGLE_API_KEY')
    openai_api_key: Optional[SecretStr] = Field(None, alias='OPENAI_API_KEY')

    # Use model_config with SettingsConfigDict for Pydantic V2/pydantic-settings
    model_config = SettingsConfigDict(
        env_file='.env', # Load from a .env file if present
        env_file_encoding='utf-8',
        extra='ignore' # Ignore extra environment variables
    )

class AppConfig(BaseSettings):
    """Main application configuration."""
    provider_keys: ProviderKeys = Field(default_factory=ProviderKeys)

    # Add other general application settings here later if needed
    # e.g., default_model: str = "openai:gpt-4o"

    # Use model_config with SettingsConfigDict for Pydantic V2/pydantic-settings
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    ) 