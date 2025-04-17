# Pydantic models for configuration will go here 
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings
from typing import Optional

class ProviderKeys(BaseSettings):
    """API Keys for various AI providers, loaded from environment variables."""
    # Use SecretStr to prevent accidental logging of keys
    openai_api_key: Optional[SecretStr] = Field(None, alias='OPENAI_API_KEY', env='OPENAI_API_KEY')
    google_api_key: Optional[SecretStr] = Field(None, alias='GOOGLE_API_KEY', env='GOOGLE_API_KEY')
    anthropic_api_key: Optional[SecretStr] = Field(None, alias='ANTHROPIC_API_KEY', env='ANTHROPIC_API_KEY')

    class Config:
        env_file = '.env' # Load from a .env file if present
        env_file_encoding = 'utf-8'
        extra = 'ignore' # Ignore extra environment variables

class Settings(BaseSettings):
    """Main application settings."""
    provider_keys: ProviderKeys = Field(default_factory=ProviderKeys)

    # Add other general application settings here later if needed
    # e.g., default_model: str = "openai:gpt-4o"

    class Config:
        # If Settings itself needs to load from env or .env
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore' 