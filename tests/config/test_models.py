import pytest
import os
from unittest.mock import patch
from pydantic import SecretStr

from aipip.config.models import Settings, ProviderKeys

@patch.dict(os.environ, {
    "OPENAI_API_KEY": "sk-openai-12345",
    "GOOGLE_API_KEY": "google-secret-key",
    "ANTHROPIC_API_KEY": "anthropic-test-key"
    }, clear=True)
def test_settings_load_from_env():
    """Test that Settings model loads provider keys from environment variables."""
    # Arrange & Act: Load settings within the patched environment
    settings = Settings()

    # Assert: Check if keys are loaded correctly and are SecretStr
    assert isinstance(settings.provider_keys.openai_api_key, SecretStr)
    assert settings.provider_keys.openai_api_key.get_secret_value() == "sk-openai-12345"

    assert isinstance(settings.provider_keys.google_api_key, SecretStr)
    assert settings.provider_keys.google_api_key.get_secret_value() == "google-secret-key"

    assert isinstance(settings.provider_keys.anthropic_api_key, SecretStr)
    assert settings.provider_keys.anthropic_api_key.get_secret_value() == "anthropic-test-key"

@patch.dict(os.environ, {}, clear=True) # Clear all env vars for this test
def test_settings_optional_keys():
    """Test that missing optional keys result in None when env is empty."""
    # Arrange & Act: Load settings, preventing .env file loading
    # Pass dummy _env_file path to prevent loading default .env
    provider_keys = ProviderKeys(_env_file='/dev/null') # Or any non-existent path
    settings = Settings(provider_keys=provider_keys, _env_file='/dev/null')

    # Assert: Check that missing keys are None
    assert settings.provider_keys.openai_api_key is None
    assert settings.provider_keys.google_api_key is None
    assert settings.provider_keys.anthropic_api_key is None 