import pytest
import os
from unittest.mock import patch
from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from aipip.config.models import AppConfig, ProviderKeys

@patch.dict(os.environ, {
    "OPENAI_API_KEY": "sk-openai-12345",
    "GOOGLE_API_KEY": "google-secret-key",
    "ANTHROPIC_API_KEY": "anthropic-test-key"
    }, clear=True)
def test_appconfig_load_from_env():
    """Test that AppConfig model loads provider keys from environment variables."""
    # Arrange & Act: Load config (instantiate AppConfig) within the patched environment
    config = AppConfig()

    # Assert: Check if keys are loaded correctly and are SecretStr
    assert isinstance(config.provider_keys.openai_api_key, SecretStr)
    assert config.provider_keys.openai_api_key.get_secret_value() == "sk-openai-12345"

    assert isinstance(config.provider_keys.google_api_key, SecretStr)
    assert config.provider_keys.google_api_key.get_secret_value() == "google-secret-key"

    assert isinstance(config.provider_keys.anthropic_api_key, SecretStr)
    assert config.provider_keys.anthropic_api_key.get_secret_value() == "anthropic-test-key"

@patch.dict(os.environ, {}, clear=True) # Clear all env vars for this test
def test_appconfig_optional_keys(monkeypatch):
    """Test that missing optional keys result in None when env is empty.

    Also ensures .env file loading is temporarily disabled for this test.
    """
    # Arrange & Act: Load config, preventing .env file loading
    # Temporarily disable .env loading for the ProviderKeys sub-model
    original_pk_model_config = ProviderKeys.model_config
    monkeypatch.setattr(ProviderKeys, 'model_config', SettingsConfigDict(env_file=None))

    config = AppConfig()

    # Assert: Check that missing keys are None
    assert config.provider_keys.openai_api_key is None
    assert config.provider_keys.google_api_key is None
    assert config.provider_keys.anthropic_api_key is None 