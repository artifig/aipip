#!/usr/bin/env python3

import pytest
import os
from pydantic import ValidationError
from unittest.mock import patch, MagicMock

# Import the loader and models
from aipip.config import models
from aipip.config import loader # Keep import as loader
from aipip.config.models import AppConfig, ProviderKeys, SettingsConfigDict


def test_load_config_success(monkeypatch):
    """Test loading config successfully with environment variables."""
    # Mock environment variables using aliases defined in ProviderKeys model
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")

    # Clear lru_cache for the test
    loader.load_config.cache_clear()

    # Call the loader function (now named load_config)
    loaded_config = loader.load_config()

    assert isinstance(loaded_config, AppConfig)
    assert isinstance(loaded_config.provider_keys, ProviderKeys)
    # Access keys via the nested provider_keys object and use get_secret_value()
    assert loaded_config.provider_keys.openai_api_key.get_secret_value() == "sk-test-openai"
    assert loaded_config.provider_keys.anthropic_api_key.get_secret_value() == "sk-test-anthropic"
    assert loaded_config.provider_keys.google_api_key.get_secret_value() == "test-google-key"

def test_load_config_missing_key(monkeypatch):
    """Test that loading returns None for an optional key if env var is missing.

    Also ensures .env file loading is temporarily disabled for this test.
    """
    # Mock environment variables for other keys
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    # Ensure OPENAI_API_KEY is *not* set in the environment for this test
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Temporarily disable .env file loading for ProviderKeys model
    original_model_config = models.ProviderKeys.model_config
    monkeypatch.setattr(models.ProviderKeys, 'model_config', SettingsConfigDict(env_file=None))

    # Clear lru_cache for load_config
    loader.load_config.cache_clear()

    # Load config under the temporary config
    config_instance = loader.load_config()

    # Assert that the missing key is None (since it's Optional and no source provided it)
    assert config_instance.provider_keys.openai_api_key is None
    # Assert that existing keys (set via env vars) are loaded correctly
    assert config_instance.provider_keys.anthropic_api_key.get_secret_value() == "sk-test-anthropic"
    assert config_instance.provider_keys.google_api_key.get_secret_value() == "test-google-key"


# Test lru_cache functionality
@patch('aipip.config.loader.AppConfig')
def test_load_config_uses_lru_cache(mock_config_class):
    """Test that load_config uses lru_cache and only calls AppConfig() once."""
    # Create a mock instance to be returned by the mocked class constructor
    mock_instance = MagicMock(spec=AppConfig)
    mock_config_class.return_value = mock_instance

    # Clear the cache before the test
    loader.load_config.cache_clear()

    # Call load_config multiple times
    config1 = loader.load_config() # First call - should call AppConfig()
    config2 = loader.load_config() # Second call - should use cache

    # Assert that AppConfig() was called exactly once
    mock_config_class.assert_called_once()

    # Assert that both returned objects are the same instance
    assert config1 is config2
    assert config1 is mock_instance 