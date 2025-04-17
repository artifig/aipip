import pytest
from pydantic import SecretStr

from aipip.config.models import Settings


def test_settings_load_from_env(monkeypatch):
    """Test that Settings model loads provider keys from environment variables."""
    # Arrange: Set mock environment variables
    mock_openai_key = "sk-openai-12345"
    mock_google_key = "google-secret-key"
    mock_anthropic_key = "anthropic-test-key"

    monkeypatch.setenv('OPENAI_API_KEY', mock_openai_key)
    monkeypatch.setenv('GOOGLE_API_KEY', mock_google_key)
    monkeypatch.setenv('ANTHROPIC_API_KEY', mock_anthropic_key)

    # Act: Load the settings
    settings = Settings()

    # Assert: Check if keys are loaded correctly and are SecretStr
    assert isinstance(settings.provider_keys.openai_api_key, SecretStr)
    assert settings.provider_keys.openai_api_key.get_secret_value() == mock_openai_key

    assert isinstance(settings.provider_keys.google_api_key, SecretStr)
    assert settings.provider_keys.google_api_key.get_secret_value() == mock_google_key

    assert isinstance(settings.provider_keys.anthropic_api_key, SecretStr)
    assert settings.provider_keys.anthropic_api_key.get_secret_value() == mock_anthropic_key


def test_settings_optional_keys(monkeypatch):
    """Test that missing optional keys result in None."""
    # Arrange: Ensure no relevant env vars are set (monkeypatch handles cleanup)
    # monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    # monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    # monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    # Note: Explicitly deleting isn't strictly necessary if they aren't set,
    # but shown here for clarity if needed in other tests.

    # Act: Load the settings
    settings = Settings()

    # Assert: Check that missing keys are None
    assert settings.provider_keys.openai_api_key is None
    assert settings.provider_keys.google_api_key is None
    assert settings.provider_keys.anthropic_api_key is None 