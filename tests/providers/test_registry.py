import pytest
import os
from unittest.mock import MagicMock, patch
from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from aipip.config.models import AppConfig, ProviderKeys
from aipip.providers.registry import ProviderRegistry, UnknownProviderError, ProviderNotConfiguredError
from aipip.providers.clients.openai_client import OpenAIClient
from aipip.providers.clients.google_client import GoogleClient
from aipip.providers.clients.anthropic_client import AnthropicClient

# Fixtures no longer needed, tests control env via patch.dict

# --- Test Cases ---

def test_registry_init():
    """Test that the registry initializes correctly."""
    # Use patch.dict to ensure clean environment for this test
    with patch.dict(os.environ, {}, clear=True):
        keys = ProviderKeys()
        config = AppConfig(provider_keys=keys)
        registry = ProviderRegistry(config=config)
        assert registry.config == config
        assert registry._instances == {}

@patch('aipip.providers.registry.OpenAIClient')
@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-key", "GOOGLE_API_KEY": "google-key", "ANTHROPIC_API_KEY": "anthropic-key"}, clear=True)
def test_get_provider_openai_success(MockOpenAIClient):
    """Test getting OpenAI provider successfully when configured via env."""
    # Arrange
    mock_instance = MagicMock(spec=OpenAIClient)
    MockOpenAIClient.return_value = mock_instance
    keys = ProviderKeys()
    config = AppConfig(provider_keys=keys)
    registry = ProviderRegistry(config=config)

    # Act
    provider = registry.get_provider("openai")

    # Assert
    assert config.provider_keys.openai_api_key is not None
    MockOpenAIClient.assert_called_once_with(api_key=config.provider_keys.openai_api_key)
    assert provider == mock_instance
    assert registry._instances["openai"] == mock_instance

@patch('aipip.providers.registry.GoogleClient')
@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-key", "GOOGLE_API_KEY": "google-key", "ANTHROPIC_API_KEY": "anthropic-key"}, clear=True)
def test_get_provider_google_success(MockGoogleClient):
    """Test getting Google provider successfully when configured via env."""
    # Arrange
    mock_instance = MagicMock(spec=GoogleClient)
    MockGoogleClient.return_value = mock_instance
    keys = ProviderKeys()
    config = AppConfig(provider_keys=keys)
    registry = ProviderRegistry(config=config)

    # Act
    provider = registry.get_provider("google")

    # Assert
    assert config.provider_keys.google_api_key is not None
    MockGoogleClient.assert_called_once_with(api_key=config.provider_keys.google_api_key)
    assert provider == mock_instance
    assert registry._instances["google"] == mock_instance

@patch('aipip.providers.registry.AnthropicClient')
@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-key", "GOOGLE_API_KEY": "google-key", "ANTHROPIC_API_KEY": "anthropic-key"}, clear=True)
def test_get_provider_anthropic_success(MockAnthropicClient):
    """Test getting Anthropic provider successfully when configured via env."""
    # Arrange
    mock_instance = MagicMock(spec=AnthropicClient)
    MockAnthropicClient.return_value = mock_instance
    keys = ProviderKeys()
    config = AppConfig(provider_keys=keys)
    registry = ProviderRegistry(config=config)

    # Act
    provider = registry.get_provider("anthropic")

    # Assert
    assert config.provider_keys.anthropic_api_key is not None
    MockAnthropicClient.assert_called_once_with(api_key=config.provider_keys.anthropic_api_key)
    assert provider == mock_instance
    assert registry._instances["anthropic"] == mock_instance

@patch.dict(os.environ, {}, clear=True)
def test_get_provider_openai_not_configured(monkeypatch):
    """Test ProviderNotConfiguredError when OpenAI key is missing."""
    # Arrange
    # Temporarily disable .env loading
    original_pk_model_config = ProviderKeys.model_config
    monkeypatch.setattr(ProviderKeys, 'model_config', SettingsConfigDict(env_file=None))

    config = AppConfig()
    registry = ProviderRegistry(config=config)

    # Act & Assert
    with pytest.raises(ProviderNotConfiguredError, match="OpenAI API key not configured"):
        registry.get_provider("openai")
    assert "openai" not in registry._instances

@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-key"}, clear=True)
def test_get_provider_google_not_configured(monkeypatch):
    """Test ProviderNotConfiguredError when Google key is missing."""
    # Arrange
    # Temporarily disable .env loading
    original_pk_model_config = ProviderKeys.model_config
    monkeypatch.setattr(ProviderKeys, 'model_config', SettingsConfigDict(env_file=None))

    config = AppConfig()
    registry = ProviderRegistry(config=config)

    # Act & Assert
    with pytest.raises(ProviderNotConfiguredError, match="Google API key not configured"):
        registry.get_provider("google")
    assert "google" not in registry._instances

@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-key"}, clear=True)
def test_get_provider_anthropic_not_configured(monkeypatch):
    """Test ProviderNotConfiguredError when Anthropic key is missing."""
    # Arrange
    # Temporarily disable .env loading
    original_pk_model_config = ProviderKeys.model_config
    monkeypatch.setattr(ProviderKeys, 'model_config', SettingsConfigDict(env_file=None))

    config = AppConfig()
    registry = ProviderRegistry(config=config)

    # Act & Assert
    with pytest.raises(ProviderNotConfiguredError, match="Anthropic API key not configured"):
        registry.get_provider("anthropic")
    assert "anthropic" not in registry._instances

@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-key"}, clear=True)
def test_get_provider_unknown_name():
    """Test UnknownProviderError for an unrecognized provider name."""
    # Arrange
    config = AppConfig()
    registry = ProviderRegistry(config=config)

    # Act & Assert
    with pytest.raises(UnknownProviderError, match="Unknown provider: 'foobar'"):
        registry.get_provider("foobar")

@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-key"}, clear=True)
def test_get_provider_case_insensitive():
    """Test that provider names are handled case-insensitively."""
    # Arrange
    config = AppConfig()
    registry = ProviderRegistry(config=config)
    with patch('aipip.providers.registry.OpenAIClient') as MockOpenAIClient:
        mock_instance = MagicMock(spec=OpenAIClient)
        MockOpenAIClient.return_value = mock_instance

        # Act
        provider = registry.get_provider("OpEnAi")

        # Assert
        assert config.provider_keys.openai_api_key is not None
        MockOpenAIClient.assert_called_once_with(api_key=config.provider_keys.openai_api_key)
        assert provider == mock_instance
        assert registry._instances["openai"] == mock_instance 