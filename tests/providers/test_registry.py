import pytest
from unittest.mock import MagicMock, patch
from pydantic import SecretStr

from aipip.config.models import Settings, ProviderKeys
from aipip.providers.registry import ProviderRegistry, UnknownProviderError, ProviderNotConfiguredError
from aipip.providers.clients.openai_client import OpenAIClient
from aipip.providers.clients.google_client import GoogleClient
from aipip.providers.clients.anthropic_client import AnthropicClient

# --- Fixtures ---

@pytest.fixture
def mock_provider_keys_all(monkeypatch) -> ProviderKeys:
    """ProviderKeys fixture with env vars set for all known keys."""
    # Set env vars *before* instantiating ProviderKeys
    monkeypatch.setenv("OPENAI_API_KEY", "sk-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    # Add anthropic env var later if needed
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    # Now instantiate - BaseSettings will load from the patched env
    keys = ProviderKeys()
    # We can assert here to be sure the loading worked as expected in the fixture
    assert keys.openai_api_key is not None
    assert keys.openai_api_key.get_secret_value() == "sk-key"
    assert keys.google_api_key is not None
    assert keys.google_api_key.get_secret_value() == "google-key"
    assert keys.anthropic_api_key is not None
    assert keys.anthropic_api_key.get_secret_value() == "anthropic-key"
    return keys

@pytest.fixture
def mock_provider_keys_openai_only(monkeypatch) -> ProviderKeys:
    """ProviderKeys fixture with env var set only for OpenAI."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-key")
    # Ensure others are not set
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    keys = ProviderKeys()
    assert keys.openai_api_key is not None
    assert keys.google_api_key is None
    assert keys.anthropic_api_key is None
    return keys

@pytest.fixture
def mock_provider_keys_none(monkeypatch) -> ProviderKeys:
    """ProviderKeys fixture ensuring no relevant env vars are set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return ProviderKeys()

# --- Test Cases ---

def test_registry_init(mock_provider_keys_none):
    """Test that the registry initializes correctly."""
    settings = Settings(provider_keys=mock_provider_keys_none)
    registry = ProviderRegistry(settings=settings)
    assert registry._settings == settings
    assert registry._instances == {}

@patch('aipip.providers.registry.OpenAIClient') # Patch the actual client class
def test_get_provider_openai_success(MockOpenAIClient, mock_provider_keys_all):
    """Test getting OpenAI provider successfully when configured."""
    # Arrange
    mock_instance = MagicMock(spec=OpenAIClient)
    MockOpenAIClient.return_value = mock_instance
    settings = Settings(provider_keys=mock_provider_keys_all) # Create Settings manually
    registry = ProviderRegistry(settings=settings)

    # Act
    provider = registry.get_provider("openai")

    # Assert
    # Access the key directly from the input fixture for assertion
    MockOpenAIClient.assert_called_once_with(api_key=mock_provider_keys_all.openai_api_key)
    assert provider == mock_instance
    assert registry._instances["openai"] == mock_instance # Check caching

@patch('aipip.providers.registry.GoogleClient') # Patch the actual client class
def test_get_provider_google_success(MockGoogleClient, mock_provider_keys_all):
    """Test getting Google provider successfully when configured."""
    # Arrange
    mock_instance = MagicMock(spec=GoogleClient)
    MockGoogleClient.return_value = mock_instance
    settings = Settings(provider_keys=mock_provider_keys_all) # Create Settings manually
    registry = ProviderRegistry(settings=settings)

    # Act
    provider = registry.get_provider("google")

    # Assert
    MockGoogleClient.assert_called_once_with(api_key=mock_provider_keys_all.google_api_key)
    assert provider == mock_instance
    assert registry._instances["google"] == mock_instance # Check caching

@patch('aipip.providers.registry.AnthropicClient') # Patch the actual client class
def test_get_provider_anthropic_success(MockAnthropicClient, mock_provider_keys_all):
    """Test getting Anthropic provider successfully when configured."""
    # Arrange
    mock_instance = MagicMock(spec=AnthropicClient)
    MockAnthropicClient.return_value = mock_instance
    settings = Settings(provider_keys=mock_provider_keys_all) # Create Settings manually
    registry = ProviderRegistry(settings=settings)

    # Act
    provider = registry.get_provider("anthropic")

    # Assert
    MockAnthropicClient.assert_called_once_with(api_key=mock_provider_keys_all.anthropic_api_key)
    assert provider == mock_instance
    assert registry._instances["anthropic"] == mock_instance # Check caching

def test_get_provider_uses_cache(mock_provider_keys_all):
    """Test that subsequent calls return the same cached instance."""
    # Arrange
    settings = Settings(provider_keys=mock_provider_keys_all) # Create Settings manually
    registry = ProviderRegistry(settings=settings)
    # Patch the client constructors *after* registry init
    with patch('aipip.providers.registry.OpenAIClient') as MockOpenAIClient:
        mock_instance = MagicMock(spec=OpenAIClient)
        MockOpenAIClient.return_value = mock_instance

        # Act
        provider1 = registry.get_provider("openai")
        provider2 = registry.get_provider("openai") # Second call

        # Assert
        MockOpenAIClient.assert_called_once() # Constructor only called once
        assert provider1 is provider2 # Should be the exact same object
        assert provider1 == mock_instance

def test_get_provider_openai_not_configured(mock_provider_keys_none):
    """Test ProviderNotConfiguredError when OpenAI key is missing."""
    # Arrange
    settings = Settings(provider_keys=mock_provider_keys_none)
    registry = ProviderRegistry(settings=settings)

    # Act & Assert
    with pytest.raises(ProviderNotConfiguredError, match="OpenAI API key not configured"):
        registry.get_provider("openai")
    assert "openai" not in registry._instances # Should not be cached

def test_get_provider_google_not_configured(mock_provider_keys_openai_only):
    """Test ProviderNotConfiguredError when Google key is missing."""
    # Arrange
    settings = Settings(provider_keys=mock_provider_keys_openai_only)
    registry = ProviderRegistry(settings=settings)

    # Act & Assert
    with pytest.raises(ProviderNotConfiguredError, match="Google API key not configured"):
        registry.get_provider("google")
    assert "google" not in registry._instances # Should not be cached

def test_get_provider_anthropic_not_configured(mock_provider_keys_openai_only):
    """Test ProviderNotConfiguredError when Anthropic key is missing."""
    # Arrange
    settings = Settings(provider_keys=mock_provider_keys_openai_only)
    registry = ProviderRegistry(settings=settings)

    # Act & Assert
    with pytest.raises(ProviderNotConfiguredError, match="Anthropic API key not configured"):
        registry.get_provider("anthropic")
    assert "anthropic" not in registry._instances # Should not be cached

def test_get_provider_unknown_name(mock_provider_keys_all):
    """Test UnknownProviderError for an unrecognized provider name."""
    # Arrange
    settings = Settings(provider_keys=mock_provider_keys_all)
    registry = ProviderRegistry(settings=settings)

    # Act & Assert
    with pytest.raises(UnknownProviderError, match="Unknown provider: 'foobar'"):
        registry.get_provider("foobar")

def test_get_provider_case_insensitive(mock_provider_keys_all):
    """Test that provider names are handled case-insensitively."""
    # Arrange
    settings = Settings(provider_keys=mock_provider_keys_all)
    registry = ProviderRegistry(settings=settings)
    with patch('aipip.providers.registry.OpenAIClient') as MockOpenAIClient:
        mock_instance = MagicMock(spec=OpenAIClient)
        MockOpenAIClient.return_value = mock_instance

        # Act
        provider = registry.get_provider("OpEnAi") # Mixed case

        # Assert
        MockOpenAIClient.assert_called_once()
        assert provider == mock_instance
        assert registry._instances["openai"] == mock_instance # Cached under lowercase name 