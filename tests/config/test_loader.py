import pytest
from unittest.mock import patch

from aipip.config import loader
from aipip.config.models import Settings


@patch('aipip.config.loader.Settings') # Mock the Settings class
def test_load_settings_instantiates_settings(MockSettings):
    """Test that load_settings calls the Settings constructor."""
    # Arrange
    loader.load_settings.cache_clear() # Clear lru_cache before test

    # Act
    loaded_settings = loader.load_settings()

    # Assert
    MockSettings.assert_called_once()
    assert loaded_settings == MockSettings()


def test_load_settings_returns_settings_instance(monkeypatch):
    """Test that load_settings returns an instance of the actual Settings model."""
    # Arrange
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key') # Set a required env var for basic loading
    loader.load_settings.cache_clear() # Clear lru_cache

    # Act
    settings_instance = loader.load_settings()

    # Assert
    assert isinstance(settings_instance, Settings)
    # We can also check if the loaded value is correct, leveraging previous model tests
    assert settings_instance.provider_keys.openai_api_key.get_secret_value() == 'test-key'


def test_load_settings_uses_cache():
    """Test that load_settings uses lru_cache."""
    # Arrange
    loader.load_settings.cache_clear() # Clear cache first
    with patch.object(loader, 'Settings', wraps=loader.Settings) as mock_settings_init:
        # Act
        settings1 = loader.load_settings() # First call - should call Settings()
        settings2 = loader.load_settings() # Second call - should use cache

        # Assert
        mock_settings_init.assert_called_once() # Settings() should only be called once
        assert settings1 is settings2 # Should return the exact same object instance 