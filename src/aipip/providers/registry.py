# Central registry for managing and providing provider client instances.

from typing import Dict, Optional, Type

from ..config.models import Settings
from .interfaces.text_provider import TextProviderInterface
from .clients.openai_client import OpenAIClient
from .clients.google_client import GoogleClient
from .clients.anthropic_client import AnthropicClient
# Import other clients like AnthropicClient later

class ProviderNotConfiguredError(Exception):
    """Exception raised when a requested provider is not configured (e.g., missing API key)."""
    pass

class UnknownProviderError(Exception):
    """Exception raised when an unknown provider name is requested."""
    pass

class ProviderRegistry:
    """Manages the instantiation and retrieval of provider clients."""

    # Map provider names (strings) to their corresponding client classes
    _PROVIDER_MAP: Dict[str, Type[TextProviderInterface]] = {
        "openai": OpenAIClient,
        "google": GoogleClient,
        # "anthropic": AnthropicClient, # Add later
        "anthropic": AnthropicClient,
    }

    def __init__(self, settings: Settings):
        """Initializes the registry with application settings.

        Args:
            settings: The loaded application settings containing API keys.
        """
        self._settings = settings
        self._instances: Dict[str, TextProviderInterface] = {}

    def get_provider(self, provider_name: str) -> TextProviderInterface:
        """Gets an initialized provider client instance.

        Instantiates the client on the first request if configured.
        Subsequent requests for the same provider return the cached instance.

        Args:
            provider_name: The name of the provider (e.g., 'openai', 'google').

        Returns:
            An initialized instance of the requested provider client.

        Raises:
            UnknownProviderError: If the provider_name is not recognized.
            ProviderNotConfiguredError: If the provider is recognized but not configured
                                       (e.g., missing API key).
        """
        provider_name = provider_name.lower()

        # Return cached instance if available
        if provider_name in self._instances:
            return self._instances[provider_name]

        # Check if provider is known
        client_class = self._PROVIDER_MAP.get(provider_name)
        if not client_class:
            raise UnknownProviderError(f"Unknown provider: '{provider_name}'. Known providers: {list(self._PROVIDER_MAP.keys())}")

        # Check configuration and instantiate
        try:
            if provider_name == "openai":
                api_key = self._settings.provider_keys.openai_api_key
                if not api_key:
                    raise ProviderNotConfiguredError("OpenAI API key not configured.")
                instance = OpenAIClient(api_key=api_key)
            elif provider_name == "google":
                api_key = self._settings.provider_keys.google_api_key
                if not api_key:
                    raise ProviderNotConfiguredError("Google API key not configured.")
                instance = GoogleClient(api_key=api_key)
            elif provider_name == "anthropic":
                api_key = self._settings.provider_keys.anthropic_api_key
                if not api_key:
                    raise ProviderNotConfiguredError("Anthropic API key not configured.")
                instance = AnthropicClient(api_key=api_key)
            else:
                # This case should be caught by the _PROVIDER_MAP check, but acts as a safeguard
                raise UnknownProviderError(f"Instantiation logic missing for known provider: '{provider_name}'")

        except ValueError as e:
            # Catch potential ValueError from client __init__ if key is invalid format (though unlikely with SecretStr)
            # or if the client itself raises ValueError for missing key (redundant check here)
            raise ProviderNotConfiguredError(f"Failed to initialize {provider_name}: {e}") from e

        # Cache and return the new instance
        self._instances[provider_name] = instance
        return instance 