# Service layer for handling text generation tasks.

from typing import Any, Dict, List, Optional

from ..providers.registry import ProviderRegistry
from ..providers.interfaces.text_provider import CompletionResponse


class TextGenerationService:
    """Provides text generation functionality using configured providers."""

    def __init__(self, registry: ProviderRegistry):
        """Initializes the service with the provider registry.

        Args:
            registry: An initialized ProviderRegistry instance.
        """
        self._registry = registry

    def generate(
        self,
        provider_name: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> CompletionResponse:
        """Generates text using the specified provider.

        Args:
            provider_name: Name of the provider to use (e.g., 'openai', 'google').
            prompt: The text prompt (optional).
            messages: A list of message history (optional).
            model: The specific model name to use (optional, defaults to provider default).
            temperature: Sampling temperature (optional).
            max_tokens: Maximum tokens to generate (optional).
            **kwargs: Additional provider-specific keyword arguments.

        Returns:
            A CompletionResponse object from the provider.

        Raises:
            UnknownProviderError: If the provider name isn't recognized by the registry.
            ProviderNotConfiguredError: If the requested provider lacks necessary configuration (e.g., API key).
            # Other exceptions from the provider client itself (e.g., APIError).
        """
        # 1. Get the provider client instance from the registry
        # Registry handles exceptions like UnknownProviderError, ProviderNotConfiguredError
        provider_client = self._registry.get_provider(provider_name)

        # 2. Call the client's generate_completion method
        # Provider client handles API calls, parameter mapping, and specific errors
        response = provider_client.generate_completion(
            prompt=prompt,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # 3. Return the response
        return response

    # Potential future methods:
    # async def generate_async(...)
    # def generate_stream(...) 