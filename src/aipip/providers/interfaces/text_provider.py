from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Placeholder for a potential structured response object later
# Using dataclass for simplicity and type hints
from dataclasses import dataclass, field

@dataclass
class CompletionResponse:
    text: str
    provider_name: str # Added provider name
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

class TextProviderInterface(ABC):
    """Abstract Base Class defining the interface for text generation providers."""

    @abstractmethod
    def generate_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        # Add other common parameters as needed (e.g., top_p, stop_sequences)
        **kwargs: Any
    ) -> CompletionResponse:
        """Generates text completion based on a prompt or message history.

        Args:
            prompt: A single string prompt (for simpler models/use cases).
            messages: A list of message dictionaries (e.g., OpenAI format: [{'role': 'user', 'content': ...}]).
                      Providers should ideally support this format.
            model: The specific model name to use for the provider.
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Provider-specific parameters.

        Returns:
            A CompletionResponse object containing the generated text and potentially metadata.

        Raises:
            ValueError: If neither prompt nor messages are provided.
            NotImplementedError: If called directly on the abstract class.
            # Specific provider exceptions (e.g., APIError) should be handled/raised by implementations.
        """
        raise NotImplementedError

    # Future methods could include:
    # - generate_completion_stream(...)
    # - count_tokens(...)
    # - list_models(...) 