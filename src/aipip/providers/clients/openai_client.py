# Concrete implementation for OpenAI
import os
from typing import Any, Dict, List, Optional
from pydantic import SecretStr

# Assuming the official openai library is installed
# We might need to add it to pyproject.toml dependencies
import openai

from ..interfaces.text_provider import TextProviderInterface, CompletionResponse

class OpenAIClient(TextProviderInterface):
    """Concrete implementation of TextProviderInterface for OpenAI models."""

    def __init__(self, api_key: Optional[SecretStr] = None):
        """Initializes the OpenAI client.

        Args:
            api_key: The OpenAI API key. If not provided, it will attempt
                     to use the OPENAI_API_KEY environment variable.
        """
        # Prefer explicitly passed key, fallback to environment variable
        resolved_key = api_key.get_secret_value() if api_key else os.environ.get('OPENAI_API_KEY')
        if not resolved_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set.")

        # Initialize the official OpenAI client
        # Note: The openai library uses the env var by default if api_key is None,
        # but we handle explicit passing/checking for clarity and consistency.
        self.client = openai.OpenAI(api_key=resolved_key)

    def generate_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.7, # Default temperature
        max_tokens: Optional[int] = 150, # Default max tokens
        **kwargs: Any
    ) -> CompletionResponse:
        """Generates text completion using the OpenAI API."""

        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Default model if not specified
        model = model or "gpt-3.5-turbo"

        try:
            # Prepare common parameters, preferring messages if available
            common_params = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs, # Include provider-specific kwargs
            }

            if messages:
                response = self.client.chat.completions.create(
                    messages=messages,
                    **common_params
                )
                completion_text = response.choices[0].message.content or ""
                # Extract potential metadata
                metadata = {
                    'finish_reason': response.choices[0].finish_reason,
                    'usage': response.usage.dict() if response.usage else None,
                    'model': response.model,
                    'id': response.id,
                    'created': response.created,
                }
            elif prompt:
                # Fallback to legacy completions endpoint if only prompt is given
                # Note: Newer models are often chat-only. This might need adjustment
                # depending on the default model and supported endpoints.
                # Consider raising an error or converting prompt to messages format.
                # For now, let's assume a suitable model is chosen that supports completion.
                response = self.client.completions.create(
                    prompt=prompt,
                    **common_params
                )
                completion_text = response.choices[0].text or ""
                metadata = {
                    'finish_reason': response.choices[0].finish_reason,
                    'usage': response.usage.dict() if response.usage else None,
                    'model': response.model,
                    'id': response.id,
                    'created': response.created,
                }
            else:
                # This case should be caught by the initial check, but added for completeness
                 raise ValueError("Could not determine completion method.")


            return CompletionResponse(text=completion_text.strip(), metadata=metadata)

        except openai.APIError as e:
            # Handle potential API errors (e.g., rate limits, authentication)
            # Re-raise or wrap in a custom exception
            print(f"OpenAI API returned an API Error: {e}")
            raise # Or raise CustomProviderError("OpenAI API error") from e
        except Exception as e:
            # Handle other unexpected errors
            print(f"An unexpected error occurred: {e}")
            raise # Or raise CustomProviderError("Unexpected error") from e 