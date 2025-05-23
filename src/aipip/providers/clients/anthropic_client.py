# Concrete implementation for Anthropic
import os
from typing import Any, Dict, List, Optional
from pydantic import SecretStr

# Assuming the official anthropic library is installed
import anthropic

from ..interfaces.text_provider import TextProviderInterface, CompletionResponse

# Helper function to validate and format messages for Anthropic
def _prepare_anthropic_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not messages:
        raise ValueError("Anthropic requires at least one message.")

    formatted_messages = []
    expected_role = "user" # Anthropic always expects the first message to be from the user

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if not role or not content:
            raise ValueError(f"Message at index {i} is missing 'role' or 'content'.")

        # Map OpenAI roles (assistant -> assistant, user -> user)
        if role == "system":
             raise ValueError("System messages should be handled via the 'system' parameter, not in the main messages list for Anthropic.")
        elif role == "assistant":
            if expected_role != "assistant":
                raise ValueError(f"Message sequence error: Expected 'user' role but got 'assistant' at index {i}.")
            formatted_messages.append({"role": "assistant", "content": content})
            expected_role = "user"
        elif role == "user":
            if expected_role != "user":
                raise ValueError(f"Message sequence error: Expected 'assistant' role but got 'user' at index {i}.")
            formatted_messages.append({"role": "user", "content": content})
            expected_role = "assistant"
        else:
            raise ValueError(f"Unknown role '{role}' encountered at index {i}.")

    # Check if the last message was from the assistant, which is usually not allowed
    # if formatted_messages and formatted_messages[-1]['role'] == 'assistant':
    #     raise ValueError("The last message in the list cannot be from the assistant for Anthropic.")
    # Note: Depending on the exact use case (e.g., completing an assistant turn), this check might be too strict.
    # For now, we allow it but acknowledge potential API errors if the API enforces this.

    return formatted_messages


class AnthropicClient(TextProviderInterface):
    """Concrete implementation of TextProviderInterface for Anthropic models."""

    def __init__(self, api_key: Optional[SecretStr] = None):
        """Initializes the Anthropic client.

        Args:
            api_key: The Anthropic API key. If not provided, it will attempt
                     to use the ANTHROPIC_API_KEY environment variable.
        """
        resolved_key = api_key.get_secret_value() if api_key else os.environ.get('ANTHROPIC_API_KEY')
        if not resolved_key:
            raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set.")

        self.client = anthropic.Anthropic(api_key=resolved_key)
        self._default_model_name = "claude-3-sonnet-20240229" # Default model

    def generate_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None, # Anthropic calls it temperature
        max_tokens: Optional[int] = 1000, # Default if not None
        # Anthropic specific parameters often passed via kwargs:
        # system: Optional[str] = None,
        # top_p: Optional[float] = None,
        # top_k: Optional[int] = None,
        # stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> CompletionResponse:
        """Generates text completion using the Anthropic API."""

        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")
        if prompt and messages:
            # Treat prompt as a system message if messages are also present?
            # Or raise error? Let's prefer messages and ignore prompt for now.
            prompt = None

        model_name = model or self._default_model_name

        # Extract system prompt from kwargs or default to None
        system_prompt = kwargs.pop('system', None)

        # Prepare messages
        if messages:
            # Extract system prompt from the first message if not in kwargs
            if not system_prompt and messages[0].get("role") == "system":
                system_prompt = messages[0].get("content")
                processed_messages = messages[1:]
            else:
                processed_messages = messages

            # Validate and format remaining messages
            anthropic_messages = _prepare_anthropic_messages(processed_messages)
        elif prompt:
            # Convert simple prompt to Anthropic user message format
            anthropic_messages = [{"role": "user", "content": prompt}]
        else:
             raise ValueError("Neither prompt nor messages were provided after processing.")

        # Ensure defaults are applied if None is passed explicitly
        final_temperature = temperature if temperature is not None else 0.7 # Example default, match OpenAI?
        final_max_tokens = max_tokens if max_tokens is not None else 1000 # Use 1000 default

        # Check required Anthropic param AFTER applying default
        if final_max_tokens is None:
             raise ValueError("'max_tokens' is required for Anthropic API and no default could be applied.")

        try:
            # Prepare arguments, using the final values
            api_kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": anthropic_messages,
                "max_tokens": final_max_tokens,
                 # Add remaining kwargs first, allowing specific params below to override if needed
                **kwargs
            }
            if system_prompt:
                api_kwargs["system"] = system_prompt
            # Use final_temperature, only add if it has a value
            if final_temperature is not None:
                api_kwargs["temperature"] = final_temperature

            # Pop known optional args from kwargs if present, otherwise they might be passed twice
            top_p = kwargs.pop('top_p', None)
            top_k = kwargs.pop('top_k', None)
            stop_sequences = kwargs.pop('stop_sequences', None)

            if top_p is not None:
                 api_kwargs["top_p"] = top_p
            if top_k is not None:
                 api_kwargs["top_k"] = top_k
            if stop_sequences is not None:
                 api_kwargs["stop_sequences"] = stop_sequences

            response = self.client.messages.create(**api_kwargs)

            # Extract text content
            completion_text = ""
            if response.content and isinstance(response.content, list):
                # Assuming text content for now
                text_blocks = [block.text for block in response.content if hasattr(block, 'text')]
                completion_text = " ".join(text_blocks)

            # Prepare metadata
            metadata = {
                "id": response.id,
                "model": response.model,
                "role": response.role,
                "stop_reason": response.stop_reason,
                "stop_sequence": response.stop_sequence,
                "type": response.type,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            }

            return CompletionResponse(
                text=completion_text.strip(),
                provider_name="anthropic",
                metadata=metadata
            )

        except anthropic.APIError as e:
            print(f"Anthropic API returned an API Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

 