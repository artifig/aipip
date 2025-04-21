# Concrete implementation for Google Generative AI
import os
from typing import Any, Dict, List, Optional
from pydantic import SecretStr

# Assuming the official google-generativeai library is installed
import google.generativeai as genai
from google.generativeai.types import generation_types, safety_types

from ..interfaces.text_provider import TextProviderInterface, CompletionResponse

# Helper to map roles
def _map_role_to_google(role: str) -> str:
    """Maps OpenAI-style roles to Google Gemini roles."""
    if role == "user":
        return "user"
    elif role == "assistant":
        return "model" # Google uses 'model' for assistant messages
    elif role == "system":
        # System role is handled separately by extracting the message
        # and passing it to the model's system_instruction parameter.
        # This function should not be called with role='system'.
        raise ValueError("System messages should be handled separately, not mapped directly.")
        # return "user" # Old behavior
    else:
        # Treat unknown roles as user for now
        return "user"

class GoogleClient(TextProviderInterface):
    """Concrete implementation of TextProviderInterface for Google Generative AI models."""

    def __init__(self, api_key: Optional[SecretStr] = None):
        """Initializes the Google Generative AI client.

        Args:
            api_key: The Google API key. If not provided, it will attempt
                     to use the GOOGLE_API_KEY environment variable.
        """
        resolved_key = api_key.get_secret_value() if api_key else os.environ.get('GOOGLE_API_KEY')
        if not resolved_key:
            raise ValueError("Google API key not provided and GOOGLE_API_KEY environment variable not set.")

        genai.configure(api_key=resolved_key)
        self._default_model_name = "gemini-1.5-flash" # Default model

    def generate_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        # Google specific parameters often passed via kwargs:
        # top_p: Optional[float] = None,
        # top_k: Optional[int] = None,
        # stop_sequences: Optional[List[str]] = None,
        # safety_settings: Optional[Dict[safety_types.HarmCategory, safety_types.HarmBlockThreshold]] = None,
        **kwargs: Any
    ) -> CompletionResponse:
        """Generates text completion using the Google Generative AI API."""

        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")
        if prompt and messages:
            prompt = None # Prefer messages over prompt

        model_name = model or self._default_model_name

        system_instruction_content: Optional[str] = None
        processed_messages: List[Dict[str, str]] = []

        # Prepare Google's 'contents' structure and extract system instruction
        if messages:
            if messages[0]["role"] == "system":
                system_instruction_content = messages[0]["content"]
                processed_messages = messages[1:] # Use the rest for contents
            else:
                processed_messages = messages
            # Convert message list, handling roles
            # Note: Google expects alternating user/model roles. This simple conversion
            # might fail if the input list has consecutive user/model messages.
            contents = []
            for msg in processed_messages:
                role = msg.get("role")
                content = msg.get("content")
                if role and content and role != "system": # Ignore any other system messages
                    contents.append({"role": _map_role_to_google(role), "parts": [{"text": content}]})
                # Consider adding a warning or error for invalid message structure
        elif prompt:
            # Convert simple prompt
            contents = [prompt]
        else:
             # This case is already checked at the beginning
            raise ValueError("Neither prompt nor messages were provided after processing.") # Should not happen

        # Initialize model with system instruction if present
        generative_model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction_content
        )

        # Prepare generation config
        generation_config_params = {}
        if temperature is not None:
            generation_config_params["temperature"] = temperature
        if max_tokens is not None:
            generation_config_params["max_output_tokens"] = max_tokens # Google uses max_output_tokens
        if "top_p" in kwargs:
            generation_config_params["top_p"] = kwargs.pop("top_p")
        if "top_k" in kwargs:
            generation_config_params["top_k"] = kwargs.pop("top_k")
        if "stop_sequences" in kwargs:
             generation_config_params["stop_sequences"] = kwargs.pop("stop_sequences")

        generation_config = generation_types.GenerationConfig(**generation_config_params) if generation_config_params else None

        # Extract safety settings if provided
        safety_settings = kwargs.pop("safety_settings", None)

        try:
            # Note: The 'contents' format supports richer structures (images etc.)
            # This implementation currently only handles basic text conversion.
            response = generative_model.generate_content(
                contents=contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
                # Pass remaining kwargs if any (though most common are handled)
                **kwargs
            )

            # Extract text and metadata
            completion_text = ""
            finish_reason = None
            usage_metadata = None # Google API doesn't typically return token usage in the same way
            safety_ratings = None

            # Handle potential lack of response content or errors indicated in the response
            if response.parts:
                completion_text = response.text # Convenience property
            elif response.prompt_feedback:
                 # Handle cases where the prompt was blocked
                 print(f"Warning: Prompt blocked due to {response.prompt_feedback.block_reason}")
                 # Or raise an error, depending on desired behavior
                 # raise ValueError(f"Prompt blocked: {response.prompt_feedback.block_reason}")

            if hasattr(response, 'candidates') and response.candidates:
                 # Finish reason might be in the candidate
                 finish_reason = response.candidates[0].finish_reason.name if hasattr(response.candidates[0].finish_reason, 'name') else str(response.candidates[0].finish_reason)
                 safety_ratings = [r.to_dict() for r in response.candidates[0].safety_ratings] if hasattr(response.candidates[0], 'safety_ratings') else None

            metadata = {
                'finish_reason': finish_reason,
                'safety_ratings': safety_ratings,
                # Extract prompt_feedback attributes directly if needed, avoid to_dict()
                'prompt_feedback_block_reason': response.prompt_feedback.block_reason.name if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback.block_reason, 'name') else None,
                # 'prompt_feedback_safety_ratings': [r.to_dict() for r in response.prompt_feedback.safety_ratings] if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'safety_ratings') else None,
                # 'usage': usage_metadata, # Usage metadata not typically available here
                'model': model_name,
            }

            return CompletionResponse(text=completion_text.strip(), metadata=metadata)

        except Exception as e:
            # Catch potential exceptions from the Google library
            # (e.g., google.api_core.exceptions.PermissionDenied, InvalidArgument)
            print(f"Google Generative AI API returned an error: {e}")
            raise # Or raise CustomProviderError("Google API error") from e 