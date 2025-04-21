# Integration tests for Anthropic (require live API key and network access)
import pytest
import os
from dotenv import load_dotenv

# Load .env file first if it exists
load_dotenv()

# Import necessary components AFTER loading .env
from aipip.providers.clients.anthropic_client import AnthropicClient
from aipip.providers.interfaces.text_provider import CompletionResponse

# Check if the required API key is present in the environment
ANTHROPIC_API_KEY_PRESENT = bool(os.environ.get("ANTHROPIC_API_KEY"))
REASON_ANTHROPIC_KEY_MISSING = "Anthropic API key (ANTHROPIC_API_KEY) not found in environment variables or .env file."

# Mark all tests in this module to be skipped if the key is not present
pytestmark = pytest.mark.skipif(not ANTHROPIC_API_KEY_PRESENT, reason=REASON_ANTHROPIC_KEY_MISSING)

@pytest.mark.integration
def test_anthropic_live_message_completion():
    """Tests a live call to Anthropic messages endpoint."""
    # Arrange: Use the real client
    try:
        client = AnthropicClient()
    except ValueError as e:
        pytest.fail(f"Failed to initialize AnthropicClient even though key should be present: {e}")

    messages = [
        {"role": "user", "content": "Say 'Test successful!' in a short phrase."}
    ]
    # Use a relatively fast/cheap model like Haiku
    model = "claude-3-haiku-20240307"
    max_tokens = 10 # Anthropic requires max_tokens

    # Act: Make the live API call
    try:
        response = client.generate_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=0.1
        )
    except Exception as e:
        pytest.fail(f"Anthropic API call failed unexpectedly: {e}")

    # Assert: Check response structure and basic properties
    assert isinstance(response, CompletionResponse)
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    print(f"\nAnthropic Response Text: {response.text}") # Print for visibility

    assert isinstance(response.metadata, dict)
    assert response.metadata.get('model') is not None
    assert response.metadata.get('finish_reason') is not None # stop_reason
    assert response.metadata.get('id') is not None
    assert response.metadata.get('role') == 'assistant'
    assert isinstance(response.metadata.get('usage'), dict)
    assert response.metadata['usage'].get('output_tokens', 0) > 0 