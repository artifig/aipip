# Integration tests (require live API keys and network access)
import pytest
import os
from pydantic import SecretStr
from dotenv import load_dotenv

# Load .env file first if it exists
load_dotenv()

# Import necessary components AFTER loading .env
from aipip.providers.clients.openai_client import OpenAIClient
from aipip.providers.interfaces.text_provider import CompletionResponse

# Check if the required API key is present in the environment (loaded from .env or system env)
OPENAI_API_KEY_PRESENT = bool(os.environ.get("OPENAI_API_KEY"))
REASON_OPENAI_KEY_MISSING = "OpenAI API key (OPENAI_API_KEY) not found in environment variables or .env file."

# Mark all tests in this module to be skipped if the key is not present
pytestmark = pytest.mark.skipif(not OPENAI_API_KEY_PRESENT, reason=REASON_OPENAI_KEY_MISSING)

# Mark tests with the 'integration' marker
@pytest.mark.integration
def test_openai_live_chat_completion():
    """Tests a live call to OpenAI chat completion endpoint."""
    # Arrange: Use the real client. It reads the key from the environment (already loaded).
    try:
        client = OpenAIClient()
    except ValueError as e:
        pytest.fail(f"Failed to initialize OpenAIClient even though key should be present: {e}")

    messages = [
        {"role": "user", "content": "Say 'Test successful!'"}
    ]
    model = "gpt-3.5-turbo"
    max_tokens = 10

    # Act: Make the live API call
    try:
        response = client.generate_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=0.1
        )
    except Exception as e:
        pytest.fail(f"OpenAI API call failed unexpectedly: {e}")

    # Assert: Check response structure and basic properties
    assert isinstance(response, CompletionResponse)
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    print(f"\nOpenAI Response Text: {response.text}")

    assert isinstance(response.metadata, dict)
    assert response.metadata.get('model') is not None
    assert response.metadata.get('finish_reason') is not None
    assert response.metadata.get('id') is not None
    assert isinstance(response.metadata.get('usage'), dict)
    assert response.metadata['usage'].get('total_tokens', 0) > 0 