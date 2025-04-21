# Integration tests for Google (require live API key and network access)
import pytest
import os
from dotenv import load_dotenv

# Load .env file first if it exists
load_dotenv()

# Import necessary components AFTER loading .env
from aipip.providers.clients.google_client import GoogleClient
from aipip.providers.interfaces.text_provider import CompletionResponse

# Check if the required API key is present in the environment
GOOGLE_API_KEY_PRESENT = bool(os.environ.get("GOOGLE_API_KEY"))
REASON_GOOGLE_KEY_MISSING = "Google API key (GOOGLE_API_KEY) not found in environment variables or .env file."

# Mark all tests in this module to be skipped if the key is not present
pytestmark = pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason=REASON_GOOGLE_KEY_MISSING)

@pytest.mark.integration
def test_google_live_text_completion():
    """Tests a live call to Google Generative AI text completion endpoint."""
    # Arrange: Use the real client
    try:
        client = GoogleClient()
    except ValueError as e:
        pytest.fail(f"Failed to initialize GoogleClient even though key should be present: {e}")

    # Use prompt for simplicity with Google's API structure
    prompt = "Say 'Test successful!' in a short sentence."
    model = "gemini-1.5-flash" # Use a standard, available model
    max_tokens = 15

    # Act: Make the live API call
    try:
        response = client.generate_completion(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=0.1
        )
    except Exception as e:
        pytest.fail(f"Google API call failed unexpectedly: {e}")

    # Assert: Check response structure and basic properties
    assert isinstance(response, CompletionResponse)
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    print(f"\nGoogle Response Text: {response.text}") # Print for visibility

    assert isinstance(response.metadata, dict)
    assert response.metadata.get('model') == model # Google client extracts requested model
    assert response.metadata.get('finish_reason') is not None # Should have a finish reason
    # Note: Google API might not return usage stats or detailed IDs in the same way
    # as OpenAI in this basic response structure, so we focus on text and finish reason.
    assert response.metadata.get('safety_ratings') is not None
    assert response.metadata.get('prompt_feedback_block_reason') is not None