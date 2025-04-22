import pytest
from unittest.mock import patch, MagicMock
from pydantic import SecretStr
import google.generativeai as genai # Import the actual library
from google.generativeai.types import generation_types, safety_types

from aipip.providers.clients.google_client import GoogleClient, CompletionResponse

# --- Fixtures ---

@pytest.fixture
def mock_google_api_key() -> SecretStr:
    """Provides a dummy SecretStr API key."""
    return SecretStr("google-testkey-xyz")

@pytest.fixture
def mock_google_model_instance():
    """Provides a mock instance of the GenerativeModel."""
    mock_model = MagicMock()

    # Mock the response from generate_content
    mock_response = MagicMock()
    mock_response.text = " Mock Google completion text " # Convenience property
    mock_response.parts = [MagicMock(text=" Mock Google completion text ")] # Simulate parts having text

    # Mock candidate information
    mock_candidate = MagicMock()
    # Mock the .name attribute directly
    mock_candidate.finish_reason.name = "STOP"
    mock_candidate.safety_ratings = [
        # Keep spec for nested safety rating if known, otherwise remove too
        MagicMock(category=safety_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, probability=safety_types.HarmProbability.NEGLIGIBLE)
    ]
    mock_candidate.safety_ratings[0].to_dict.return_value = {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability': 'NEGLIGIBLE'} # Mock to_dict()
    mock_response.candidates = [mock_candidate]

    # Mock prompt feedback (optional)
    mock_response.prompt_feedback = MagicMock()
    # Initialize block_reason to None in the fixture
    mock_response.prompt_feedback.block_reason = None

    mock_model.generate_content.return_value = mock_response
    return mock_model

@pytest.fixture(autouse=True)
def mock_genai_configure():
    """Automatically mock genai.configure for all tests in this file."""
    with patch('aipip.providers.clients.google_client.genai.configure') as mock_configure:
        yield mock_configure

# --- Test Cases ---

@patch('aipip.providers.clients.google_client.genai.GenerativeModel')
def test_init_with_api_key(MockGenerativeModelCtor, mock_google_api_key, mock_genai_configure):
    """Test initialization with an explicitly provided API key."""
    client = GoogleClient(api_key=mock_google_api_key)
    mock_genai_configure.assert_called_once_with(api_key=mock_google_api_key.get_secret_value())
    # We don't directly store the model, so no need to assert client.model == ...
    # Just ensure configure was called correctly.

@patch('aipip.providers.clients.google_client.genai.GenerativeModel')
def test_init_with_env_var(MockGenerativeModelCtor, mock_google_api_key, monkeypatch, mock_genai_configure):
    """Test initialization using the environment variable."""
    key_value = mock_google_api_key.get_secret_value()
    monkeypatch.setenv('GOOGLE_API_KEY', key_value)
    client = GoogleClient()
    mock_genai_configure.assert_called_once_with(api_key=key_value)

@patch('aipip.providers.clients.google_client.genai.GenerativeModel')
def test_init_no_key_raises_error(MockGenerativeModelCtor, monkeypatch, mock_genai_configure):
    """Test ValueError is raised if no key is provided or found in env."""
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    with pytest.raises(ValueError, match="Google API key not provided"):
        GoogleClient()
    mock_genai_configure.assert_not_called()

@patch('aipip.providers.clients.google_client.genai.GenerativeModel')
def test_generate_completion_with_messages(MockGenerativeModelCtor, mock_google_api_key, mock_google_model_instance):
    """Test generate_completion with user/assistant messages."""
    # Arrange
    MockGenerativeModelCtor.return_value = mock_google_model_instance
    messages = [
        {"role": "user", "content": "Hello model"},
        {"role": "assistant", "content": "Hello user"},
        {"role": "user", "content": "How are you?"}
    ]
    test_model = "gemini-pro-test"
    test_temp = 0.6
    test_max_tokens = 99
    extra_kwarg = {"top_k": 40}

    client = GoogleClient(api_key=mock_google_api_key)

    # Act
    response = client.generate_completion(
        messages=messages,
        model=test_model,
        temperature=test_temp,
        max_tokens=test_max_tokens,
        **extra_kwarg
    )

    # Assert Model Initialization
    MockGenerativeModelCtor.assert_called_once_with(
        model_name=test_model,
        system_instruction=None # No system instruction provided
    )

    # Assert generate_content call
    expected_contents = [
        {'role': 'user', 'parts': [{'text': 'Hello model'}]},
        {'role': 'model', 'parts': [{'text': 'Hello user'}]},
        {'role': 'user', 'parts': [{'text': 'How are you?'}]},
    ]
    expected_config = generation_types.GenerationConfig(temperature=test_temp, max_output_tokens=test_max_tokens, top_k=40)
    mock_google_model_instance.generate_content.assert_called_once_with(
        contents=expected_contents,
        generation_config=expected_config,
        safety_settings=None # Not provided
        # No remaining kwargs expected here
    )

    # Assert Response
    assert isinstance(response, CompletionResponse)
    assert response.text == "Mock Google completion text"
    assert response.metadata['model'] == test_model
    assert response.metadata['finish_reason'] == "STOP"
    assert len(response.metadata['safety_ratings']) == 1
    assert response.metadata['safety_ratings'][0]['category'] == 'HARM_CATEGORY_DANGEROUS_CONTENT'

@patch('aipip.providers.clients.google_client.genai.GenerativeModel')
def test_generate_completion_with_prompt(MockGenerativeModelCtor, mock_google_api_key, mock_google_model_instance):
    """Test generate_completion with a simple prompt."""
    # Arrange
    MockGenerativeModelCtor.return_value = mock_google_model_instance
    prompt = "Simple question"
    test_model = "gemini-flash-test"

    client = GoogleClient(api_key=mock_google_api_key)

    # Act
    response = client.generate_completion(prompt=prompt, model=test_model)

    # Assert Model Initialization
    MockGenerativeModelCtor.assert_called_once_with(
        model_name=test_model,
        system_instruction=None
    )

    # Assert generate_content call
    mock_google_model_instance.generate_content.assert_called_once()
    call_args, call_kwargs = mock_google_model_instance.generate_content.call_args
    assert call_kwargs.get('contents') == [prompt]
    # Check that generation_config is created with the default max_tokens
    gen_config = call_kwargs.get('generation_config')
    assert isinstance(gen_config, generation_types.GenerationConfig)
    assert gen_config.max_output_tokens == 1000 # Check the default value
    assert gen_config.temperature is None # Ensure other defaults weren't set
    assert gen_config.top_p is None
    assert gen_config.top_k is None
    assert call_kwargs.get('safety_settings') is None # Check other args

    # Assert Response
    assert isinstance(response, CompletionResponse)
    assert response.text == "Mock Google completion text"

@patch('aipip.providers.clients.google_client.genai.GenerativeModel')
def test_generate_completion_with_system_instruction(MockGenerativeModelCtor, mock_google_api_key, mock_google_model_instance):
    """Test generate_completion correctly extracts and uses system instruction."""
    # Arrange
    MockGenerativeModelCtor.return_value = mock_google_model_instance
    system_prompt = "You are a helpful assistant."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is the weather?"}
    ]
    test_model = "gemini-1.5-pro-test"

    client = GoogleClient(api_key=mock_google_api_key)

    # Act
    response = client.generate_completion(messages=messages, model=test_model)

    # Assert Model Initialization
    MockGenerativeModelCtor.assert_called_once_with(
        model_name=test_model,
        system_instruction=system_prompt # System instruction should be passed here
    )

    # Assert generate_content call
    expected_contents = [
        {'role': 'user', 'parts': [{'text': 'What is the weather?'}]},
    ]
    mock_google_model_instance.generate_content.assert_called_once()
    call_args, call_kwargs = mock_google_model_instance.generate_content.call_args
    assert call_kwargs.get('contents') == expected_contents # System message removed from contents

    # Assert Response
    assert isinstance(response, CompletionResponse)
    assert response.text == "Mock Google completion text"

@patch('aipip.providers.clients.google_client.genai.GenerativeModel')
def test_generate_completion_maps_safety_settings(MockGenerativeModelCtor, mock_google_api_key, mock_google_model_instance):
    """Test that safety_settings kwarg is passed correctly."""
    # Arrange
    MockGenerativeModelCtor.return_value = mock_google_model_instance
    messages = [{"role": "user", "content": "Risky maybe?"}]
    safety_settings_input = {
        safety_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: safety_types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    }

    client = GoogleClient(api_key=mock_google_api_key)

    # Act
    client.generate_completion(messages=messages, safety_settings=safety_settings_input)

    # Assert generate_content call
    mock_google_model_instance.generate_content.assert_called_once()
    call_args, call_kwargs = mock_google_model_instance.generate_content.call_args
    assert call_kwargs.get('safety_settings') == safety_settings_input

@patch('aipip.providers.clients.google_client.genai.GenerativeModel')
def test_generate_completion_no_input_raises_error(MockGenerativeModelCtor, mock_google_api_key):
    """Test ValueError is raised if neither prompt nor messages are provided."""
    client = GoogleClient(api_key=mock_google_api_key)
    with pytest.raises(ValueError, match="Either 'prompt' or 'messages' must be provided"):
        client.generate_completion()

@patch('aipip.providers.clients.google_client.genai.GenerativeModel')
def test_generate_completion_uses_default_model(MockGenerativeModelCtor, mock_google_api_key, mock_google_model_instance):
    """Test that the default model is used if none is specified."""
    # Arrange
    MockGenerativeModelCtor.return_value = mock_google_model_instance
    prompt = "Use default model"
    client = GoogleClient(api_key=mock_google_api_key)

    # Act
    client.generate_completion(prompt=prompt)

    # Assert Model Initialization
    MockGenerativeModelCtor.assert_called_once_with(
        model_name="gemini-1.5-flash", # Check default name
        system_instruction=None
    )

@patch('aipip.providers.clients.google_client.genai.GenerativeModel')
def test_generate_completion_handles_api_error(MockGenerativeModelCtor, mock_google_api_key, mock_google_model_instance):
    """Test that genai API errors are caught and re-raised."""
    # Arrange
    MockGenerativeModelCtor.return_value = mock_google_model_instance
    messages = [{"role": "user", "content": "Cause error"}]
    mock_google_model_instance.generate_content.side_effect = genai.types.generation_types.StopCandidateException("API Error")

    client = GoogleClient(api_key=mock_google_api_key)
    # Act & Assert
    with pytest.raises(genai.types.generation_types.StopCandidateException):
        client.generate_completion(messages=messages)

@patch('aipip.providers.clients.google_client.genai.GenerativeModel')
def test_generate_completion_handles_blocked_prompt(MockGenerativeModelCtor, mock_google_api_key, mock_google_model_instance):
    """Test handling of a response indicating a blocked prompt."""
    # Arrange
    MockGenerativeModelCtor.return_value = mock_google_model_instance
    messages = [{"role": "user", "content": "Blocked content"}]

    # Modify mock response *within this test* to simulate blockage
    mock_response = mock_google_model_instance.generate_content.return_value
    mock_response.parts = [] # No parts returned
    mock_response.text = ""
    # Set the block_reason to a mock object that has a .name attribute
    mock_block_reason = MagicMock()
    mock_block_reason.name = "SAFETY"
    mock_response.prompt_feedback.block_reason = mock_block_reason
    # mock_response.prompt_feedback.to_dict.return_value = {'block_reason': 'SAFETY'} # Removed
    mock_response.candidates = [] # No candidates when blocked usually

    client = GoogleClient(api_key=mock_google_api_key)

    # Act
    response = client.generate_completion(messages=messages)

    # Assert
    assert response.text == "" # Should be empty string
    # Assert against the value we store in metadata
    assert response.metadata['prompt_feedback_block_reason'] == "SAFETY"
    assert response.metadata['finish_reason'] is None # No candidate to get reason from
    assert response.metadata['safety_ratings'] is None 