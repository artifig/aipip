import pytest
from unittest.mock import patch, MagicMock
from pydantic import SecretStr
import openai # Import the actual library to mock its errors

from aipip.providers.clients.openai_client import OpenAIClient, CompletionResponse

# --- Fixtures ---

@pytest.fixture
def mock_openai_api_key() -> SecretStr:
    """Provides a dummy SecretStr API key."""
    return SecretStr("sk-testkey123")

@pytest.fixture
def mock_openai_client_instance():
    """Provides a mock instance of the OpenAI client library."""
    mock_client = MagicMock()

    # Mock chat completion response
    mock_chat_choice = MagicMock()
    mock_chat_choice.message.content = " Mock chat completion text "
    mock_chat_choice.finish_reason = "stop"
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [mock_chat_choice]
    # Configure mock usage object for chat
    mock_chat_usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    mock_chat_usage.dict.return_value = { # Define what .dict() should return
        'prompt_tokens': 10,
        'completion_tokens': 20,
        'total_tokens': 30
    }
    mock_chat_response.usage = mock_chat_usage
    mock_chat_response.model = "gpt-4o-test"
    mock_chat_response.id = "chatcmpl-123"
    mock_chat_response.created = 1677652288
    mock_client.chat.completions.create.return_value = mock_chat_response

    # Mock legacy completion response
    mock_legacy_choice = MagicMock()
    mock_legacy_choice.text = " Mock legacy completion text "
    mock_legacy_choice.finish_reason = "length"
    mock_legacy_response = MagicMock()
    mock_legacy_response.choices = [mock_legacy_choice]
    # Configure mock usage object for legacy
    mock_legacy_usage = MagicMock(prompt_tokens=5, completion_tokens=15, total_tokens=20)
    mock_legacy_usage.dict.return_value = { # Define what .dict() should return
        'prompt_tokens': 5,
        'completion_tokens': 15,
        'total_tokens': 20
    }
    mock_legacy_response.usage = mock_legacy_usage
    mock_legacy_response.model = "gpt-3.5-turbo-instruct-test"
    mock_legacy_response.id = "cmpl-456"
    mock_legacy_response.created = 1677652299
    mock_client.completions.create.return_value = mock_legacy_response

    return mock_client

# --- Test Cases ---

@patch('aipip.providers.clients.openai_client.openai.OpenAI')
def test_init_with_api_key(MockOpenAI, mock_openai_api_key):
    """Test initialization with an explicitly provided API key."""
    client = OpenAIClient(api_key=mock_openai_api_key)
    MockOpenAI.assert_called_once_with(api_key=mock_openai_api_key.get_secret_value())
    assert client.client == MockOpenAI()

@patch('aipip.providers.clients.openai_client.openai.OpenAI')
def test_init_with_env_var(MockOpenAI, mock_openai_api_key, monkeypatch):
    """Test initialization using the environment variable."""
    key_value = mock_openai_api_key.get_secret_value()
    monkeypatch.setenv('OPENAI_API_KEY', key_value)
    client = OpenAIClient()
    MockOpenAI.assert_called_once_with(api_key=key_value)
    assert client.client == MockOpenAI()

@patch('aipip.providers.clients.openai_client.openai.OpenAI')
def test_init_no_key_raises_error(MockOpenAI, monkeypatch):
    """Test ValueError is raised if no key is provided or found in env."""
    # Ensure env var is not set
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(ValueError, match="OpenAI API key not provided"):
        OpenAIClient()
    MockOpenAI.assert_not_called()

def test_generate_completion_with_messages(mock_openai_api_key, mock_openai_client_instance):
    """Test generate_completion using the chat completions endpoint."""
    # Arrange
    messages = [{"role": "user", "content": "Hello"}]
    test_model = "gpt-4o-test"
    test_temp = 0.5
    test_max_tokens = 100
    extra_kwarg = {"top_p": 0.9}

    # Patch the OpenAI class within the module scope only for this test
    with patch('aipip.providers.clients.openai_client.openai.OpenAI', return_value=mock_openai_client_instance) as MockOpenAICtor:
        client = OpenAIClient(api_key=mock_openai_api_key)

        # Act
        response = client.generate_completion(
            messages=messages,
            model=test_model,
            temperature=test_temp,
            max_tokens=test_max_tokens,
            **extra_kwarg
        )

        # Assert
        mock_openai_client_instance.chat.completions.create.assert_called_once_with(
            messages=messages,
            model=test_model,
            temperature=test_temp,
            max_tokens=test_max_tokens,
            top_p=0.9 # Check extra kwarg was passed
        )
        mock_openai_client_instance.completions.create.assert_not_called()
        assert isinstance(response, CompletionResponse)
        assert response.text == "Mock chat completion text"
        assert response.metadata['model'] == test_model
        assert response.metadata['finish_reason'] == "stop"
        assert response.metadata['usage']['total_tokens'] == 30

def test_generate_completion_with_prompt(mock_openai_api_key, mock_openai_client_instance):
    """Test generate_completion using the legacy completions endpoint."""
    # Arrange
    prompt = "Once upon a time"
    test_model = "gpt-3.5-turbo-instruct-test"
    test_temp = 0.8
    test_max_tokens = 50

    with patch('aipip.providers.clients.openai_client.openai.OpenAI', return_value=mock_openai_client_instance) as MockOpenAICtor:
        client = OpenAIClient(api_key=mock_openai_api_key)

        # Act
        response = client.generate_completion(
            prompt=prompt,
            model=test_model,
            temperature=test_temp,
            max_tokens=test_max_tokens
        )

        # Assert
        mock_openai_client_instance.completions.create.assert_called_once_with(
            prompt=prompt,
            model=test_model,
            temperature=test_temp,
            max_tokens=test_max_tokens
        )
        mock_openai_client_instance.chat.completions.create.assert_not_called()
        assert isinstance(response, CompletionResponse)
        assert response.text == "Mock legacy completion text"
        assert response.metadata['model'] == test_model
        assert response.metadata['finish_reason'] == "length"
        assert response.metadata['usage']['total_tokens'] == 20

def test_generate_completion_no_input_raises_error(mock_openai_api_key):
    """Test ValueError is raised if neither prompt nor messages are provided."""
    with patch('aipip.providers.clients.openai_client.openai.OpenAI'): # Need to patch ctor even if not used
        client = OpenAIClient(api_key=mock_openai_api_key)
        with pytest.raises(ValueError, match="Either 'prompt' or 'messages' must be provided"):
            client.generate_completion()

def test_generate_completion_uses_default_model(mock_openai_api_key, mock_openai_client_instance):
    """Test that the default model is used if none is specified."""
    messages = [{"role": "user", "content": "Default?"}]
    with patch('aipip.providers.clients.openai_client.openai.OpenAI', return_value=mock_openai_client_instance) as MockOpenAICtor:
        client = OpenAIClient(api_key=mock_openai_api_key)
        client.generate_completion(messages=messages)
        # Assert model passed to the mock call was the default
        call_args, call_kwargs = mock_openai_client_instance.chat.completions.create.call_args
        assert call_kwargs.get('model') == "gpt-3.5-turbo"

def test_generate_completion_handles_api_error(mock_openai_api_key, mock_openai_client_instance):
    """Test that openai.APIError is caught and re-raised."""
    # Arrange
    messages = [{"role": "user", "content": "Cause error"}]
    mock_openai_client_instance.chat.completions.create.side_effect = openai.APIError("Test API Error", request=None, body=None)

    with patch('aipip.providers.clients.openai_client.openai.OpenAI', return_value=mock_openai_client_instance) as MockOpenAICtor:
        client = OpenAIClient(api_key=mock_openai_api_key)
        # Act & Assert
        with pytest.raises(openai.APIError):
            client.generate_completion(messages=messages)

def test_generate_completion_handles_generic_exception(mock_openai_api_key, mock_openai_client_instance):
    """Test that generic Exceptions are caught and re-raised."""
    # Arrange
    messages = [{"role": "user", "content": "Cause generic error"}]
    mock_openai_client_instance.chat.completions.create.side_effect = Exception("Generic Test Error")

    with patch('aipip.providers.clients.openai_client.openai.OpenAI', return_value=mock_openai_client_instance) as MockOpenAICtor:
        client = OpenAIClient(api_key=mock_openai_api_key)
        # Act & Assert
        with pytest.raises(Exception, match="Generic Test Error"):
            client.generate_completion(messages=messages) 