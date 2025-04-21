import pytest
from unittest.mock import patch, MagicMock
from pydantic import SecretStr
import anthropic # Import the actual library

from aipip.providers.clients.anthropic_client import AnthropicClient, CompletionResponse, _prepare_anthropic_messages

# --- Fixtures ---

@pytest.fixture
def mock_anthropic_api_key() -> SecretStr:
    """Provides a dummy SecretStr API key."""
    return SecretStr("anthropic-testkey-xyz")

@pytest.fixture
def mock_anthropic_client_instance():
    """Provides a mock instance of the Anthropic client library."""
    mock_client = MagicMock()

    # Mock the response from messages.create
    mock_response = MagicMock(spec=anthropic.types.Message)
    mock_response.content = [MagicMock(text=" Mock Anthropic completion text ")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage = MagicMock(input_tokens=15, output_tokens=25)
    mock_response.model = "claude-3-test"
    mock_response.id = "msg_01AptSBDDvuYxRwpC3wG53E4"
    mock_response.role = "assistant"
    mock_response.type = "message"
    mock_response.stop_sequence = None

    mock_client.messages.create.return_value = mock_response
    return mock_client

# --- Helper Function Tests (_prepare_anthropic_messages) ---

def test_prepare_anthropic_messages_valid():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Test"}
    ]
    expected = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Test"}
    ]
    assert _prepare_anthropic_messages(messages) == expected

def test_prepare_anthropic_messages_starts_with_assistant_error():
    messages = [{"role": "assistant", "content": "Hi"}]
    with pytest.raises(ValueError, match="Expected 'user' role but got 'assistant' at index 0"):
        _prepare_anthropic_messages(messages)

def test_prepare_anthropic_messages_consecutive_user_error():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Again?"}
    ]
    with pytest.raises(ValueError, match="Expected 'assistant' role but got 'user' at index 1"):
        _prepare_anthropic_messages(messages)

def test_prepare_anthropic_messages_consecutive_assistant_error():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "assistant", "content": "Again!"}
    ]
    with pytest.raises(ValueError, match="Expected 'user' role but got 'assistant' at index 2"):
        _prepare_anthropic_messages(messages)

def test_prepare_anthropic_messages_system_message_error():
    messages = [{"role": "system", "content": "Act like a pirate"}]
    with pytest.raises(ValueError, match="System messages should be handled"):
        _prepare_anthropic_messages(messages)

def test_prepare_anthropic_messages_empty_error():
    with pytest.raises(ValueError, match="Anthropic requires at least one message"):
        _prepare_anthropic_messages([])

# --- Client Test Cases ---

@patch('aipip.providers.clients.anthropic_client.anthropic.Anthropic')
def test_init_with_api_key(MockAnthropic, mock_anthropic_api_key):
    """Test initialization with an explicitly provided API key."""
    client = AnthropicClient(api_key=mock_anthropic_api_key)
    MockAnthropic.assert_called_once_with(api_key=mock_anthropic_api_key.get_secret_value())
    assert client.client == MockAnthropic()

@patch('aipip.providers.clients.anthropic_client.anthropic.Anthropic')
def test_init_with_env_var(MockAnthropic, mock_anthropic_api_key, monkeypatch):
    """Test initialization using the environment variable."""
    key_value = mock_anthropic_api_key.get_secret_value()
    monkeypatch.setenv('ANTHROPIC_API_KEY', key_value)
    client = AnthropicClient()
    MockAnthropic.assert_called_once_with(api_key=key_value)
    assert client.client == MockAnthropic()

@patch('aipip.providers.clients.anthropic_client.anthropic.Anthropic')
def test_init_no_key_raises_error(MockAnthropic, monkeypatch):
    """Test ValueError is raised if no key is provided or found in env."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    with pytest.raises(ValueError, match="Anthropic API key not provided"):
        AnthropicClient()
    MockAnthropic.assert_not_called()

@patch('aipip.providers.clients.anthropic_client.anthropic.Anthropic')
def test_generate_completion_with_messages(MockAnthropicCtor, mock_anthropic_api_key, mock_anthropic_client_instance):
    """Test generate_completion with standard user/assistant messages."""
    # Arrange
    MockAnthropicCtor.return_value = mock_anthropic_client_instance
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "Tell me a joke"}
    ]
    test_model = "claude-3-opus-test"
    test_temp = 0.5
    test_max_tokens = 500
    extra_kwarg = {"top_k": 50}

    client = AnthropicClient(api_key=mock_anthropic_api_key)

    # Act
    response = client.generate_completion(
        messages=messages,
        model=test_model,
        temperature=test_temp,
        max_tokens=test_max_tokens,
        **extra_kwarg
    )

    # Assert messages.create call
    expected_anthropic_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "Tell me a joke"}
    ]
    mock_anthropic_client_instance.messages.create.assert_called_once_with(
        model=test_model,
        messages=expected_anthropic_messages,
        temperature=test_temp,
        max_tokens=test_max_tokens,
        top_k=50, # Check extra kwarg was passed
    )

    # Assert Response
    assert isinstance(response, CompletionResponse)
    assert response.text == "Mock Anthropic completion text"
    assert response.metadata['model'] == mock_anthropic_client_instance.messages.create.return_value.model
    assert response.metadata['finish_reason'] == "end_turn"
    assert response.metadata['usage'] == {'input_tokens': 15, 'output_tokens': 25}

@patch('aipip.providers.clients.anthropic_client.anthropic.Anthropic')
def test_generate_completion_with_prompt(MockAnthropicCtor, mock_anthropic_api_key, mock_anthropic_client_instance):
    """Test generate_completion converting prompt to user message."""
    # Arrange
    MockAnthropicCtor.return_value = mock_anthropic_client_instance
    prompt = "Simple prompt"
    test_model = "claude-3-sonnet-test"

    client = AnthropicClient(api_key=mock_anthropic_api_key)

    # Act
    response = client.generate_completion(prompt=prompt, model=test_model)

    # Assert messages.create call
    expected_anthropic_messages = [{"role": "user", "content": prompt}]
    mock_anthropic_client_instance.messages.create.assert_called_once()
    call_args, call_kwargs = mock_anthropic_client_instance.messages.create.call_args
    assert call_kwargs.get('messages') == expected_anthropic_messages
    assert call_kwargs.get('model') == test_model
    assert call_kwargs.get('system') is None
    assert call_kwargs.get('max_tokens') == 1024 # Default

    # Assert Response
    assert isinstance(response, CompletionResponse)
    assert response.text == "Mock Anthropic completion text"

@patch('aipip.providers.clients.anthropic_client.anthropic.Anthropic')
def test_generate_completion_with_system_message(MockAnthropicCtor, mock_anthropic_api_key, mock_anthropic_client_instance):
    """Test generate_completion extracting system message from messages list."""
    # Arrange
    MockAnthropicCtor.return_value = mock_anthropic_client_instance
    system_prompt = "You are a poet."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Write a haiku about tests."}
    ]
    test_model = "claude-3-haiku-test"

    client = AnthropicClient(api_key=mock_anthropic_api_key)

    # Act
    response = client.generate_completion(messages=messages, model=test_model)

    # Assert messages.create call
    expected_anthropic_messages = [{"role": "user", "content": "Write a haiku about tests."}]
    mock_anthropic_client_instance.messages.create.assert_called_once()
    call_args, call_kwargs = mock_anthropic_client_instance.messages.create.call_args
    assert call_kwargs.get('system') == system_prompt
    assert call_kwargs.get('messages') == expected_anthropic_messages
    assert call_kwargs.get('model') == test_model

@patch('aipip.providers.clients.anthropic_client.anthropic.Anthropic')
def test_generate_completion_with_system_kwarg(MockAnthropicCtor, mock_anthropic_api_key, mock_anthropic_client_instance):
    """Test generate_completion using system kwarg directly."""
    # Arrange
    MockAnthropicCtor.return_value = mock_anthropic_client_instance
    system_prompt = "You are a helpful bot."
    messages = [{"role": "user", "content": "User query"}]
    test_model = "claude-3-sonnet-test"

    client = AnthropicClient(api_key=mock_anthropic_api_key)

    # Act
    response = client.generate_completion(messages=messages, model=test_model, system=system_prompt)

    # Assert messages.create call
    mock_anthropic_client_instance.messages.create.assert_called_once()
    call_args, call_kwargs = mock_anthropic_client_instance.messages.create.call_args
    assert call_kwargs.get('system') == system_prompt
    assert call_kwargs.get('messages') == messages # Original messages passed


def test_generate_completion_no_input_raises_error(mock_anthropic_api_key):
    """Test ValueError is raised if neither prompt nor messages are provided."""
    with patch('aipip.providers.clients.anthropic_client.anthropic.Anthropic'):
        client = AnthropicClient(api_key=mock_anthropic_api_key)
        with pytest.raises(ValueError, match="Either 'prompt' or 'messages' must be provided"):
            client.generate_completion()

def test_generate_completion_no_max_tokens_raises_error(mock_anthropic_api_key):
    """Test ValueError is raised if max_tokens is explicitly None."""
    with patch('aipip.providers.clients.anthropic_client.anthropic.Anthropic'):
        client = AnthropicClient(api_key=mock_anthropic_api_key)
        with pytest.raises(ValueError, match="'max_tokens' is required"):
            client.generate_completion(prompt="test", max_tokens=None)

@patch('aipip.providers.clients.anthropic_client.anthropic.Anthropic')
def test_generate_completion_uses_default_model(MockAnthropicCtor, mock_anthropic_api_key, mock_anthropic_client_instance):
    """Test that the default model is used if none is specified."""
    MockAnthropicCtor.return_value = mock_anthropic_client_instance
    prompt = "Use default"
    client = AnthropicClient(api_key=mock_anthropic_api_key)
    client.generate_completion(prompt=prompt)
    mock_anthropic_client_instance.messages.create.assert_called_once()
    call_args, call_kwargs = mock_anthropic_client_instance.messages.create.call_args
    assert call_kwargs.get('model') == "claude-3-sonnet-20240229"

@patch('aipip.providers.clients.anthropic_client.anthropic.Anthropic')
def test_generate_completion_handles_api_error(MockAnthropicCtor, mock_anthropic_api_key, mock_anthropic_client_instance):
    """Test that anthropic.APIError is caught and re-raised."""
    MockAnthropicCtor.return_value = mock_anthropic_client_instance
    messages = [{"role": "user", "content": "Cause error"}]
    mock_anthropic_client_instance.messages.create.side_effect = anthropic.APIError("Test API Error", request=MagicMock(), body=None)

    client = AnthropicClient(api_key=mock_anthropic_api_key)
    with pytest.raises(anthropic.APIError):
        client.generate_completion(messages=messages) 