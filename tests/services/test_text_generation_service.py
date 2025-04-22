import pytest
from unittest.mock import MagicMock, patch

from aipip.services.text_generation_service import TextGenerationService
from aipip.providers.registry import ProviderRegistry
from aipip.providers.interfaces.text_provider import TextProviderInterface, CompletionResponse

# --- Fixtures ---

@pytest.fixture
def mock_provider_registry() -> MagicMock:
    """Fixture for a mocked ProviderRegistry."""
    return MagicMock(spec=ProviderRegistry)

@pytest.fixture
def mock_provider_client() -> MagicMock:
    """Fixture for a mocked TextProviderInterface (like OpenAIClient)."""
    mock_client = MagicMock(spec=TextProviderInterface)
    # Configure the mock response from the client's generate_completion
    mock_response = CompletionResponse(
        text="Mocked completion text",
        provider_name="mock_provider",
        metadata={'model': 'mock-model'}
    )
    mock_client.generate_completion.return_value = mock_response
    return mock_client

# --- Test Cases ---

def test_service_init(mock_provider_registry):
    """Test service initialization."""
    service = TextGenerationService(registry=mock_provider_registry)
    assert service._registry == mock_provider_registry

def test_generate_calls_registry_and_client(mock_provider_registry, mock_provider_client):
    """Test that generate gets provider from registry and calls its method."""
    # Arrange
    mock_provider_registry.get_provider.return_value = mock_provider_client
    service = TextGenerationService(registry=mock_provider_registry)

    provider = "openai"
    prompt = "Test prompt"
    model = "test-model"
    temp = 0.5
    max_tok = 100
    kw = {"top_p": 0.9}

    # Act
    response = service.generate(
        provider_name=provider,
        prompt=prompt,
        model=model,
        temperature=temp,
        max_tokens=max_tok,
        **kw
    )

    # Assert
    # 1. Check registry was called correctly
    mock_provider_registry.get_provider.assert_called_once_with(provider)
    # 2. Check the retrieved client's method was called correctly
    mock_provider_client.generate_completion.assert_called_once_with(
        prompt=prompt,
        messages=None, # Ensure messages is None if not passed
        model=model,
        temperature=temp,
        max_tokens=max_tok,
        top_p=0.9 # Check kwargs are passed
    )
    # 3. Check the response is the one returned by the mock client
    assert isinstance(response, CompletionResponse)
    assert response.text == "Mocked completion text"
    assert response.metadata['model'] == 'mock-model'

def test_generate_passes_messages(mock_provider_registry, mock_provider_client):
    """Test that generate passes messages correctly."""
    # Arrange
    mock_provider_registry.get_provider.return_value = mock_provider_client
    service = TextGenerationService(registry=mock_provider_registry)

    provider = "google"
    messages = [{"role": "user", "content": "Hi"}]

    # Act
    service.generate(provider_name=provider, messages=messages)

    # Assert
    mock_provider_registry.get_provider.assert_called_once_with(provider)
    mock_provider_client.generate_completion.assert_called_once()
    call_args, call_kwargs = mock_provider_client.generate_completion.call_args
    assert call_kwargs.get('messages') == messages
    assert call_kwargs.get('prompt') is None # Ensure prompt is None 