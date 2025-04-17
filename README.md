# AIPIP: AI Provider Interaction Platform

## Vision

This project, AIPIP (AI Provider Interaction Platform), aims to build a flexible and extensible Python platform for interacting with various AI provider APIs. While the initial focus is on text generation using Large Language Models (LLMs), the architecture is designed to accommodate future expansion into other modalities like image generation, audio processing, streaming responses, and more complex multi-modal interactions as AI capabilities evolve.

## Architecture Overview

The platform follows a modular, service-oriented architecture to promote decoupling, testability, and maintainability.

**Note on Inspiration:** This architecture draws inspiration from projects like [`aisuite`](https://github.com/andrewyng/aisuite), which also provide a unified interface to multiple AI providers. However, we have chosen a custom layered approach with a distinct **Service Layer** to better support the goal of building a broader application platform. This separation allows for more flexibility in integrating diverse functionalities (e.g., complex evaluation workflows, data analysis, problem generation) on top of the core provider interactions, leading to better separation of concerns and maintainability as the application scope grows beyond simple API calls.

1.  **Configuration (`config/`)**:
    *   Uses Pydantic models for defining structured and validated configurations.
    *   Centralized loading mechanism (e.g., from environment variables, `.env` files, or dedicated config files) to manage API keys, provider settings, model parameters, etc.
    *   Secure handling of sensitive information like API keys.

2.  **Provider Abstraction (`providers/interfaces/`)**:
    *   Defines abstract base classes (ABCs) or interfaces for different types of AI interactions (e.g., `TextProviderInterface`, `ImageProviderInterface`).
    *   These interfaces enforce a common set of methods (e.g., `generate_completion`, `generate_image`) that specific provider implementations must adhere to.

3.  **Provider Implementations (`providers/clients/`)**:
    *   Concrete classes implementing the provider interfaces for specific vendors (e.g., `OpenAIClient`, `GoogleClient`, `AnthropicClient` implementing `TextProviderInterface`).
    *   Each client class encapsulates the logic for interacting with a specific provider's SDK/API.
    *   Clients receive their necessary configuration (API key, etc.) via dependency injection during initialization, making them stateless regarding configuration loading.

4.  **Provider Registry/Factory (`providers/registry.py`)**:
    *   A central component responsible for instantiating and managing provider client objects based on the loaded application configuration.
    *   Provides a way for other parts of the application (like services) to request and obtain initialized provider instances without needing to know the instantiation details.

5.  **Service Layer (`services/`)**:
    *   Contains modules with specific business logic (e.g., `CompletionService`, `EvaluationService`, `AnalysisService`).
    *   Services depend on the Provider Registry to get the necessary provider clients via their interfaces.
    *   Encapsulates workflows and orchestrates calls to providers.

6.  **Application Entry Points (`cli/`, `api/`)**:
    *   Entry points for interacting with the application (e.g., Command-Line Interface scripts, a future Web API/UI).
    *   These components initialize the configuration, the provider registry, and the required services, then invoke service methods.

7.  **Utilities (`utils/`)**:
    *   Shared helper functions and classes used across different parts of the application.

8.  **Testing (`tests/`)**:
    *   Comprehensive unit and integration tests for all components, facilitated by the decoupled architecture and dependency injection.

9.  **Tool Calling / Function Calling**:
    *   Support for provider-specific tool/function calling mechanisms will be integrated.
    *   The common `TextProviderInterface` will likely include methods or parameters to pass tool schemas and receive tool invocation requests from the LLM.
    *   Provider client implementations (`providers/clients/`) will handle the specific API interactions for tool use.
    *   Services (`services/`) can then orchestrate multi-turn conversations involving tool execution, potentially drawing inspiration from patterns like `aisuite`'s automatic execution flow in later phases.

## Proposed Directory Structure

```
config/
│   ├── __init__.py
│   ├── models.py         # Pydantic models for configuration
│   └── loader.py         # Logic for loading config (env vars, files)
providers/
│   ├── __init__.py
│   ├── interfaces/       # Abstract base classes for providers
│   │   ├── __init__.py
│   │   └── text_provider.py
│   │   # └── image_provider.py (Future)
│   │   # └── audio_provider.py (Future)
│   ├── clients/          # Concrete implementations for each provider
│   │   ├── __init__.py
│   │   ├── openai_client.py
│   │   ├── google_client.py
│   │   ├── anthropic_client.py
│   │   # └── stability_client.py (Future)
│   └── registry.py       # Provider Registry/Factory
services/
│   ├── __init__.py
│   ├── completion_service.py # Example service using Text Providers
│   # └── evaluation_service.py (Future/Refactored)
│   # └── analysis_service.py   (Future)
utils/
│   ├── __init__.py
│   └── helpers.py        # General utility functions
cli/
│   ├── __init__.py
│   └── run_completion.py # Example CLI entry point
│   # └── run_evaluation.py (Future/Refactored)
tests/
│   # Unit and integration tests mirroring the structure
main.py               # Potential main application runner (e.g., for API)
README.md             # This file
LICENSE               # License file
pyproject.toml        # Build system config (to be added)
setup.cfg             # Package config (to be added)
```

## Roadmap & Current Status

This README outlines the target architecture. We will migrate functionality from the old structure progressively.

**Phase 1: Core Text Generation Setup (In Progress)**

*   [ ] **Configuration System:** Define Pydantic models (`config/models.py`) and loading mechanism (`config/loader.py`).
*   [ ] **Text Provider Interface:** Define `TextProviderInterface` (`providers/interfaces/text_provider.py`).
*   [ ] **Provider Implementations:** Refactor `openai_client.py`, `google_client.py`, `anthropic_client.py` into classes implementing the interface (`providers/clients/`). Ensure they accept configuration via `__init__`.
*   [ ] **Provider Registry:** Implement `ProviderRegistry` (`providers/registry.py`) to instantiate and provide clients.
*   [ ] **Completion Service:** Create an initial `CompletionService` (`services/completion_service.py`) using the registry and text providers.
*   [ ] **Basic CLI Entry Point:** Create a simple CLI script (`cli/run_completion.py`) to test the new structure.
*   [ ] **Unit Tests:** Add basic unit tests for config loading, registry, and provider clients (using mocks).

**Phase 2: Migrate Existing Functionality & Enhance Core**

*   [ ] Refactor `run_logic_eval.py` logic into an `EvaluationService`.
*   [ ] Adapt prompt generation (`makeprompt_v1`) and result parsing (`parse_result`) into reusable utilities or part of the `EvaluationService`.
*   [ ] Refactor `analyze_results.py` logic into an `AnalysisService`.
*   [ ] **Tool Calling Support:** Implement basic tool/function calling capabilities in the text provider interface and clients.
*   [ ] Update CLI entry points (`cli/run_evaluation.py`, `cli/analyze_results.py`).
*   [ ] Add comprehensive tests for migrated services and tool calling.

**Phase 3: Future Enhancements (Examples)**

*   [ ] Image Generation Provider Interface & Implementations
*   [ ] Audio Processing Provider Interface & Implementations
*   [ ] Streaming Support in Providers & Services
*   [ ] Advanced Error Handling, Retries, and Rate Limiting
*   [ ] **Adapting to Evolving Standards:** Monitor and adapt provider clients and interfaces to support emerging standards for structured context communication (e.g., Anthropic's Model Context Protocol - MCP) as they gain adoption.
*   [ ] Asynchronous Provider Implementations (`asyncio`)
*   [ ] Problem Generation Service
*   [ ] Web API (e.g., using FastAPI)
*   [ ] User Interface
*   [ ] Deployment Setup (Docker, CI/CD)
*   [ ] **Advanced Tool Calling:** Implement more sophisticated tool handling (e.g., automatic execution flows).

*(This list will be updated as the project progresses)*

## Setup & Usage

*(Instructions will be added here once the core components are functional)*

```
# Example (Placeholder)
pip install -r requirements.txt
export OPENAI_API_KEY="your_key"
export GOOGLE_API_KEY="your_key"
# ... other env vars ...

python cli/run_completion.py --provider openai --prompt "Hello world"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

## Contributing

*(Contribution guidelines will be added later)*