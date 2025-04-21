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
.
├── src/
│   └── aipip/              # Main package source code
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   └── loader.py
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── interfaces/
│       │   │   ├── __init__.py
│       │   │   └── text_provider.py
│       │   ├── clients/
│       │   │   ├── __init__.py
│       │   │   ├── openai_client.py
│       │   │   ├── google_client.py
│       │   │   └── anthropic_client.py
│       │   └── registry.py
│       ├── services/
│       │   ├── __init__.py
│       │   └── completion_service.py
│       ├── utils/
│       │   ├── __init__.py
│       │   └── helpers.py
│       └── cli/
│           ├── __init__.py
│           └── run_completion.py
├── tests/
│   └── ...               # Unit and integration tests
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml        # Build system & project metadata
└── main.py               # Optional: Example top-level script (if needed)
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

```bash
# Example assuming installation via pip install .
# (Specific CLI command might change)
python -m aipip.cli.run_completion --provider openai --prompt "Hello world"
```

## Local Development Setup

It is highly recommended to use a virtual environment for local development to isolate project dependencies.

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```

2.  **Activate the environment:**
    *   macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
    *   Windows (Command Prompt/PowerShell):
        ```bash
        .\.venv\Scripts\activate
        ```
    (Your terminal prompt should now show `(.venv)`)

3.  **Install the package in editable mode with development dependencies:**
    ```bash
    pip install -e '.[dev]'
    ```
    The `-e` flag installs the package in "editable" mode, meaning changes to the source code in `src/` will be reflected immediately without needing to reinstall. The `[dev]` part installs the extra dependencies listed under `[project.optional-dependencies.dev]` in `pyproject.toml` (like `pytest`).

4.  **Run tests:**
    With the virtual environment activated, you can run tests using pytest:
    ```bash
    pytest
    ```

5.  **Deactivate the environment** when you're finished:
    ```bash
    deactivate
    ```

## Testing Strategy

This project uses `pytest` as the testing framework.

- Tests are located in the `tests/` directory.
- The structure of `tests/` should mirror the structure of `src/aipip/` where applicable (e.g., tests for `src/aipip/config/` go into `tests/config/`).
- The goal is to achieve good test coverage through a combination of:
    - **Unit Tests:** Testing individual functions, classes, or methods in isolation.
    - **Integration Tests:** Testing the interaction between different components (e.g., a service interacting with a provider client).
- Focus on testing the core logic, public interfaces, and expected behaviors (including edge cases and error handling) of the package components.
- Tests can be run using the `pytest` command after setting up the local development environment (see "Local Development Setup" section).

### Running Tests

Tests can be run using `pytest` after setting up the local development environment.

- **Run all tests (Unit & Integration):**
  ```bash
  pytest
  ```
- **Run only unit tests:** (Faster, no network/API keys needed)
  ```bash
  pytest -m "not integration"
  ```
- **Run only integration tests:** (Requires network and API keys in `.env` or environment)
  ```bash
  pytest -m integration
  ```
- **See output during tests:** To see `print()` statements (like the responses from integration tests), use the `-s` flag:
  ```bash
  pytest -s -m integration # Run integration tests and show output
  pytest -s # Run all tests and show output
  ```

### Integration Tests (`tests/integration/`)

- Integration tests verify the interaction with live provider APIs.
- They are marked with the `integration` marker (configured in `pyproject.toml`).
- **Prerequisites:** Requires valid API keys for the providers being tested. These keys should be stored in a `.env` file in the project root (and this file should be in `.gitignore`) or exported as environment variables (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`). The tests load the `.env` file automatically and will skip if the required key for a specific provider test is not found.
- **Cost & Time:** Be aware that running these tests incurs API costs and takes longer than unit tests.
- **Purpose:** Verify connectivity, authentication, and basic request/response compatibility with the live APIs.

## Handling Upstream API Changes

This library relies on the official Python SDKs provided by the respective AI vendors (e.g., `openai`, `google-generativeai`, `anthropic`). Changes to these upstream SDKs or their underlying APIs can impact `aipip`.

**Strategy:**

1.  **Dependency Management:** We specify version ranges for provider SDKs in `pyproject.toml` (e.g., `openai>=1.0,<2.0`) to prevent automatically pulling in potentially breaking major version updates. Minor/patch updates from providers will be tested before updating the lower bound.
2.  **Interface Stability:** The core `TextProviderInterface` aims for stability. Common parameters are defined explicitly. Provider-specific parameters are handled via `**kwargs` passed directly to the client implementation, allowing flexibility without constant interface changes.
3.  **Client Implementation Responsibility:** Each concrete client class (e.g., `OpenAIClient`) is responsible for adapting to changes in its specific upstream SDK. This involves updating:
    *   Parameter mapping (from interface calls to SDK calls).
    *   Method calls to the SDK.
    *   Response parsing.
4.  **Testing:** Our unit tests for each client (e.g., `test_openai_client.py`) use mocking to simulate the provider SDK. These tests are crucial for detecting when an SDK update breaks our client's implementation, as the mocks or the expected call signatures/responses will no longer align.
5.  **Monitoring & Maintenance:** We will monitor provider announcements and SDK releases. When breaking changes occur in an upstream SDK, the corresponding `aipip` client implementation and its tests will be updated, and a new version of `aipip` will be released.

This approach allows `aipip` to provide a consistent interface while managing the inevitable evolution of the underlying provider APIs and SDKs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

## Contributing

*(Contribution guidelines will be added later)*

## Releasing to PyPI

This project uses [PyPI's trusted publishing](https://docs.pypi.org/trusted-publishers/) for automated releases.

The release process is triggered automatically by pushing a Git tag that matches the version pattern `v*.*.*` (e.g., `v0.1.0`, `v1.2.3`).

The `.github/workflows/publish-to-pypi.yml` GitHub Actions workflow handles:
1.  Building the source distribution and wheel.
2.  Uploading the package to PyPI using the trusted publisher configuration.

No manual API token configuration is required in GitHub secrets.