# LLMLogicEvaluator: Evaluating LLM Performance on Propositional Logic

## Overview

This application evaluates the performance of Large Language Models (LLMs) on propositional logic problems. It uses the `aipip` library to interact with various AI providers and orchestrates a workflow involving problem generation, LLM querying, and results analysis.

The goal is to assess how well different LLMs and configurations can determine the satisfiability of propositional logic formulas.

## Attribution and License

This application is heavily based on the concepts and code structure from the `llmlog` repository by Tanel Tammet:

*   **Original Repository:** [https://github.com/tammet/llmlog](https://github.com/tammet/llmlog)
*   **Original Author:** Tanel Tammet
*   **Original License:** Apache-2.0 License

The refactored code within this application aims to maintain compliance with the original Apache-2.0 license. Please include the original copyright notice and license file if distributing or modifying this derived work significantly.

## Structure

This application is structured as a Python package (`llmlogic_evaluator`) with a command-line interface driven by subcommands:

*   **`llmlogic_evaluator/generation.py`:** Contains the core logic for generating problem datasets.
*   **`llmlogic_evaluator/querying.py`:** Contains the core logic for querying LLMs using the `aipip` library.
*   **`llmlogic_evaluator/analysis.py`:** Contains the core logic for analyzing the results.
*   **`llmlogic_evaluator/cli.py`:** Defines the main CLI structure using `argparse` and subparsers (`generate`, `query`, `analyze`). It imports and calls functions from the core logic modules.
*   **`llmlogic_evaluator/__main__.py`:** Allows running the application using `python -m llmlogic_evaluator`.

This structure separates the core application logic from the CLI handling, improving testability and reusability.

## Workflow & Usage

The evaluation process involves three main stages, executed via CLI subcommands:

1.  **Generate Problems:**
    *   **Command:** `python -m apps.llmlogic_evaluator generate [options]`
    *   **Functionality:** Creates a dataset of propositional logic problems based on specified parameters (variable count, clause length, Horn property, total count).
    *   **Output:** Saves problems to a specified file (default: `apps/llmlogic_evaluator/output/problems.jsonl`) in JSON Lines format. Each line contains problem metadata, definition, and ground truth.
    *   **Key Options:**
        *   `--output <path>`: Set output file (default: `apps/llmlogic_evaluator/output/problems.jsonl`).
        *   `--count <N>`: Problems per configuration (default: 20, must be even).
        *   `--resolution-strategy {original|standard}`: Selects the resolution proving algorithm used for unsatisfiable problems (default: `original`). This affects the generation of the `proof_or_model` field for unsatisfiable problems.
            *   **`original` (Default):** Uses the specific resolution heuristic found in the original `makeproblems.py` script. 
                *   **Pros:** Significantly faster, especially for problems with many variables (e.g., 14+).
                *   **Cons:** Less standard algorithmically.
                *   **Use Case:** Best for quickly generating large datasets.
            *   **`standard`:** Implements a more conventional resolution step.
                *   **Pros:** More thorough exploration, adheres to a standard algorithm.
                *   **Cons:** Can be substantially slower for complex unsatisfiable problems.
                *   **Use Case:** Suitable when adherence to a standard process is preferred.
        *   *(TODO: Add arguments for `--var-range`, `--len-range`, `--horn-flags`)*
    *   **Examples:**
        ```bash
        # Generate default problems to apps/llmlogic_evaluator/output/problems.jsonl
        python -m apps.llmlogic_evaluator generate

        # Generate 100 problems per case to a specific file
        python -m apps.llmlogic_evaluator generate --output apps/llmlogic_evaluator/output/problems_100.jsonl --count 100

        # Generate default problems using the standard resolution strategy
        python -m apps.llmlogic_evaluator generate --resolution-strategy standard
        ```

2.  **Query LLMs:**
    *   **Command:** `python -m apps.llmlogic_evaluator query [options]`
    *   **Functionality:** Reads problems from the generated file, formats prompts, uses the `aipip` library to query specified providers/models, parses responses, and saves results.
    *   **Output:** Saves detailed results to a specified file (default: `apps/llmlogic_evaluator/output/results.jsonl`). Results are appended if the file exists.
    *   **Key Options:**
        *   `--input <path>`: Input problems file (default: `apps/llmlogic_evaluator/output/problems.jsonl`).
        *   `--output <path>`: Output results file (default: `apps/llmlogic_evaluator/output/results.jsonl`, appends if exists).
        *   `--models <name1> [<name2> ...]`: One or more model names (required, e.g., `claude-3-sonnet-20240229`, `gpt-4o`).
        *   `--temperature <float>`: Set sampling temperature (optional).
        *   `--max-tokens <int>`: Set max tokens for response (optional, defaults to 1000).
    *   **Examples:**
        ```bash
        # Query OpenAI's gpt-4o using default input/output files
        python -m apps.llmlogic_evaluator query --models gpt-4o

        # Query multiple models, setting max_tokens and saving to a specific file
        python -m apps.llmlogic_evaluator query --models claude-3-haiku-20240307 gpt-4o --max-tokens 150 --output apps/llmlogic_evaluator/output/run1_results.jsonl

        # Query a Google model using problems from a specific file
        python -m apps.llmlogic_evaluator query --input apps/llmlogic_evaluator/output/problems_custom.jsonl --models gemini-1.5-flash-latest
        ```

3.  **Analyze Results:**
    *   **Command:** `python -m apps.llmlogic_evaluator analyze [options]`
    *   **Functionality:** Reads the results file, compares LLM claims to ground truth, calculates accuracy statistics, and generates a summary report.
    *   **Output:** Saves the analysis report to a specified file (default: `apps/llmlogic_evaluator/output/report.txt`).
    *   **Key Options:**
        *   `--input <path>`: Input results file (default: `apps/llmlogic_evaluator/output/results.jsonl`).
        *   `--report <path>`: Output report file (default: `apps/llmlogic_evaluator/output/report.txt`).
        *   *(TODO: Add arguments for analysis options like grouping)*
    *   **Examples:**
        ```bash
        # Analyze default results.jsonl and save to apps/llmlogic_evaluator/output/report.txt
        python -m apps.llmlogic_evaluator analyze

        # Analyze a specific results file and save to a different report file
        python -m apps.llmlogic_evaluator analyze --input apps/llmlogic_evaluator/output/run1_results.jsonl --report apps/llmlogic_evaluator/output/run1_analysis.txt
        ```

Detailed CLI arguments and options will be finalized during implementation.

## Integration with `aipip`

This application relies heavily on the `aipip` library for:

*   Loading API key configuration (`aipip.config`).
*   Instantiating provider clients via the registry (`aipip.providers.registry`).
*   Generating text completions using a unified interface (`aipip.services.TextGenerationService`).

## Testing Strategy

While the core `aipip` library has its own tests, applications built on top of it, like `llmlogic_evaluator`, require their own testing strategy focused on the application-specific logic and workflow.

Tests for this application should reside within the `apps/llmlogic_evaluator/tests/` directory.

1.  **Unit Tests (High Priority):**
    *   **Location:** `apps/llmlogic_evaluator/tests/unit/`
    *   **Focus:** Test individual functions within `generation.py`, `querying.py`, `analysis.py`, and `cli.py` in isolation.
    *   **Key Areas:**
        *   **Generation:** Verify correctness of solvers (`truth_table_solve`, `solve_prop_problem`, etc.) with known small inputs. Test helper logic (`normalize_problem`). Test balancing logic by mocking solvers.
        *   **Querying:** Test prompt formatting, response parsing (for various LLM outputs), and the main `run_querying` logic (crucially, **mocking** the `aipip.services.TextGenerationService` to check inputs and simulate outputs).
        *   **Analysis:** Test accuracy calculations, grouping logic, and basic report structure.
        *   **CLI:** Test argument parsing and that the correct `run_...` function is called based on subcommands.

2.  **Integration Tests (Medium Priority):**
    *   **Location:** `apps/llmlogic_evaluator/tests/integration/`
    *   **Focus:** Test the interaction between the application's modules, file I/O, and mocked external systems.
    *   **Key Areas:**
        *   **File Handling:** Use `pytest`'s `tmp_path` fixture to test reading/writing of `problems.jsonl`, `results.jsonl`, `report.txt`, ensuring correct formats (e.g., JSON Lines).
        *   **Workflow (Mocked Service):** Test the full `generate -> query -> analyze` flow driven by `cli.py`, but with a **mocked `TextGenerationService`**. This verifies the orchestration and data handoffs without live API calls.

3.  **End-to-End Tests (Lower Priority / Optional):**
    *   **Focus:** Testing the complete user workflow, potentially including live API calls.
    *   **Challenges:** Difficult to automate reliably due to external dependencies (cost, time, non-determinism). The mocked integration tests often provide sufficient confidence.

**Running Tests:**

Application-specific tests can be run using `pytest`. You might need to configure `pytest` (e.g., in `pyproject.toml`) to discover tests within the `apps/` directory or run `pytest` specifically targeting that path:

```bash
pytest apps/llmlogic_evaluator/tests/
``` 