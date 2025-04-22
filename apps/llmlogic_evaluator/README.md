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
    *   **Output:** Saves problems to a specified file (e.g., `problems.jsonl`) in JSON Lines format. Each line contains problem metadata, definition, and ground truth.
    *   *(Example conceptual arguments: `--output <path>`, `--count <N>`, `--var-range 3 10`, `--len-range 3 4`, `--horn True`)*

2.  **Query LLMs:**
    *   **Command:** `python -m apps.llmlogic_evaluator query [options]`
    *   **Functionality:** Reads problems from the generated file, formats prompts, uses the `aipip` library to query specified providers/models, parses responses, and saves results.
    *   **Output:** Saves detailed results (problem data, prompt, provider/model info, LLM response text, parsed claim, metadata) to a specified file (e.g., `results.jsonl`).
    *   *(Example conceptual arguments: `--input <path>`, `--output <path>`, `--providers openai google`, `--models gpt-4o gemini-1.5-flash`, `--temperature 0.5`, `--max-tokens 100`)*

3.  **Analyze Results:**
    *   **Command:** `python -m apps.llmlogic_evaluator analyze [options]`
    *   **Functionality:** Reads the results file, compares LLM claims to ground truth, calculates accuracy statistics (overall and potentially grouped), and generates a summary report.
    *   **Output:** Saves the analysis report to a specified file (e.g., `report.txt`).
    *   *(Example conceptual arguments: `--input <path>`, `--report <path>`, `--group-by model varnr`)*

Detailed CLI arguments and options will be finalized during implementation.

## Integration with `aipip`

This application relies heavily on the `aipip` library for:

*   Loading API key configuration (`aipip.config`).
*   Instantiating provider clients via the registry (`aipip.providers.registry`).
*   Generating text completions using a unified interface (`aipip.services.TextGenerationService`). 