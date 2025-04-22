# Main CLI entry point for LLMLogicEvaluator
import argparse
import sys
import os # For checking if output file exists

# Import subcommand functions
from .generation import run_generation
from .querying import run_querying
from .analysis import run_analysis

# Import aipip components
from aipip.config.loader import load_config
from aipip.providers.registry import ProviderRegistry
from aipip.services.text_generation_service import TextGenerationService

# Helper function to infer provider from model name (simple version)
def infer_provider(model_name: str) -> str:
    """Infers the provider name based on common model name prefixes.

    Args:
        model_name: The name of the model.

    Returns:
        The inferred provider name (lowercase).

    Raises:
        ValueError: If the provider cannot be inferred from the name.
    """
    model_lower = model_name.lower()
    if model_lower.startswith('claude-'):
        return "anthropic"
    elif model_lower.startswith('gemini-'):
        return "google"
    elif model_lower.startswith('gpt-'):
        return "openai"
    # Add more rules as needed
    else:
        # Default or raise error? Let's raise for now.
        raise ValueError(f"Could not infer provider for model: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="LLMLogicEvaluator: Evaluate LLMs on propositional logic.")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help', required=True)

    # --- Generate Subcommand ---
    parser_generate = subparsers.add_parser('generate', help='Generate logic problems.')
    parser_generate.add_argument("--output", default="problems.jsonl", help="Output file for generated problems (JSON Lines format).")
    parser_generate.add_argument("--count", type=int, default=20, help="Total number of problems per configuration (must be even).")
    parser_generate.add_argument(
        "--resolution-strategy",
        choices=['original', 'standard'],
        default='original',
        help="Resolution strategy for unsatisfiability proofs: 'original' (faster heuristic) or 'standard' (thorough)."
    )
    parser_generate.set_defaults(func=run_generation)

    # --- Query Subcommand ---
    parser_query = subparsers.add_parser('query', help='Query LLMs with generated problems.')
    parser_query.add_argument("--input", default="problems.jsonl", help="Input file containing generated problems.")
    parser_query.add_argument("--output", default="results.jsonl", help="Output file for LLM results (JSON Lines format).")
    parser_query.add_argument("--models", required=True, nargs='+', help="Model name(s) to use (e.g., claude-3-haiku-20240307 gemini-1.5-flash-latest gpt-4o)")
    parser_query.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser_query.add_argument("--max-tokens", type=int, default=1000, help="Maximum generation tokens (default: 1000).")
    parser_query.set_defaults(func=run_querying)

    # --- Analyze Subcommand ---
    parser_analyze = subparsers.add_parser('analyze', help='Analyze LLM results.')
    parser_analyze.add_argument("--input", default="results.jsonl", help="Input file containing LLM results.")
    parser_analyze.add_argument("--report", default="report.txt", help="Output file for analysis report.")
    parser_analyze.set_defaults(func=run_analysis)

    args = parser.parse_args()

    # Shared aipip setup (only needed for query command)
    aipip_service = None
    if args.command == 'query':
        print("Initializing aipip service...")
        try:
            config = load_config()
            registry = ProviderRegistry(config=config)
            aipip_service = TextGenerationService(registry=registry)
            print("aipip service initialized successfully.")
        except Exception as e:
            print(f"Error initializing aipip components: {e}")
            print("Please ensure API keys are set in .env or environment variables.")
            sys.exit(1)

        # Check if output file exists and warn user about appending
        if os.path.exists(args.output):
            print(f"Warning: Output file '{args.output}' already exists. Results will be appended.")
            user_input = input("Continue? (y/n): ").lower()
            if user_input != 'y':
                print("Aborted.")
                sys.exit(0)

    # Execute the function associated with the chosen subcommand
    print(f"\nExecuting command '{args.command}'...")
    if args.command == 'generate':
        gen_kwargs = {}
        # TODO: Parse range/flag args for generate
        run_generation(
            output_file=args.output,
            count=args.count,
            resolution_strategy=args.resolution_strategy,
            **gen_kwargs
        )
    elif args.command == 'query':
        if aipip_service:
            query_kwargs = {
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            # Convert models to list of (provider, model) tuples
            model_provider_list = []
            try:
                for model_name in args.models:
                    provider_name = infer_provider(model_name)
                    model_provider_list.append((provider_name, model_name))
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)

            run_querying(
                input_file=args.input,
                output_file=args.output,
                service=aipip_service,
                model_provider_list=model_provider_list, # Pass list of tuples
                **query_kwargs
            )
        else:
            # Should have exited earlier if service failed to init
            print("Error: aipip service not available for query command.")
            sys.exit(1)

    elif args.command == 'analyze':
        # TODO: Pass args correctly to run_analysis
        run_analysis(input_file=args.input, report_file=args.report)
    else:
        # Should not be reachable due to required=True
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main() 