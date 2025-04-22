# Main CLI entry point for LLMLogicEvaluator
import argparse
import sys

# Import subcommand functions
from .generation import run_generation
from .querying import run_querying
from .analysis import run_analysis

def main():
    parser = argparse.ArgumentParser(description="LLMLogicEvaluator: Evaluate LLMs on propositional logic.")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help', required=True)

    # --- Generate Subcommand --- (Placeholder arguments)
    parser_generate = subparsers.add_parser('generate', help='Generate logic problems.')
    parser_generate.add_argument("--output", default="problems.jsonl", help="Output file for generated problems (JSON Lines format).")
    parser_generate.add_argument("--count", type=int, default=20, help="Total number of problems per configuration (must be even).")
    parser_generate.add_argument(
        "--resolution-strategy",
        choices=['original', 'standard'],
        default='original',
        help="Resolution strategy for unsatisfiability proofs: 'original' (faster heuristic) or 'standard' (thorough)."
    )
    # TODO: Add arguments for varnr_range, cl_len_range, horn_flags from makeproblems.py
    parser_generate.set_defaults(func=run_generation) # Link to function

    # --- Query Subcommand --- (Placeholder arguments)
    parser_query = subparsers.add_parser('query', help='Query LLMs with generated problems.')
    parser_query.add_argument("--input", default="problems.jsonl", help="Input file containing generated problems.")
    parser_query.add_argument("--output", default="results.jsonl", help="Output file for LLM results (JSON Lines format).")
    parser_query.add_argument("--providers", required=True, nargs='+', help="Provider(s) to use (e.g., openai google)")
    parser_query.add_argument("--models", required=True, nargs='+', help="Model(s) to use (e.g., gpt-4o gemini-1.5-flash)") # Simple list for now
    parser_query.add_argument("--temperature", type=float, help="Sampling temperature.")
    parser_query.add_argument("--max-tokens", type=int, help="Maximum generation tokens.")
    # TODO: Add way to specify different prompts/templates
    parser_query.set_defaults(func=run_querying) # Link to function

    # --- Analyze Subcommand --- (Placeholder arguments)
    parser_analyze = subparsers.add_parser('analyze', help='Analyze LLM results.')
    parser_analyze.add_argument("--input", default="results.jsonl", help="Input file containing LLM results.")
    parser_analyze.add_argument("--report", default="report.txt", help="Output file for analysis report.")
    # TODO: Add arguments for analysis options (grouping, plotting?)
    parser_analyze.set_defaults(func=run_analysis) # Link to function

    args = parser.parse_args()

    # Execute the function associated with the chosen subcommand
    if hasattr(args, 'func'):
        # TODO: Need to pass args to the functions correctly
        # args.func(args) - this passes all args, functions need specific ones
        print(f"Executing command '{args.command}'...")
        # Placeholder call - actual call needs argument mapping
        if args.command == 'generate':
            # Pass the necessary arguments to run_generation
            # Extract generation-specific args, potentially handling varnr_range etc. later
            gen_kwargs = {}
            # TODO: Add logic here to parse range/flag args if they are added
            run_generation(
                output_file=args.output,
                count=args.count,
                resolution_strategy=args.resolution_strategy, # Pass strategy
                **gen_kwargs
            )
        elif args.command == 'query':
            # Need to init aipip service here!
            print("ERROR: Query command needs aipip service initialization!")
            # run_querying(...) # Example call
        elif args.command == 'analyze':
            run_analysis(input_file=args.input, report_file=args.report) # Example call
        else:
            parser.print_help()
            sys.exit(1)
    else:
        # This should not happen if subparsers are required
        parser.print_help()
        sys.exit(1)
    # print(f"Command '{args.command}' chosen. Implementation pending.") # Placeholder execution
    # print("Arguments:", args)

if __name__ == '__main__':
    main() 