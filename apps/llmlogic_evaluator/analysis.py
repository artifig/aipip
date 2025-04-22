# Logic for analyzing LLM results
# Adapted from https://github.com/tammet/llmlog/blob/main/analyze.py by Tanel Tammet
# Original License: Apache-2.0

import json
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter
import statistics # For potential future use (e.g., std dev)

# Type definition for cleaner code
Stats = Dict[str, int] # e.g., {'total': 10, 'correct': 8, ...}
ModelStats = Dict[str, Stats] # e.g., {'gpt-4o': Stats, ...}
GroupedStats = Dict[Tuple, Stats] # e.g., {(3, 3, True): Stats, ...}
ModelGroupedStats = Dict[str, GroupedStats] # e.g., {'gpt-4o': GroupedStats, ...}

def run_analysis(input_file: str, report_file: str, **kwargs):
    """Analyzes LLM query results from a JSON Lines file.

    Calculates overall and per-model accuracy based on comparing the
    LLM's parsed claim to the ground truth satisfiability.

    Args:
        input_file: Path to the input results file (JSON Lines).
        report_file: Path to the output analysis report file.
        **kwargs: Additional analysis parameters (e.g., group_by - not currently implemented).
    """
    print(f"Analyzing results from {input_file}...")

    # Initialize statistics counters
    # We'll group by model primarily
    model_stats: ModelGroupedStats = defaultdict(lambda: defaultdict(lambda: Counter()))
    # Counter keys: 'total', 'correct', 'sat_correct', 'unsat_correct', 'unknown'

    results_read = 0
    errors_parsing = 0

    try:
        with open(input_file, 'r') as infile:
            for line_num, line in enumerate(infile, 1):
                try:
                    result_data = json.loads(line.strip())
                    results_read += 1

                    # Extract necessary data points
                    model = result_data["query_info"]["model"]
                    ground_truth_sat = result_data["problem"]["is_satisfiable"]
                    # Convert ground truth boolean to 0/1 for comparison
                    ground_truth_claim = 1 if ground_truth_sat else 0
                    llm_claim = result_data["llm_response"]["parsed_claim"]

                    # Extract grouping keys (example: by problem params)
                    # TODO: Make grouping configurable via kwargs
                    max_vars = result_data["problem"]["max_vars"]
                    max_len = result_data["problem"]["max_clause_len"]
                    is_horn = result_data["problem"]["is_horn_intended"]
                    group_key = (max_vars, max_len, is_horn)

                    # Update counts for this model and group
                    stats = model_stats[model][group_key]
                    stats['total'] += 1

                    if llm_claim == 2: # Unknown/Parsing Error
                        stats['unknown'] += 1
                    elif llm_claim == ground_truth_claim:
                        stats['correct'] += 1
                        if ground_truth_sat:
                            stats['sat_correct'] += 1
                        else:
                            stats['unsat_correct'] += 1
                    else:
                        # Incorrect claim (but not unknown)
                        pass # Total is incremented, correct is not

                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"Warning: Skipping invalid/malformed line {line_num} in {input_file}: {e}")
                    errors_parsing += 1
                    continue

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return
    except IOError as e:
        print(f"Error reading input file {input_file}: {e}")
        return

    if results_read == 0:
        print("No valid results found in input file. Cannot generate report.")
        return

    # --- Generate Report --- 
    report_lines = []
    report_lines.append("LLM Logic Evaluation Report")
    report_lines.append("=============================")
    report_lines.append(f"Input File: {input_file}")
    report_lines.append(f"Total Results Read: {results_read}")
    if errors_parsing > 0:
        report_lines.append(f"Lines Skipped Due to Errors: {errors_parsing}")
    report_lines.append("\n--- Overall Summary --- (Aggregated across all models and problem types)")

    # Calculate overall totals
    overall_total = 0
    overall_correct = 0
    overall_unknown = 0
    for model, grouped_data in model_stats.items():
        for group, stats in grouped_data.items():
            overall_total += stats['total']
            overall_correct += stats['correct']
            overall_unknown += stats['unknown']

    overall_accuracy = (overall_correct / (overall_total - overall_unknown) * 100) if (overall_total - overall_unknown) > 0 else 0
    report_lines.append(f"  Total Problems Evaluated: {overall_total}")
    report_lines.append(f"  Correct Claims: {overall_correct}")
    report_lines.append(f"  Unknown/Unparsed Claims: {overall_unknown}")
    report_lines.append(f"  Accuracy (Correct / (Total - Unknown)): {overall_accuracy:.2f}%")

    report_lines.append("\n--- Accuracy per Model --- (Aggregated across all problem types)")
    for model in sorted(model_stats.keys()):
        model_total = 0
        model_correct = 0
        model_unknown = 0
        for group, stats in model_stats[model].items():
            model_total += stats['total']
            model_correct += stats['correct']
            model_unknown += stats['unknown']

        model_accuracy = (model_correct / (model_total - model_unknown) * 100) if (model_total - model_unknown) > 0 else 0
        report_lines.append(f"  Model: {model}")
        report_lines.append(f"    Total Evaluated: {model_total}")
        report_lines.append(f"    Correct Claims: {model_correct}")
        report_lines.append(f"    Unknown Claims: {model_unknown}")
        report_lines.append(f"    Accuracy: {model_accuracy:.2f}%\n")

    report_lines.append("\n--- Detailed Accuracy per Model and Problem Type (vars, len, horn) ---")
    # Group by problem type first, then model for better readability
    problem_type_stats = defaultdict(lambda: defaultdict(lambda: Counter()))
    for model, grouped_data in model_stats.items():
         for group_key, stats in grouped_data.items():
             problem_type_stats[group_key][model].update(stats)

    # Sort problem types for consistent output (e.g., by vars, then len, then horn)
    sorted_problem_types = sorted(problem_type_stats.keys())

    for group_key in sorted_problem_types:
        max_vars, max_len, is_horn = group_key
        horn_str = "Horn" if is_horn else "General"
        report_lines.append(f"  Problem Type: Vars={max_vars}, Len={max_len}, Type={horn_str}")

        for model in sorted(problem_type_stats[group_key].keys()):
            stats = problem_type_stats[group_key][model]
            total = stats['total']
            correct = stats['correct']
            unknown = stats['unknown']
            accuracy = (correct / (total - unknown) * 100) if (total - unknown) > 0 else 0
            report_lines.append(
                f"    - {model:<25}: Acc={accuracy:>6.2f}% (Correct: {correct:>3}/{total-unknown:>3}, Unknown: {unknown:>3}, Total: {total:>4})"
            )
        report_lines.append("") # Add blank line between problem types

    # Write report to file
    try:
        with open(report_file, 'w') as f:
            f.write("\n".join(report_lines))
        print(f"Analysis report successfully written to {report_file}")
    except IOError as e:
        print(f"Error writing report file {report_file}: {e}") 