import pytest
import json
from collections import Counter
from apps.llmlogic_evaluator.analysis import run_analysis

# ==========================================
# Test Fixtures
# ==========================================

@pytest.fixture
def mock_results_data():
    """Provides a list of mock result dictionaries."""
    return [
        # Problem 1 (SAT: True, Claim: 1 (Correct))
        {"problem": {"id": 1, "max_vars": 3, "max_clause_len": 3, "is_horn_intended": True, "is_satisfiable": True}, "query_info": {"model": "model-A", "provider": "mock-provider-a"}, "llm_response": {"parsed_claim": 1}},
        # Problem 1 (SAT: True, Claim: 0 (Incorrect))
        {"problem": {"id": 1, "max_vars": 3, "max_clause_len": 3, "is_horn_intended": True, "is_satisfiable": True}, "query_info": {"model": "model-B", "provider": "mock-provider-b"}, "llm_response": {"parsed_claim": 0}},
        # Problem 2 (SAT: False, Claim: 0 (Correct))
        {"problem": {"id": 2, "max_vars": 3, "max_clause_len": 3, "is_horn_intended": True, "is_satisfiable": False}, "query_info": {"model": "model-A", "provider": "mock-provider-a"}, "llm_response": {"parsed_claim": 0}},
        # Problem 2 (SAT: False, Claim: 1 (Incorrect))
        {"problem": {"id": 2, "max_vars": 3, "max_clause_len": 3, "is_horn_intended": True, "is_satisfiable": False}, "query_info": {"model": "model-B", "provider": "mock-provider-b"}, "llm_response": {"parsed_claim": 1}},
        # Problem 3 (SAT: True, Claim: 2 (Unknown))
        {"problem": {"id": 3, "max_vars": 4, "max_clause_len": 3, "is_horn_intended": False, "is_satisfiable": True}, "query_info": {"model": "model-A", "provider": "mock-provider-a"}, "llm_response": {"parsed_claim": 2}},
        # Problem 3 (SAT: True, Claim: 1 (Correct))
        {"problem": {"id": 3, "max_vars": 4, "max_clause_len": 3, "is_horn_intended": False, "is_satisfiable": True}, "query_info": {"model": "model-B", "provider": "mock-provider-b"}, "llm_response": {"parsed_claim": 1}},
        # Problem 4 (SAT: False, Claim: 0 (Correct))
        {"problem": {"id": 4, "max_vars": 4, "max_clause_len": 3, "is_horn_intended": False, "is_satisfiable": False}, "query_info": {"model": "model-A", "provider": "mock-provider-a"}, "llm_response": {"parsed_claim": 0}},
        # Problem 4 (SAT: False, Claim: 0 (Correct))
        {"problem": {"id": 4, "max_vars": 4, "max_clause_len": 3, "is_horn_intended": False, "is_satisfiable": False}, "query_info": {"model": "model-B", "provider": "mock-provider-b"}, "llm_response": {"parsed_claim": 0}},
    ]

@pytest.fixture
def create_mock_results_file(tmp_path, mock_results_data):
    """Creates a mock results file in JSON Lines format."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    input_file = input_dir / "results.jsonl"
    with open(input_file, 'w') as f:
        for result in mock_results_data:
            json.dump(result, f)
            f.write('\n')
    return input_file

# ==========================================
# Test cases for run_analysis
# ==========================================

def test_run_analysis_produces_report(tmp_path, create_mock_results_file):
    """Test that run_analysis creates an output report file."""
    input_file = create_mock_results_file
    report_file = tmp_path / "report.txt"

    run_analysis(input_file=str(input_file), report_file=str(report_file))

    assert report_file.exists()
    assert report_file.read_text() != "" # Check it's not empty

def test_run_analysis_overall_accuracy(tmp_path, create_mock_results_file):
    """Test the calculation of overall accuracy."""
    # Based on mock_results_data:
    # Total = 8, Unknown = 1
    # Correct = 5 (P1/A, P2/A, P3/B, P4/A, P4/B)
    # Expected Accuracy = 5 / (8 - 1) * 100 = 5/7 * 100 = 71.43%
    input_file = create_mock_results_file
    report_file = tmp_path / "report.txt"

    run_analysis(input_file=str(input_file), report_file=str(report_file))

    report_content = report_file.read_text()
    assert "Overall Summary" in report_content
    assert "Total Problems Evaluated: 8" in report_content
    assert "Correct Claims: 5" in report_content
    assert "Unknown/Unparsed Claims: 1" in report_content
    assert "Accuracy (Correct / (Total - Unknown)): 71.43%" in report_content

def test_run_analysis_provider_model_summary(tmp_path, create_mock_results_file):
    """Test the hierarchical provider/model accuracy summary section."""
    input_file = create_mock_results_file
    report_file = tmp_path / "report.txt"

    run_analysis(input_file=str(input_file), report_file=str(report_file))

    report_content = report_file.read_text()
    assert "Detailed Accuracy per Model and Problem Type" in report_content

    # Check Type (3, 3, True) - Recreate string with exact formatting
    assert "Problem Type: Vars=3, Len=3, Type=Horn" in report_content
    model_a_str_3_3_t = f"    - {'model-A':<25}: Acc={100.00:>6.2f}% (Correct: {2:>3}/{2-0:>3}, Unknown: {0:>3}, Total: {2:>4})"
    model_b_str_3_3_t = f"    - {'model-B':<25}: Acc={0.00:>6.2f}% (Correct: {0:>3}/{2-0:>3}, Unknown: {0:>3}, Total: {2:>4})"
    assert model_a_str_3_3_t in report_content
    assert model_b_str_3_3_t in report_content

    # Check Type (4, 3, False) - Recreate string with exact formatting
    assert "Problem Type: Vars=4, Len=3, Type=General" in report_content
    model_a_str_4_3_f = f"    - {'model-A':<25}: Acc={100.00:>6.2f}% (Correct: {1:>3}/{2-1:>3}, Unknown: {1:>3}, Total: {2:>4})"
    model_b_str_4_3_f = f"    - {'model-B':<25}: Acc={100.00:>6.2f}% (Correct: {2:>3}/{2-0:>3}, Unknown: {0:>3}, Total: {2:>4})"
    assert model_a_str_4_3_f in report_content
    assert model_b_str_4_3_f in report_content

def test_run_analysis_empty_input(tmp_path):
    """Test run_analysis with an empty input file."""
    input_file = tmp_path / "empty_results.jsonl"
    input_file.touch()
    report_file = tmp_path / "report.txt"

    run_analysis(input_file=str(input_file), report_file=str(report_file))

    # Report should not be generated, check if it exists and is empty or contains error
    assert not report_file.exists() or report_file.read_text() == ""
    # Check console output? (Harder in unit test, maybe check logs if implemented)

def test_run_analysis_malformed_input(tmp_path):
    """Test run_analysis with a malformed JSON line."""
    input_file = tmp_path / "malformed_results.jsonl"
    report_file = tmp_path / "report.txt"
    # Add the missing 'provider' key to the valid JSON line
    valid_json_data = {
        "problem": {"id": 1, "max_vars": 3, "max_clause_len": 3, "is_horn_intended": True, "is_satisfiable": True},
        "query_info": {"model": "model-A", "provider": "mock-provider-a"}, # Added provider
        "llm_response": {"parsed_claim": 1}
    }
    input_file.write_text("this is not json\n" + json.dumps(valid_json_data)+"\n")

    run_analysis(input_file=str(input_file), report_file=str(report_file))

    report_content = report_file.read_text()
    assert "Lines Skipped Due to Errors: 1" in report_content
    assert "Total Results Read: 1" in report_content # Only one line was valid
    assert "Overall Summary" in report_content # Ensure report was still generated 