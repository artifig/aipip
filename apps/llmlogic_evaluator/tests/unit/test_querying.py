import pytest
from apps.llmlogic_evaluator.querying import format_prompt_v1

# ==========================================
# Test cases for format_prompt_v1
# ==========================================

@pytest.fixture
def sample_problem_data():
    """Provides a sample problem data dictionary."""
    return {
        "id": 10,
        "max_vars": 3,
        "max_clause_len": 3,
        "is_horn_intended": False,
        "is_satisfiable": True,
        "problem_clauses": [[-1, 2], [3]],
        "proof_or_model": [ -1, 2, 3 ],
        "horn_derived_units": [3]
    }

def test_format_prompt_v1_structure(sample_problem_data):
    """Test the overall structure and content of the formatted prompt."""
    prompt = format_prompt_v1(sample_problem_data)

    # Check for key sections
    assert "Your task is to solve a problem in propositional logic." in prompt
    assert "Propositional variables are represent as 'pN'" in prompt
    assert "Example 1." in prompt
    assert "Example 2." in prompt
    assert "Statements:" in prompt
    assert "Please think step by step" in prompt

    # Check if clauses are formatted correctly
    assert "p1 is false or p2 is true." in prompt
    assert "p3 is true." in prompt

def test_format_prompt_v1_missing_clauses():
    """Test that it raises ValueError if 'problem_clauses' is missing."""
    bad_data = {"id": 1}
    with pytest.raises(ValueError, match="Problem data missing 'problem_clauses' key."):
        format_prompt_v1(bad_data)


# ==========================================
# Test cases for parse_llm_response
# ==========================================

from apps.llmlogic_evaluator.querying import parse_llm_response

@pytest.mark.parametrize("response_text, expected_claim", [
    # Standard cases
    ("The statements lead to a contradiction.", 0),
    ("Therefore, the set of statements is unsatisfiable. Contradiction", 0),
    ("The analysis shows the statements are satisfiable.", 1),
    ("Conclusion: Satisfiable", 1),
    ("It appears to be consistent. Satisfied", 1),
    ("The model is SAT.", 1),
    ("I cannot determine satisfiability. Unknown", 2),
    ("The result is uncertain.", 2),
    # Case variations
    (" CONTRADICTION", 0),
    (" satisfiable ", 1),
    # Punctuation and formatting
    ("It's unsatisfiable... Contradictory.", 0),
    ("Yes, this seems satisfiable: SAT", 1),
    # Empty/Invalid
    ("", 2),
    ("   ", 2),
    ("No conclusion reached.", 2),
    ("The variables p1, p2 are assigned...", 2),
    # Edge cases with keywords not at the end
    ("A contradiction was found early on, but let me double check... maybe satisfiable?", 1), # SAT keyword present
    ("It seems satisfiable, although some lines point to a contradiction.", 0), # CONTRADICTION keyword present
])
def test_parse_llm_response(response_text, expected_claim):
    """Test parse_llm_response with various inputs."""
    assert parse_llm_response(response_text) == expected_claim


# ==========================================
# Test cases for run_querying
# ==========================================
import json
from unittest.mock import MagicMock, call
from aipip.services.text_generation_service import TextGenerationService # For type hinting
from aipip.providers.interfaces.text_provider import CompletionResponse
from apps.llmlogic_evaluator.querying import run_querying

# Define mock problem data as JSON strings for input file
MOCK_PROBLEM_1_STR = json.dumps({
    "id": 1,
    "max_vars": 2,
    "problem_clauses": [[1], [-1]],
    "is_satisfiable": False
})
MOCK_PROBLEM_2_STR = json.dumps({
    "id": 2,
    "max_vars": 2,
    "problem_clauses": [[1, 2]],
    "is_satisfiable": True
})

# Define mock service responses
MOCK_RESPONSE_MODEL_A_P1 = CompletionResponse(
    text="Definitely contradiction.", provider_name="mock_provider_a", metadata={}
)
MOCK_RESPONSE_MODEL_B_P1 = CompletionResponse(
    text="I think it's unsatisfiable.", provider_name="mock_provider_b", metadata={}
)
MOCK_RESPONSE_MODEL_A_P2 = CompletionResponse(
    text="Looks satisfiable to me!", provider_name="mock_provider_a", metadata={}
)
MOCK_RESPONSE_MODEL_B_P2 = CompletionResponse(
    text="The model is SAT.", provider_name="mock_provider_b", metadata={}
)

@pytest.fixture
def mock_text_generation_service() -> MagicMock:
    """Fixture for a mocked TextGenerationService."""
    service = MagicMock(spec=TextGenerationService)

    # Configure side effects based on model/prompt (or simpler logic)
    def generate_side_effect(*args, **kwargs):
        model = kwargs.get('model')
        prompt = kwargs.get('prompt')
        # Rough check based on prompt content (assuming unique enough)
        if "p1 is true.\np1 is false." in prompt:
            if model == "model_a": return MOCK_RESPONSE_MODEL_A_P1
            if model == "model_b": return MOCK_RESPONSE_MODEL_B_P1
        elif "p1 is true or p2 is true." in prompt:
            if model == "model_a": return MOCK_RESPONSE_MODEL_A_P2
            if model == "model_b": return MOCK_RESPONSE_MODEL_B_P2
        # Default fallback or error
        return CompletionResponse(text="Unknown mock input", provider_name="mock_fallback", metadata={})

    service.generate.side_effect = generate_side_effect
    return service

def test_run_querying_workflow(tmp_path, mock_text_generation_service):
    """Test the main run_querying workflow with mocked service and files."""
    # Arrange: Create dummy input file
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    input_file = input_dir / "problems.jsonl"
    input_file.write_text(MOCK_PROBLEM_1_STR + "\n" + MOCK_PROBLEM_2_STR + "\n")

    # Arrange: Define output path
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output_file = output_dir / "results.jsonl"

    # Arrange: Define models and parameters
    models_providers = [("mock_provider_a", "model_a"), ("mock_provider_b", "model_b")]
    gen_params = {"temperature": 0.6}

    # Act: Run the querying function
    run_querying(
        input_file=str(input_file),
        output_file=str(output_file),
        service=mock_text_generation_service,
        model_provider_list=models_providers,
        **gen_params
    )

    # Assert: Check service calls
    assert mock_text_generation_service.generate.call_count == 4 # 2 problems * 2 models
    # Example check for one call (more specific checks can be added)
    # Note: Checking prompts exactly can be brittle
    mock_text_generation_service.generate.assert_any_call(
        provider_name="mock_provider_a",
        model="model_a",
        prompt=format_prompt_v1(json.loads(MOCK_PROBLEM_1_STR)), # Check formatted prompt
        temperature=0.6
    )
    mock_text_generation_service.generate.assert_any_call(
        provider_name="mock_provider_b",
        model="model_b",
        prompt=format_prompt_v1(json.loads(MOCK_PROBLEM_2_STR)), # Check formatted prompt
        temperature=0.6
    )

    # Assert: Check output file content
    assert output_file.exists()
    results = []
    with open(output_file, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))

    assert len(results) == 4 # 2 problems * 2 models

    # Check structure and content of one result (e.g., problem 1, model B)
    result_p1_mb = next((r for r in results if r["problem"]["id"] == 1 and r["query_info"]["model"] == "model_b"), None)
    assert result_p1_mb is not None
    assert result_p1_mb["problem"]["problem_clauses"] == [[1], [-1]]
    assert result_p1_mb["query_info"]["provider"] == "mock_provider_b"
    assert result_p1_mb["query_info"]["generation_params"] == gen_params
    assert result_p1_mb["llm_response"]["text"] == "I think it's unsatisfiable."
    assert result_p1_mb["llm_response"]["parsed_claim"] == 0 # UNSAT

    # Check structure and content of another result (e.g., problem 2, model A)
    result_p2_ma = next((r for r in results if r["problem"]["id"] == 2 and r["query_info"]["model"] == "model_a"), None)
    assert result_p2_ma is not None
    assert result_p2_ma["problem"]["problem_clauses"] == [[1, 2]]
    assert result_p2_ma["query_info"]["provider"] == "mock_provider_a"
    assert result_p2_ma["llm_response"]["text"] == "Looks satisfiable to me!"
    assert result_p2_ma["llm_response"]["parsed_claim"] == 1 # SAT 