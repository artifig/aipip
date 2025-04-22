import pytest
from apps.llmlogic_evaluator.generation import (
    truth_table_solve,
    solve_prop_problem,
    normalize_problem # Import normalize for consistent test input
)

# Test cases for truth_table_solve

def test_truth_table_solve_simple_sat():
    """Test truth_table_solve with a simple satisfiable formula: (A)."""
    clauses = [[1]]
    # Expected: [model, trace_string]. We only care about the model part.
    # Model should contain 1.
    result = truth_table_solve(clauses)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] is not False # Should have found a model
    assert isinstance(result[0], list)
    assert 1 in result[0]

def test_truth_table_solve_simple_unsat():
    """Test truth_table_solve with a simple unsatisfiable formula: (A) and (-A)."""
    clauses = [[1], [-1]]
    # Expected: [False, trace_string].
    result = truth_table_solve(clauses)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] is False # Should not have found a model

def test_truth_table_solve_multiple_vars_sat():
    """Test truth_table_solve with a satisfiable multi-variable formula: (A or B). One model is A=T, B=F -> [1, -2]"""
    clauses = [[1, 2]] # (A or B)
    result = truth_table_solve(clauses)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] is not False
    assert isinstance(result[0], list)
    # Check if the found model satisfies the clause
    model = result[0]
    model_set = set(model)
    satisfied = False
    for clause in clauses:
        clause_satisfied = False
        for lit in clause:
            if lit in model_set:
                clause_satisfied = True
                break
        if not clause_satisfied: # Should not happen for SAT
             pytest.fail(f"Model {model} does not satisfy clause {clause}")
    assert True # If loop completes, model is valid

def test_truth_table_solve_multiple_vars_unsat():
    """Test truth_table_solve with an unsatisfiable multi-variable formula: (A or B) and (-A) and (-B)."""
    clauses = [[1, 2], [-1], [-2]]
    result = truth_table_solve(clauses)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] is False

def test_truth_table_solve_too_many_vars():
    """Test that truth_table_solve returns None for > 20 variables."""
    # Create a dummy clause list with variable 21
    clauses = [[21]]
    result = truth_table_solve(clauses)
    assert result is None

# ==========================================
# Test cases for solve_prop_problem
# ==========================================

# Helper to check if result indicates unsatisfiable (empty clause found)
def is_unsat_result(result):
    return isinstance(result, list) and len(result) == 3 and result[2] == frozenset()

@pytest.mark.parametrize("strategy", ['original', 'standard'])
def test_solve_prop_problem_simple_sat(strategy):
    """Test solve_prop_problem with a simple satisfiable formula: (A)."""
    clauses = normalize_problem([[1]])
    result = solve_prop_problem(clauses, resolution_strategy=strategy)
    assert result is None # Should return None for satisfiable

@pytest.mark.parametrize("strategy", ['original', 'standard'])
def test_solve_prop_problem_simple_unsat(strategy):
    """Test solve_prop_problem with a simple unsatisfiable formula: (A) and (-A)."""
    clauses = normalize_problem([[1], [-1]])
    result = solve_prop_problem(clauses, resolution_strategy=strategy)
    assert is_unsat_result(result) # Should return empty clause structure

@pytest.mark.parametrize("strategy", ['original', 'standard'])
def test_solve_prop_problem_multi_var_sat(strategy):
    """Test solve_prop_problem with a satisfiable multi-variable formula: (A or B)."""
    clauses = normalize_problem([[1, 2]])
    result = solve_prop_problem(clauses, resolution_strategy=strategy)
    assert result is None

@pytest.mark.parametrize("strategy", ['original', 'standard'])
def test_solve_prop_problem_multi_var_unsat(strategy):
    """Test solve_prop_problem with an unsatisfiable multi-variable formula: (A or B) and (-A) and (-B)."""
    clauses = normalize_problem([[1, 2], [-1], [-2]])
    result = solve_prop_problem(clauses, resolution_strategy=strategy)
    assert is_unsat_result(result)

@pytest.mark.parametrize("strategy", ['original', 'standard'])
def test_solve_prop_problem_tautology_input(strategy):
    """Test solve_prop_problem with input containing a tautology: (A or -A). Should be SAT."""
    # Note: The solver pre-filters tautologies, but let's test the case
    clauses = normalize_problem([[1, -1]])
    result = solve_prop_problem(clauses, resolution_strategy=strategy)
    # Depending on implementation details, it might find it satisfiable (None)
    # or potentially error if not handled robustly. Expect None.
    assert result is None

@pytest.mark.parametrize("strategy", ['original', 'standard'])
def test_solve_prop_problem_empty_input(strategy):
    """Test solve_prop_problem with empty input. Should be SAT."""
    clauses = []
    result = solve_prop_problem(clauses, resolution_strategy=strategy)
    assert result is None

# Example of a slightly more complex case (still simple)
# (A v B) & (-B v C) & (-C) & (-A)
@pytest.mark.parametrize("strategy", ['original', 'standard'])
def test_solve_prop_problem_complex_unsat(strategy):
    """Test solve_prop_problem with a slightly more complex unsatisfiable formula."""
    clauses = normalize_problem([[1, 2], [-2, 3], [-3], [-1]])
    result = solve_prop_problem(clauses, resolution_strategy=strategy)
    assert is_unsat_result(result) 