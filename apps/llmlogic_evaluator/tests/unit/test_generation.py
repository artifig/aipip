import pytest
from apps.llmlogic_evaluator.generation import truth_table_solve

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