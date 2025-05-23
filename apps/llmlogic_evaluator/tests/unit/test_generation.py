import pytest
from apps.llmlogic_evaluator.generation import (
    truth_table_solve,
    solve_prop_problem,
    normalize_problem, # Import normalize for consistent test input
    solve_prop_horn_problem
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

# ==========================================
# Test cases for solve_prop_horn_problem
# ==========================================

def test_solve_prop_horn_simple_derivation():
    """Test Horn solver: Simple derivation A, -A v B => B."""
    clauses = [[1], [-1, 2]]
    # Input clauses might not be normalized, but the function sorts them internally
    result = solve_prop_horn_problem(clauses)
    assert sorted(result) == [1, 2]

def test_solve_prop_horn_contradiction():
    """Test Horn solver: Contradiction A, -A => Contradiction (0)."""
    clauses = [[1], [-1]]
    result = solve_prop_horn_problem(clauses)
    assert sorted(result) == [0, 1] # Contradiction is represented by 0

def test_solve_prop_horn_no_new_derivation():
    """Test Horn solver: No new units derivable."""
    clauses = [[1], [-2, 3]]
    result = solve_prop_horn_problem(clauses)
    assert sorted(result) == [1] # Only the initial unit

def test_solve_prop_horn_non_horn_input():
    """Test Horn solver: Input is not Horn (A v B). Should derive nothing new."""
    clauses = [[1, 2]]
    result = solve_prop_horn_problem(clauses)
    assert result == [] # No initial units to start propagation

def test_solve_prop_horn_multi_step_derivation():
    """Test Horn solver: Multi-step derivation A, B, -A v -B v C => C."""
    clauses = [[1], [2], [-1, -2, 3]]
    result = solve_prop_horn_problem(clauses)
    assert sorted(result) == [1, 2, 3]

def test_solve_prop_horn_empty_input():
    """Test Horn solver: Empty input."""
    clauses = []
    result = solve_prop_horn_problem(clauses)
    assert result == []

def test_solve_prop_horn_already_satisfied_rule():
    """Test Horn solver: Rule is satisfied by initial units, no new derivation."""
    clauses = [[1], [-1, 2], [2]] # A, -A v B, B
    result = solve_prop_horn_problem(clauses)
    assert sorted(result) == [1, 2] # Should not derive B again


# ==========================================
# Test cases for Helper Functions
# ==========================================

from apps.llmlogic_evaluator.generation import (
    normalize_problem,
    is_tautology,
    is_tautology_set,
    is_horn
)

def test_normalize_problem_sorting_and_deduplication():
    """Test normalize_problem sorts clauses and literals, removes duplicates."""
    problem = [[2, 1], [-1], [1, 2], [-3, -2]]
    expected = [[-1], [-3, -2], [1, 2]] # Sorted by length, then literals; duplicates removed
    normalized = normalize_problem(problem)
    assert normalized == expected

def test_normalize_problem_empty():
    """Test normalize_problem with empty input."""
    assert normalize_problem([]) == []

def test_is_tautology_simple_true():
    """Test is_tautology for a simple tautology (A or -A)."""
    assert is_tautology([1, -1]) is True

def test_is_tautology_simple_false():
    """Test is_tautology for a non-tautological clause."""
    assert is_tautology([1, 2]) is False
    assert is_tautology([-1, -2]) is False
    assert is_tautology([1]) is False

def test_is_tautology_empty():
    """Test is_tautology for an empty clause."""
    assert is_tautology([]) is False

# is_tautology_set works on sets/frozensets, used by solver
def test_is_tautology_set_true():
    """Test is_tautology_set for a simple tautology."""
    assert is_tautology_set(frozenset({1, -1})) is True

def test_is_tautology_set_false():
    """Test is_tautology_set for a non-tautological clause."""
    assert is_tautology_set(frozenset({1, 2})) is False
    assert is_tautology_set(frozenset({-1})) is False

def test_is_horn_true():
    """Test is_horn for valid Horn clauses."""
    assert is_horn([-1, -2, 3]) is True  # Rule
    assert is_horn([-1, -2]) is True     # Goal/Constraint
    assert is_horn([1]) is True          # Fact
    assert is_horn([]) is True           # Empty clause is Horn

def test_is_horn_false():
    """Test is_horn for non-Horn clauses (more than one positive literal)."""
    assert is_horn([1, 2]) is False
    assert is_horn([-1, 2, 3]) is False


# ==========================================
# Test cases for make_prop_problem
# ==========================================

# Need to import the function being tested
from apps.llmlogic_evaluator.generation import make_prop_problem, is_horn

# Use a fixed seed for reproducibility in tests involving randomness
import random
RANDOM_SEED = 42

def test_make_prop_problem_basic_generation():
    """Test basic properties of generated problems."""
    random.seed(RANDOM_SEED)
    varnr = 5
    maxlen = 3
    ratio = 4.0
    hornflag = False
    problem = make_prop_problem(varnr, maxlen, ratio, hornflag)

    assert isinstance(problem, list)
    assert len(problem) > 0 # Should generate some clauses
    assert all(isinstance(clause, list) for clause in problem)

    # Check clause length and variable range
    for clause in problem:
        assert len(clause) <= maxlen
        assert len(clause) > 0 # Clauses shouldn't be empty
        for lit in clause:
            assert isinstance(lit, int)
            assert 1 <= abs(lit) <= varnr
            assert lit != 0

def test_make_prop_problem_horn_flag_true():
    """Test that generated clauses are Horn clauses when hornflag is True."""
    random.seed(RANDOM_SEED)
    varnr = 6
    maxlen = 4
    ratio = 3.0
    hornflag = True
    # Generate multiple times or a larger set to increase confidence
    for _ in range(5): # Run a few times with the same seed reset
        random.seed(RANDOM_SEED + _) # Slight variation seed
        problem = make_prop_problem(varnr, maxlen, ratio, hornflag)
        assert len(problem) > 0
        for clause in problem:
            assert is_horn(clause), f"Clause {clause} is not Horn, but hornflag=True"

def test_make_prop_problem_includes_pos_neg_clauses():
    """Test that the problem includes at least one fully positive and one fully negative clause."""
    random.seed(RANDOM_SEED)
    varnr = 7
    maxlen = 3
    ratio = 5.0 # Higher ratio increases likelihood of diverse clauses
    hornflag = False
    problem = make_prop_problem(varnr, maxlen, ratio, hornflag)

    has_pos = any(all(lit > 0 for lit in clause) for clause in problem)
    has_neg = any(all(lit < 0 for lit in clause) for clause in problem)

    assert has_pos, "Generated problem missing a fully positive clause"
    assert has_neg, "Generated problem missing a fully negative clause"

def test_make_prop_problem_varnr_edge_case():
    """Test make_prop_problem returns empty list if varnr < 2."""
    assert make_prop_problem(1, 3, 4.0, False) == []
    assert make_prop_problem(0, 3, 4.0, False) == []


# ==========================================
# Test cases for makeproof
# ==========================================

from apps.llmlogic_evaluator.generation import makeproof

def test_makeproof_simple_unsat():
    """Test makeproof reconstruction for a simple UNSAT case: (A) & (-A)."""
    # Simulate the output of solve_prop_problem for [[1], [-1]]
    # 1. allcls dictionary populated during the solve run
    allcls_from_run = {
        1: [1, None, frozenset({1})],    # Input clause 1
        2: [2, None, frozenset({-1})],   # Input clause 2
        3: [3, [1, 2], frozenset()]     # Derived empty clause from 1 and 2
    }
    # 2. The final contradiction clause returned by solve_prop_problem
    resolve_res = [3, [1, 2], frozenset()]

    # Expected proof structure after renumbering and sorting
    expected_proof = [
        [1, [], [1]],   # Renumbered clause 1
        [2, [], [-1]],  # Renumbered clause 2
        [3, [1, 2], []] # Renumbered clause 3 (empty clause)
    ]

    actual_proof = makeproof(resolve_res, allcls_from_run)

    assert actual_proof == expected_proof

def test_makeproof_no_proof():
    """Test makeproof returns empty list if solver didn't find contradiction."""
    # Simulate solver returning None (satisfiable)
    resolve_res_none = None
    allcls_from_run = {
        1: [1, None, frozenset({1})]
    }
    assert makeproof(resolve_res_none, allcls_from_run) == []

    # Simulate solver returning a non-empty clause (should not happen for UNSAT)
    resolve_res_non_empty = [1, None, frozenset({1})]
    assert makeproof(resolve_res_non_empty, allcls_from_run) == []

# Add a slightly more complex case later if needed
# def test_makeproof_complex_unsat():
#     # Simulate allcls and resolve_res for [[1, 2], [-1], [-2]]
#     # ... define allcls_from_run and resolve_res ...
#     # ... define expected_proof ...
#     actual_proof = makeproof(resolve_res, allcls_from_run)
#     assert actual_proof == expected_proof 